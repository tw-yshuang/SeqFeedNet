import re, sys, time, random
from pathlib import Path
from typing import Callable, Generator, List, Tuple

from tqdm import tqdm
from rich.table import Table
from rich.console import Console
from rich.progress import track
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.unet import *
from models.Unet3D import *
from models.SEMwithMEM import *
from utils.transforms import *
from utils.evaluate.losses import *
from models.unet import UNetVgg16 as BackBone
from models.SEMwithMEM import SMNet2D as Model
from utils.ResultOperator import ResultOperator
from utils.data_process import CDNet2014Dataset
from utils.data_process import get_data_LoadersAndSet, DatasetConfig
from utils.transforms import RandomCrop, CustomCompose, IterativeCustomCompose
from utils.evaluate.losses import IOULoss4CDNet2014 as Loss
from utils.evaluate.accuracy import calculate_acc_metrics as acc_func
from utils.evaluate.eval_utils import (
    ACC_NAMES,
    LOSS_NAMES,
    ORDER_NAMES,
    EvalMeasure,
    BasicRecord,
    SummaryRecord,
    OneEpochVideosAccumulator,
)
from submodules.UsefulFileTools.WordOperator import str_format
from submodules.UsefulFileTools.FileOperator import check2create_dir

PROJECT_DIR = str(Path(__file__).resolve())


def get_device(id: int = 0):
    device = 'cpu'
    if torch.cuda.is_available():
        device = f'cuda:{id}'
    else:
        print('CUDA device not found, use CPU!!')

    return torch.device(device)


class Processor:
    def __init__(
        self,
        model: nn.Module | Model,
        optimizer: optim.Optimizer,
        train_transforms: IterativeCustomCompose,
        test_transforms: IterativeCustomCompose = None,
        acc_func: Callable = acc_func,
        loss_func: Callable = Loss(),
        eval_measure: EvalMeasure | nn.Module = EvalMeasure(0.5, Loss(reduction='none')),
        device: str = get_device(),
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.acc_func = acc_func
        self.loss_func = loss_func
        self.eval_measure = eval_measure
        self.device = device

        self.console = Console()
        self.epoch = 0
        self.best_epoch = 0
        self.best_records = torch.zeros(len(ORDER_NAMES), dtype=torch.float32) * 0

        self.loss_idx = -len(LOSS_NAMES)
        self.best_records[self.loss_idx :] = 1e5

        self.summaries: List[SummaryRecord] = []  # [train_summary, val_summary, test_summary]

    def create_measure_table(self):
        measure_table = Table(show_header=True, header_style='bold magenta')

        measure_table.add_column(f"e{self.epoch:03}", style='dim')
        [measure_table.add_column(name, justify='right') for name in ORDER_NAMES]

        return measure_table

    def fix_testing_size(self, dataset: CDNet2014Dataset | Dataset, idx: int):
        if idx == len(dataset.data_infos):
            return

        hw = dataset.data_infos[idx][1].ROI_mask.shape[-2:]
        if str(self.model.me_model) == 'UNetVgg16':
            h, w = int(hw[0] // 16 * 16), int(hw[1] // 16 * 16)  # for fix the UNet concat dimension problem
        else:
            h, w = int(hw[0]), int(hw[1])
        dataset.transforms_cpu.transforms[0] = transforms.Resize((h, w), antialias=True)
        self.test_transforms.target_size = (h, w)

    def testing(self, saveDir: str, dataset: CDNet2014Dataset | Dataset, saveResult: bool = False):
        summaryRecord = SummaryRecord(saveDir, 1, None, self.acc_func, mode='Test')
        self.__testing(dataset, summaryRecord=summaryRecord, saveResult=saveResult, saveDir=saveDir)
        summaryRecord.export2csv()

    def __testing(
        self,
        dataset: CDNet2014Dataset | Dataset,
        summaryRecord: SummaryRecord | None = None,
        saveResult: bool = False,
        saveDir: str = 'out',
    ):
        if summaryRecord is None:
            summaryRecord = self.summaries[-1]
        videos_accumulator = OneEpochVideosAccumulator()

        self.model.eval()
        with torch.no_grad():
            video_id: int
            features: torch.Tensor
            frame: torch.Tensor
            label: torch.Tensor
            test_iter: Generator[Tuple[torch.Tensor, torch.Tensor]]

            self.fix_testing_size(dataset, 0)

            if saveResult:
                result_opt = ResultOperator(
                    dataset.data_infos[0], sizeHW=dataset.data_infos[0][1].ROI_mask.shape[-2:], taskDir=saveDir
                )
            else:
                result_opt = lambda *args: None

            for next_idx, (video_id, features, test_iter) in enumerate(track(dataset, "Test Video Processing..."), 1):
                video_id = torch.tensor(video_id).to(self.device).reshape(1, 1)
                features = features.to(self.device).unsqueeze(0)

                for i, (frame, empty_frame, label) in enumerate(test_iter):
                    frame, label = frame.to(self.device).unsqueeze(1), label.to(self.device)
                    empty_frame = empty_frame.to(self.device).unsqueeze(1)
                    label = label.to(self.device).unsqueeze(1)

                    if i != 0:
                        frame, empty_frame, label, _ = self.test_transforms(frame, empty_frame, label, None)
                    else:
                        frame, empty_frame, label, features = self.test_transforms(frame, empty_frame, label, features)
                        bg_only_img = features[:, 0]
                        features = self.model.erd_model(features)

                    pred, frame, features = self.model(frame, empty_frame, features, bg_only_img)
                    loss: torch.Tensor = self.loss_func(pred, label)

                    bg_only_img, pred_mask = self.get_bgOnly_and_mask(frame, pred)
                    videos_accumulator.batchLevel_matrix[-2] += loss.to('cpu')  # batchLevel loss is different with others
                    videos_accumulator.accumulate(self.eval_measure(label, pred, pred_mask, video_id))

                    result_opt(pred_mask, pred, features)
                if saveResult and next_idx != len(dataset.data_infos):
                    result_opt = ResultOperator(
                        dataset.data_infos[next_idx], sizeHW=dataset.data_infos[next_idx][1].ROI_mask.shape[-2:], taskDir=saveDir
                    )
                self.fix_testing_size(dataset, next_idx)

            summaryRecord.records(videos_accumulator)

    def __validating(self, loader: DataLoader):
        videos_accumulator = OneEpochVideosAccumulator()

        self.model.eval()
        with torch.no_grad():
            self.proposed_training_method(loader, videos_accumulator, self.test_transforms, isTrain=False)
            self.summaries[1].records(videos_accumulator)

    def training(
        self,
        num_epochs: int,
        loader: DataLoader,
        val_loader: DataLoader = None,
        test_set: CDNet2014Dataset = None,
        saveDir: Path = PROJECT_DIR,
        early_stop: int = 50,
        checkpoint: int = 20,
        *args,
        **kwargs,
    ):
        # best record create
        best_acc_record, best_loss_records = self.best_records[: self.loss_idx], self.best_records[self.loss_idx :]
        data_infos = [val_loader, test_set]
        checker_active_idx = 1 - data_infos.count(None)  # best record priority: test > val

        # Summary created
        summary_path = f'{saveDir}/summary'
        writer = SummaryWriter(summary_path)
        for data, name in zip([loader, val_loader, test_set], ['Train', 'Val', 'Test']):
            if data is not None:
                self.summaries.append(SummaryRecord(summary_path, num_epochs, writer, self.acc_func, mode=name))

        isStop = False
        for self.epoch in range(num_epochs):
            BasicRecord.update_row_id(self.epoch)
            CDNet2014Dataset.update_frame_gap(self.epoch)
            measure_table = self.create_measure_table()
            videos_accumulator = OneEpochVideosAccumulator()
            isBest = False

            self.model.train()
            self.proposed_training_method(loader, videos_accumulator, self.train_transforms, isTrain=True)
            self.summaries[0].records(videos_accumulator)

            measure_table.add_row('Train', *[f'{l:.3e}' for l in self.summaries[0].batchLevel.last_scores])

            for i, (data_info, tasking, name) in enumerate(zip(data_infos, [self.__validating, self.__testing], ['Val', 'Test'])):
                if data_info is None:
                    continue

                tasking(data_info)
                measure_table.add_row(name, *[f'{l:.3e}' for l in self.summaries[-1].batchLevel.last_scores])

                if i != checker_active_idx:
                    continue
                best_loss_checker = best_loss_records > self.summaries[-1].overall.last_scores[self.loss_idx :]
                best_loss_records[best_loss_checker] = self.summaries[-1].overall.last_scores[self.loss_idx :][best_loss_checker]

                best_acc_checker = best_acc_record < self.summaries[-1].overall.last_scores[: self.loss_idx]
                best_acc_record[best_acc_checker] = self.summaries[-1].overall.last_scores[: self.loss_idx][best_acc_checker]

                if best_acc_checker.any() or best_loss_checker.any():
                    self.best_epoch = self.epoch
                    isBest = True

            self.console.print(measure_table)

            # * Save Stage
            isCheckpoint = self.epoch % checkpoint == 0
            isStop = early_stop == (self.epoch - self.best_epoch) or num_epochs == (self.epoch + 1)

            save_path_heads: List[str] = []
            save_path = f'loss-{self.summaries[-1].overall.last_scores[-1]:.3e}_F1-{self.summaries[-1].overall.last_scores[3]:.3f}'
            if isCheckpoint:
                save_path_heads.append(f'checkpoint_e{self.epoch:03}')
            if isBest:
                save_path_heads.extend(
                    [f'bestLoss-{name}' for name, is_best in zip(LOSS_NAMES, best_loss_checker) if is_best],
                )
                save_path_heads.extend(
                    [f'bestAcc-{name}' for name, is_best in zip(ACC_NAMES, best_acc_checker) if is_best],
                )

            if isStop:
                save_path_heads.append(f'final_e{self.epoch:03}_')

            for i, path_head in enumerate(save_path_heads):
                if i == 0:
                    epoch_path = f'e{self.epoch:03}_{save_path}'
                    self.save(self.model, str(saveDir / epoch_path))
                    print(f"Save Model: {str_format(str(epoch_path), fore='g')}")
                    [summary.export2csv() for summary in self.summaries]

                path: Path = saveDir / f'{path_head}.pt'
                path.unlink(missing_ok=True)
                path.symlink_to(f'{epoch_path}.pt')
                print(f"symlink: {str_format(str(path_head), fore='y'):<36} -> {epoch_path}")

            if isStop:
                print(str_format("Stop!!", fore='y'))
                break

    def proposed_training_method(
        self,
        loader: DataLoader,
        videos_accumulator: OneEpochVideosAccumulator,
        transforms: IterativeCustomCompose,
        isTrain: bool = True,
    ):
        video_id: torch.IntTensor
        features: torch.Tensor
        frames: torch.Tensor
        empty_frames: torch.Tensor
        labels: torch.Tensor
        for video_id, frames, empty_frames, labels, features in tqdm(loader):
            video_id = video_id.to(self.device).unsqueeze(1)
            features = features.to(self.device)
            frames = frames.to(self.device)
            empty_frames = empty_frames.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                frames, empty_frames, labels, features = transforms(frames, empty_frames, labels, features)

            bg_only_imgs = features[:, 0]
            features = self.model.erd_model(features)
            # losses = torch.zeros(frames.shape[1], dtype=torch.float32, device=self.device)
            step_noDetachMEM = frames.shape[1] - 1
            for step in range(frames.shape[1]):
                isDetachMEM = 1 - (step // step_noDetachMEM)
                # isDetachMEM = 0
                frame, empty_frame, label = frames[:, step], empty_frames[:, step], labels[:, step]

                # features = features.detach()  # create a new tensor to detach previous computational graph
                pred: torch.Tensor
                frame: torch.Tensor
                pred, frame, features = self.model(frame, empty_frame, features, bg_only_imgs, isDetachMEM)
                loss: torch.Tensor = self.loss_func(pred, label)

                if isTrain:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                with torch.no_grad():
                    bg_only_imgs, pred_mask = self.get_bgOnly_and_mask(frame, pred)
                    if step > 2:
                        videos_accumulator.batchLevel_matrix[-2] += loss.item()  # batchLevel loss is different with others
                        videos_accumulator.accumulate(self.eval_measure(label, pred, pred_mask, video_id))

    def get_bgOnly_and_mask(self, frame: torch.Tensor, pred: torch.Tensor):
        pred_mask = torch.where(pred > self.eval_measure.thresh, 1, 0).type(dtype=torch.int32)
        bg_only_imgs = frame * (1 - pred)
        return bg_only_imgs, pred_mask

    def save(self, model: Model | nn.Module, path: str, isFull: bool = False):
        if isFull:
            torch.save(model, f'{path}.pt')
            torch.save(self.optimizer, f'{path}_{self.optimizer.__class__.__name__}.pt')
        else:
            torch.save(model.state_dict(), f'{path}.pt')
            torch.save(self.optimizer.state_dict(), f'{path}_{self.optimizer.__class__.__name__}.pt')

    @staticmethod
    def load(path: str, model: nn.Module = None, optimizer: optim.Optimizer = None, isFull: bool = False, device: str = 'cpu'):
        model: nn.Module
        optimizer: optim.Optimizer

        path = str(Path(path).resolve())
        optimizer_path = f'{path[:-3]}_{optimizer.__class__.__name__}.pt'
        if isFull:
            model = torch.load(path)
            optimizer = torch.load(optimizer_path)
        else:
            model.load_state_dict(torch.load(path, map_location=device))

            if Path(optimizer_path).exists():
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

        return model, optimizer


class Parser:
    SE_Net: nn.Module | BackBone
    ME_Net: nn.Module | BackBone
    SM_Net: nn.Module | Model
    DEVICE: int

    LOSS: nn.Module | Loss
    OPTIMIZER: torch.optim
    LEARNING_RATE: float

    BATCH_SIZE: int
    NUM_EPOCHS: int
    NUM_WORKERS: int
    CV_SET: int
    SIZE_HW: Tuple[int]
    DATA_SPLIT_RATE: float

    useTestAsVal: bool
    DO_TESTING: bool
    testFromBegin: bool
    saveTestResult: bool

    PRETRAIN_WEIGHT: str
    OUT: str


def convertStr2Parser(
    se_network: str = 'UNetVgg16',
    me_network: str = 'UNetVgg16',
    sm_network: str = 'SMNet2D',
    loss_func: str = 'IOULoss4CDNet2014',
    optimizer: str = '',
    learning_rate: float = 1e-4,
    weight_decay: float = 0,
    num_epochs: int = 0,
    batch_size: int = 8,
    num_workers: int = 1,
    cv_set_number: int = 1,
    img_sizeHW: str = '224-224',
    data_split_rate: float = 1.0,
    use_test_as_val: bool = False,
    device: int = 0,
    do_testing: bool = False,
    test_from_begin: bool = True,
    save_test_result: bool = False,
    pretrain_weight: str = '',
    output: str = '',
):
    '''
    Args
        se_network: "Sequence Extract Network",
        me_network: "Mask Extract Network",
        sm_network: "Sequence to mask Network",
        loss_func: "Please check utils/evaluate/losses.py to find others",
        optimizer: "Optimizer that provide by Pytorch",
        learning_rate: "Learning Rate for optimizer",
        weight_decay: "Weight Decay for optimizer"
        num_epochs: "Number of epochs",
        batch_size: "Number of batch_size",
        num_workers: "Number of workers for data processing",
        cv_set_number: "Cross validation set number for training and test videos will be selected",
        img_sizeHW: "Image size for training",
        data_split_rate: "Split data to train_set & val_set",
        use_test_as_val: "Use test_data as validation data, use this flag will set '--data_split_rate=1.0'",
        device: "CUDA ID, if system can not find Nvidia GPU, it will use CPU",
        do_testing: "Do testing evaluation is a time-consuming process, suggest not do it",
        test_from_begin: "Do testing evaluation from beginning"
        save_test_result: "Save testing all the result"
        pretrain_weight: "Pretrain weight, model structure must same with the setting",
        output: "Model output directory"
    '''
    parser = Parser()
    module_locate = sys.modules[__name__]

    parser.OUT = output
    parser.DEVICE = get_device(device)
    #! ========== Network ==========
    parser.PRETRAIN_WEIGHT = pretrain_weight
    if pretrain_weight != '':
        sm_network, nets = pretrain_weight.split('/')[-2].split('_')[-5].split('.')
        se_network, me_network = nets.split('-')
    parser.SE_Net: nn.Module | BackBone = getattr(module_locate, se_network)
    parser.ME_Net: nn.Module | BackBone = getattr(module_locate, me_network)
    parser.SM_Net: nn.Module | Model = getattr(module_locate, sm_network)

    #! ========== Hyperparameter ==========
    if optimizer == '':
        if pretrain_weight == '':
            optimizer = 'Adam'
        else:
            optimizer = re.match(r'^[A-Za-z]+', pretrain_weight.split('/')[-2].split('_')[-4]).group(0)

    parser.OPTIMIZER: optim = getattr(optim, optimizer)
    parser.LEARNING_RATE = learning_rate
    parser.WEIGHT_DECAY = weight_decay
    parser.LOSS: nn.Module | Loss = getattr(module_locate, loss_func)
    parser.NUM_EPOCHS = num_epochs
    parser.BATCH_SIZE = batch_size

    #! ========== Dataset ==========
    parser.NUM_WORKERS = num_workers
    parser.CV_SET = cv_set_number
    parser.DO_TESTING = do_testing
    parser.testFromBegin = test_from_begin
    parser.saveTestResult = save_test_result
    parser.useTestAsVal = use_test_as_val
    parser.SIZE_HW = tuple(map(int, img_sizeHW.split('-')))

    if use_test_as_val is True:
        parser.DATA_SPLIT_RATE = 1.0
    else:
        parser.DATA_SPLIT_RATE = data_split_rate

    return parser


def execute(parser: Parser):
    # random.seed(42)
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)
    #! ========== Hyperparameter ==========
    EARLY_STOP = -1
    CHECKPOINT = 10

    #! ========== Augmentation ==========
    train_trans_cpu = CustomCompose(
        [
            transforms.RandomChoice(
                [
                    RandomCrop(crop_size=parser.SIZE_HW, p=1.0),
                    RandomShiftedCrop(parser.SIZE_HW, max_shift=5, p=1.0),
                    RandomResizedCrop(parser.SIZE_HW, scale=(0.6, 1.8), ratio=(3.0 / 5.0, 2.0), p=0.9),
                    PTZZoomCrop(parser.SIZE_HW, overlap_time=10, max_pixelMove=5, p4targets=0.75, p4others=0.9),
                    PTZPanCrop(parser.SIZE_HW, overlap_time=10, max_pixelMoveH=3, max_pixelMoveW=3, p4targets=0.75, p4others=0.9),
                ],
                p=(0.25, 0.25, 0.25, 0.25 * 0.25, 0.25 * 0.75),
            ),
            # AdditiveColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.075, p=0.9),
            # GaussianNoise(sigma=(0, 0.01)),
            # # RandomHorizontalFlip(0.5),
            # # RandomVerticalFlip(0.5),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_trans_cpu = transforms.Compose(
        [
            transforms.Resize(parser.SIZE_HW, antialias=True),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_trans_onGPU_ls = [
        AdditiveColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.075, p=0.9),
        GaussianNoise(sigma=(0, 0.01)),
        RandomHorizontalFlip(0.5),
        # RandomVerticalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    test_trans_onGPU_ls = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    train_iter_compose = IterativeCustomCompose(train_trans_onGPU_ls, target_size=parser.SIZE_HW)
    test_iter_compose = IterativeCustomCompose(test_trans_onGPU_ls, target_size=parser.SIZE_HW)

    #! ========== Datasets ==========
    dataset_cfg = DatasetConfig()
    dataset_cfg.num_epochs = parser.NUM_EPOCHS

    train_loader, val_loader, test_set = get_data_LoadersAndSet(
        dataset_cfg=dataset_cfg,
        cv_set=parser.CV_SET,
        dataset_rate=parser.DATA_SPLIT_RATE,
        batch_size=parser.BATCH_SIZE,
        num_workers=parser.NUM_WORKERS,
        pin_memory=True,
        train_transforms_cpu=train_trans_cpu,
        test_transforms_cpu=test_trans_cpu,
        label_isShadowFG=False,
        useTestAsVal=parser.useTestAsVal,
        onlyTest=parser.NUM_EPOCHS == 0,
        testFromBegin=parser.testFromBegin,
    )

    #! ========== Network ==========
    se_model: nn.Module = parser.SE_Net(4, 3) if '3D' in parser.SE_Net.__name__ else parser.SE_Net(12, 9)
    me_model: nn.Module = parser.ME_Net(15, 1)
    sm_net: nn.Module = parser.SM_Net(se_model, me_model).to(parser.DEVICE)
    optimizer: optim.Optimizer = parser.OPTIMIZER(sm_net.parameters(), lr=parser.LEARNING_RATE, weight_decay=parser.WEIGHT_DECAY)
    loss_func: nn.Module = parser.LOSS(reduction='mean')

    #! ========== Load Pretrain ==========
    if parser.PRETRAIN_WEIGHT != '':
        sm_net, optimizer = Processor.load(parser.PRETRAIN_WEIGHT, sm_net, optimizer, device=parser.DEVICE)

    #! ========= Create saveDir ==========
    split_id = parser.PRETRAIN_WEIGHT.rfind('/') + 1
    saveDir = 'out/' if parser.PRETRAIN_WEIGHT == '' else parser.PRETRAIN_WEIGHT[:split_id]
    if parser.NUM_EPOCHS == 0 and parser.DO_TESTING:  # only testing
        saveDir += f'{parser.OUT}_' if parser.OUT != '' else ''
        path = Path(parser.PRETRAIN_WEIGHT)
        if path.is_symlink():
            saveDir += f'{parser.PRETRAIN_WEIGHT[split_id:].split("_")[0]}_'
            path = str(path.readlink())
        else:
            path = parser.PRETRAIN_WEIGHT[split_id:]
        saveDir += path.split('_')[0]
    else:
        model_name = f'{sm_net.__class__.__name__}.{se_model.__class__.__name__}-{me_model.__class__.__name__}'
        optimizer_name = f'{optimizer.__class__.__name__}{optimizer.defaults["lr"]:.1e}.wd{parser.WEIGHT_DECAY}'
        saveDir += f'{time.strftime("%m%d-%H%M")}_{parser.OUT}_{model_name}_{optimizer_name}_{str(loss_func)}_BS-{parser.BATCH_SIZE}_Set-{parser.CV_SET}'

    check2create_dir(saveDir)

    #! ========== Model Setting ==========
    processor = Processor(
        sm_net,
        optimizer,
        train_iter_compose,
        test_iter_compose,
        device=parser.DEVICE,
        loss_func=loss_func,
        eval_measure=EvalMeasure(0.5, parser.LOSS(reduction='none')),
    )

    #! ========== Testing Evaluation ==========
    if parser.NUM_EPOCHS == 0 and parser.DO_TESTING:
        print(str_format("Testing Evaluate!!", fore='y'))
        processor.testing(saveDir, test_set, parser.saveTestResult)
        exit()

    #! ========== Training Process ==========
    processor.training(
        parser.NUM_EPOCHS,
        train_loader,
        val_loader,
        test_set if parser.DO_TESTING else None,
        Path(saveDir),
        EARLY_STOP,
        CHECKPOINT,
    )


if __name__ == '__main__':
    parser = convertStr2Parser()
    execute(parser)
