import random
from pathlib import Path
from typing import Callable, Generator, List, Tuple

import click
from tqdm import tqdm
from rich.table import Table
from rich.console import Console
from rich.progress import track
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.unet import UNetVgg16 as BackBone
from models.SEMwithMEM import SMNet2D as Model
from utils.data_process import CDNet2014Dataset
from utils.transforms import IterativeCustomCompose
from utils.evaluate.losses import FocalLoss4CDNet2014 as Loss
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

PROJECT_DIR = str(Path(__file__).resolve())


def get_device(id: int = 0):
    device = 'cpu'
    if torch.cuda.is_available():
        device = f'cuda:{id}'
    else:
        print('CUDA device not found, use CPU!!')

    return torch.device(device)


class DL_Model:
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

    def testing(self, saveDir: str, dataset: CDNet2014Dataset | Dataset):
        summaryRecord = SummaryRecord(saveDir, 1, None, self.acc_func, mode='Test')
        self.__testing(dataset, summaryRecord=summaryRecord)
        summaryRecord.export2csv()

    def __testing(self, dataset: CDNet2014Dataset | Dataset, summaryRecord: SummaryRecord | None = None):
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
            for video_id, features, test_iter in track(dataset, "Test Video Processing..."):
                video_id = torch.tensor(video_id).to(self.device).reshape(1, 1)
                features = features.to(self.device).unsqueeze(0)

                for i, (frame, label) in enumerate(test_iter):
                    frame, label = frame.to(self.device).unsqueeze(1), label.to(self.device).unsqueeze(1)
                    if i != 0:
                        frame, label, _ = self.test_transforms(frame, label, None)
                    else:
                        frame, label, features = self.test_transforms(frame, label, features)
                        bg_only_img = features[:, 0].unsqueeze(1)

                    pred, frame, features = self.model(frame, features, bg_only_img)
                    loss: torch.Tensor = self.loss_func(pred, label)

                    bg_only_img, pred_mask = self.get_bgOnly_and_mask(frame, pred)
                    videos_accumulator.accumulate(self.eval_measure(label, pred, pred_mask, video_id))
                    videos_accumulator.batchLevel_matrix[-2] += loss.to('cpu')  # batchLevel loss is different with others
                    videos_accumulator.batchLevel_matrix[-1] += 1  # accumulative_times += 1

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
        for data, name in zip([train_loader, val_loader, test_set], ['Train', 'Val', 'Test']):
            if data is not None:
                self.summaries.append(SummaryRecord(summary_path, num_epochs, writer, self.acc_func, mode=name))

        isStop = False
        for self.epoch in range(num_epochs):
            BasicRecord.row_id = self.epoch
            CDNet2014Dataset.next_frame_gap(self.epoch)
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
        labels: torch.Tensor
        for video_id, frames, labels, features in tqdm(loader):
            video_id = video_id.to(self.device).unsqueeze(1)
            features = features.to(self.device)
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                frames, labels, features = transforms(frames, labels, features)

            bg_only_imgs = features[:, 0].unsqueeze(1)
            for step in range(frames.shape[1]):
                frame, label = frames[:, step], labels[:, step]

                features = features.detach()  # create a new tensor to detach previous computational graph
                pred, frame, features = self.model(frame, features, bg_only_imgs)
                loss: torch.Tensor = self.loss_func(pred, label)

                if isTrain:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                with torch.no_grad():
                    bg_only_imgs, pred_mask = self.get_bgOnly_and_mask(frame, pred)
                    videos_accumulator.batchLevel_matrix[-2] += loss.item()  # pixelLevel loss is different with others
                    videos_accumulator.accumulate(self.eval_measure(label, pred, pred_mask, video_id))

    def get_bgOnly_and_mask(self, frame: torch.Tensor, pred: torch.Tensor):
        pred_mask = torch.where(pred > self.eval_measure.thresh, 1, 0).type(dtype=torch.int32)
        bg_only_imgs = frame * (1 - pred_mask)
        return bg_only_imgs, pred_mask

    def save(self, model: Model | nn.Module, path: str, isFull: bool = False):
        if isFull:
            torch.save(model, f'{path}.pt')
            torch.save(self.optimizer, f'{path}_{self.optimizer.__class__.__name__}.pickle')
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

    useStandardNorm4Features: bool
    useTestAsVal: bool
    DO_TESTING: bool

    PRETRAIN_WEIGHT: str
    OUT: str


def get_parser():
    help_doc = {
        'se_network': "Sequence Extract Network",
        'me_network': "Mask Extract Network",
        'sm_network': "Sequence to mask Network",
        'use-standard-normal': "Use standard normalization for se_model output",
        'loss_func': "Please check utils/evaluate/losses.py to find others",
        'optimizer': "Optimizer that provide by Pytorch",
        'learning_rate': "Learning Rate for optimizer",
        'num_epochs': "Number of epochs",
        'batch_size': "Number of batch_size",
        'num_workers': "Number of workers for data processing",
        'cv_set_number': "Cross validation set number for training and test videos will be selected",
        'img_sizeHW': "Image size for training",
        'data_split_rate': "Split data to train_set & val_set",
        'use_test_as_val': "Use test_data as validation data, use this flag will set '--data_split_rate=1.0'",
        'device': "CUDA ID, if system can not find Nvidia GPU, it will use CPU",
        'do_testing': "Do testing evaluation is a time-consuming process, suggest not do it",
        'pretrain_weight': "Pretrain weight, model structure must same with the setting",
        'output': "Model output directory",
    }

    @click.command(context_settings=dict(help_option_names=['-h', '--help'], max_content_width=120))
    @click.option('-se', '--se_network', default='UNetVgg16', help=help_doc['se_network'])
    @click.option('-me', '--me_network', default='UNetVgg16', help=help_doc['me_network'])
    @click.option('-sm', '--sm_network', default='SMNet2D', help=help_doc['sm_network'])
    @click.option('-use-std', '--use-standard-normal', default=False, is_flag=True, help=help_doc['use-standard-normal'])
    @click.option('-loss', '--loss_func', default='FocalLoss4CDNet2014', help=help_doc['loss_func'])
    @click.option('-opt', '--optimizer', default='Adam', help=help_doc['optimizer'])
    @click.option('-lr', '--learning_rate', default=1e-4, help=help_doc['learning_rate'])
    @click.option('-epochs', '--num_epochs', default=0, help=help_doc['num_epochs'])
    @click.option('-bs', '--batch_size', default=8, help=help_doc['batch_size'])
    @click.option('-workers', '--num_workers', default=1, help=help_doc['num_workers'])
    @click.option('-cv', '--cv_set_number', default=1, help=help_doc['cv_set_number'])
    @click.option('-imghw', '--img_sizeHW', 'img_sizeHW', default='224-224', help=help_doc['img_sizeHW'])
    @click.option('-drate', '--data_split_rate', default=1.0, help=help_doc['data_split_rate'])
    @click.option('-use-t2val', '--use_test_as_val', default=False, is_flag=True, help=help_doc['use_test_as_val'])
    @click.option('--device', default=0, help=help_doc['device'])
    @click.option('--do_testing', default=False, is_flag=True, help=help_doc['do_testing'])
    @click.option('--pretrain_weight', default='', help=help_doc['pretrain_weight'])
    @click.option('-out', '--output', default='', help=help_doc['output'])
    def cli(
        se_network: str,
        me_network: str,
        sm_network: str,
        use_standard_normal: bool,
        loss_func: str,
        optimizer: str,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        num_workers: int,
        cv_set_number: int,
        img_sizeHW: str,
        data_split_rate: float,
        use_test_as_val: bool,
        device: int,
        do_testing: bool,
        pretrain_weight: str,
        output: str,
    ):
        parser = Parser()
        module_locate = sys.modules[__name__]

        parser.DEVICE = get_device(device)
        parser.OUT = f'_{output}' if output != '' else ''
        #! ========== Network ==========
        parser.SE_Net: nn.Module | BackBone = getattr(module_locate, se_network)
        parser.ME_Net: nn.Module | BackBone = getattr(module_locate, me_network)
        parser.SM_Net: nn.Module | Model = getattr(module_locate, sm_network)
        parser.useStandardNorm4Features = use_standard_normal
        parser.PRETRAIN_WEIGHT = pretrain_weight

        #! ========== Hyperparameter ==========
        parser.LOSS: nn.Module | Loss = getattr(module_locate, loss_func)
        parser.OPTIMIZER: optim = getattr(optim, optimizer)
        parser.LEARNING_RATE = learning_rate
        parser.NUM_EPOCHS = num_epochs
        parser.BATCH_SIZE = batch_size

        #! ========== Dataset ==========
        parser.NUM_WORKERS = num_workers
        parser.CV_SET = cv_set_number
        parser.DO_TESTING = do_testing
        parser.useTestAsVal = use_test_as_val
        parser.SIZE_HW = tuple(map(int, img_sizeHW.split('-')))

        if use_test_as_val is True:
            parser.DATA_SPLIT_RATE = 1.0
        else:
            parser.DATA_SPLIT_RATE = data_split_rate

        return parser

    parser: Parser = cli(standalone_mode=False)
    if '-h' in sys.argv or '--help' in sys.argv:
        exit()
    return parser


if __name__ == '__main__':
    import sys, time

    from torch import optim
    from torchvision import transforms
    from utils.data_process import get_data_SetAndLoader, DatasetConfig
    from utils.transforms import RandomCrop, RandomResizedCrop, CustomCompose
    from submodules.UsefulFileTools.FileOperator import check2create_dir

    from models.SEMwithMEM import SMNet2D as Model
    from utils.data_process import CDNet2014Dataset
    from utils.transforms import IterativeCustomCompose
    from utils.evaluate.losses import FocalLoss4CDNet2014 as Loss

    from models.unet import *
    from models.SEMwithMEM import *
    from utils.evaluate.losses import *

    # sys.argv = 'training.py --device 2 -epochs 2 --batch_size 8 -workers 8 -cv 5 -imghw 224-224 -use-t2val -out test'.split()

    # sys.argv = "training.py --device 1 -epochs 0 --batch_size 8 -workers 8 -cv 5 -imghw 224-224 -use-t2val -out test -opt SGD --pretrain_weight out/1203-1703_SMNet2D(UNetVgg16-UNetVgg16)_Adam1.0e-04_FocalLoss_BS-9_Set-2_lastBack/bestAcc-F_score.pt".split()

    # sys.argv = "training.py --device 1 -epochs 0 --batch_size 8 -workers 8 -cv 5 -imghw 224-224 -use-t2val -out test -opt SGD --pretrain_weight out/1203-1703_SMNet2D(UNetVgg16-UNetVgg16)_Adam1.0e-04_FocalLoss_BS-9_Set-2_lastBack/bestAcc-F_score.pt --do_testing".split()

    parser = get_parser()

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    #! ========== Hyperparameter ==========
    EARLY_STOP = 30
    CHECKPOINT = 10

    #! ========== Augmentation ==========
    train_trans_cpu = CustomCompose(
        [
            # transforms.ToTensor(), # already converted in the __getitem__()
            transforms.RandomChoice(
                [
                    RandomCrop(crop_size=parser.SIZE_HW, p=1.0),
                    # RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
                ]
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_trans_cpu = transforms.Compose(
        [
            # transforms.ToTensor(), # already converted in the __getitem__()
            transforms.Resize(parser.SIZE_HW, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    argumentation_order_ls = [
        # transforms.GaussianBlur([3, 3]),
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, hue=0.2, contrast=0.5, saturation=0.2)], p=0.75),
    ]

    train_iter_compose = IterativeCustomCompose([*argumentation_order_ls], target_size=parser.SIZE_HW)
    test_iter_compose = IterativeCustomCompose([], target_size=parser.SIZE_HW)

    #! ========== Datasets ==========
    dataset_cfg = DatasetConfig()
    dataset_cfg.num_epochs = parser.NUM_EPOCHS

    train_loader, val_loader, test_set = get_data_SetAndLoader(
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
    )

    #! ========== Network ==========
    se_model: nn.Module = parser.SE_Net(9, 6)
    me_model: nn.Module = parser.ME_Net(9, 1)
    sm_net: nn.Module = parser.SM_Net(se_model, me_model, useStandardNorm4Features=parser.useStandardNorm4Features).to(parser.DEVICE)
    optimizer: optim = parser.OPTIMIZER(sm_net.parameters(), lr=parser.LEARNING_RATE)
    loss_func: nn.Module = parser.LOSS(reduction='mean')

    #! ========== Load Pretrain ==========
    if parser.PRETRAIN_WEIGHT != '':
        sm_net, optimizer = DL_Model.load(parser.PRETRAIN_WEIGHT, sm_net, optimizer, device=parser.DEVICE)

    #! ========= Create saveDir ==========
    parent_dir = 'out' if parser.PRETRAIN_WEIGHT == '' else parser.PRETRAIN_WEIGHT[: parser.PRETRAIN_WEIGHT.rfind('/')]
    model_name = f'{sm_net.__class__.__name__}.{se_model.__class__.__name__}-{me_model.__class__.__name__}'
    optimizer_name = f'{optimizer.__class__.__name__}{optimizer.defaults["lr"]:.1e}'
    saveDir = f'{parent_dir}/{time.strftime("%m%d-%H%M")}{parser.OUT}_{model_name}_{optimizer_name}_{str(loss_func)}_BS-{parser.BATCH_SIZE}_Set-{parser.CV_SET}'
    check2create_dir(saveDir)

    #! ========== Model Setting ==========
    model_process = DL_Model(
        sm_net,
        optimizer,
        train_iter_compose,
        test_iter_compose,
        device=parser.DEVICE,
        loss_func=loss_func,
        eval_measure=EvalMeasure(0.5, Loss(reduction='none')),
    )

    #! ========== Testing Evaluation ==========
    if parser.NUM_EPOCHS == 0 and parser.DO_TESTING:
        print(str_format("Testing Evaluate!!", fore='y'))
        model_process.testing(saveDir, test_set)
        exit()

    #! ========== Training Process ==========
    model_process.training(
        parser.NUM_EPOCHS,
        train_loader,
        val_loader,
        test_set if parser.DO_TESTING else None,
        Path(saveDir),
        EARLY_STOP,
        CHECKPOINT,
    )
