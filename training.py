import random
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

from models.unet import UNetVgg16
from models.SEMwithMEM import SMNet2D as Model
from utils.data_process import CDNet2014Dataset
from utils.transforms import IterativeCustomCompose
from utils.evaluate.losses import CDNet2014_JaccardLoss as Loss
from utils.evaluate.accuracy import calculate_acc_metrics as acc_func
from utils.evaluate.eval_utils import (
    ACC_NAMES,
    LOSS_NAMES,
    ORDER_NAMES,
    SummaryRecord,
    BasicRecord,
    OneEpochVideosAccumulator,
    EvalMeasure,
)
from submodules.UsefulFileTools.WordOperator import str_format
from submodules.UsefulFileTools.PickleOperator import load_pickle

PROJECT_DIR = str(Path(__file__).resolve())


def get_device(id: int = 0):
    return torch.device(f'cuda:{id}' if torch.cuda.is_available() else 'cpu')


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
        self.best_records[self.loss_idx :] = 100

        self.summaries: List[SummaryRecord] = []  # [train_summary, val_summary, test_summary]

    def create_measure_table(self):
        measure_table = Table(show_header=True, header_style='bold magenta')

        measure_table.add_column("", style="dim")
        [measure_table.add_column(name, justify='right') for name in ORDER_NAMES]

        return measure_table

    def testing(self, dataset: CDNet2014Dataset | Dataset):
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
                    videos_accumulator.pixelLevel_matrix[-2] += loss.to('cpu')  # pixelLevel loss is different with others
                    videos_accumulator.pixelLevel_matrix[-1] += 1  # accumulative_times += 1
            self.summaries[-1].records(videos_accumulator)

    def validating(self, loader: DataLoader):
        videos_accumulator = OneEpochVideosAccumulator()

        self.model.eval()
        with torch.no_grad():
            video_id: torch.IntTensor
            features: torch.Tensor
            frames: torch.Tensor
            labels: torch.Tensor
            for video_id, frames, labels, features in tqdm(loader):
                self.proposed_training_method(video_id, features, frames, labels, videos_accumulator, self.test_transforms)

            self.summaries[1].records(videos_accumulator)

    def training(
        self,
        num_epoch: int,
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
                self.summaries.append(SummaryRecord(writer, summary_path, num_epoch, mode=name))

        isStop = False
        for self.epoch in range(num_epoch):
            BasicRecord.row_id = self.epoch
            CDNet2014Dataset.next_frame_gap(self.epoch)
            measure_table = self.create_measure_table()
            videos_accumulator = OneEpochVideosAccumulator()
            isBest = False

            self.model.train()

            video_id: torch.IntTensor
            features: torch.Tensor
            frames: torch.Tensor
            labels: torch.Tensor
            for video_id, frames, labels, features in tqdm(loader):
                loss = self.proposed_training_method(video_id, features, frames, labels, videos_accumulator, self.train_transforms)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.summaries[0].records(videos_accumulator)

            measure_table.add_row('Train', *[f'{l:.3e}' for l in self.summaries[0].pixelLevel.last_scores])

            for i, (data_info, tasking, name) in enumerate(zip(data_infos, [self.validating, self.testing], ['Val', 'Test'])):
                if data_info is None:
                    continue

                tasking(data_info)
                measure_table.add_row(name, *[f'{l:.3e}' for l in self.summaries[-1].pixelLevel.last_scores])

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
            if self.best_epoch:
                save_path = f'loss-{best_loss_records[-1]:.3e}_F1-{best_acc_record[3]:.3f}'
                isStop = early_stop == (self.epoch - self.best_epoch)
            else:
                save_path = f'loss-{self.summaries[0].overall.last_scores[-1]:.3e}_acc-{self.summaries[0].overall.last_scores[3]:.3f}'

            save_path_heads: List[str] = []
            if isCheckpoint:
                save_path_heads.append(f'checkpoint_e{self.epoch:03}')
            if isBest:
                save_path_heads.extend(
                    [f'bestLoss-{name}' for name, is_best in zip(LOSS_NAMES, best_loss_checker) if is_best],
                )
                save_path_heads.extend(
                    [f'bestAcc-{name}' for name, is_best in zip(ACC_NAMES, best_acc_checker) if is_best],
                )

            isStop += self.epoch + 1 == num_epoch
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
                path.symlink_to(epoch_path)
                print(f"symlink: {str_format(str(path_head), fore='y'):<36} -> {epoch_path}")

            if isStop:
                print(str_format("Stop!!", fore='y'))
                break

    def proposed_training_method(
        self,
        video_id: torch.Tensor,
        features: torch.Tensor,
        frames: torch.Tensor,
        labels: torch.Tensor,
        videos_accumulator: OneEpochVideosAccumulator,
        transforms: IterativeCustomCompose,
    ):
        video_id = video_id.to(self.device).unsqueeze(1)
        features = features.to(self.device)
        frames = frames.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            frames, labels, features = transforms(frames, labels, features)

        bg_only_imgs = features[:, 0].unsqueeze(1)
        for step in range(frames.shape[1]):
            frame, label = frames[:, step], labels[:, step]

            pred, frame, features = self.model(frame, features, bg_only_imgs)
            loss: torch.Tensor = self.loss_func(pred, label)

            with torch.no_grad():
                bg_only_imgs, pred_mask = self.get_bgOnly_and_mask(frame, pred)
                videos_accumulator.accumulate(self.eval_measure(label, pred, pred_mask, video_id))
                videos_accumulator.pixelLevel_matrix[-2] += loss.to('cpu')  # pixelLevel loss is different with others
                videos_accumulator.pixelLevel_matrix[-1] += 1  # accumulative_times += 1

        return loss

    def get_bgOnly_and_mask(self, frame: torch.Tensor, pred: torch.Tensor):
        pred_mask = torch.where(pred > self.eval_measure.thresh, 1, 0).type(dtype=torch.int32)
        bg_only_imgs = frame * (1 - pred_mask)
        return bg_only_imgs, pred_mask

    def save(self, model: Model | nn.Module, path: str, isFull: bool = False):
        if isFull:
            torch.save(model, f'{path}.pt')
            torch.save(self.optimizer, f'{path}_Optimizer.pickle')
        else:
            torch.save(model.state_dict(), f'{path}.pt')
            torch.save(self.optimizer.state_dict(), f'{path}_Optimizer.pickle')


if __name__ == '__main__':
    import time

    from torch import optim
    from torchvision import transforms
    from utils.data_process import get_data_SetAndLoader, DatasetConfig
    from utils.transforms import RandomCrop, RandomResizedCrop, CustomCompose
    from submodules.UsefulFileTools.FileOperator import check2create_dir

    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    #! ========== Hyperparameter ==========
    # * Datasets
    BATCH_SIZE = 9
    NUM_WORKERS = 8
    CV_SET = 2
    DATA_SPLIT_RATE = 1.0

    NUM_EPOCH = 200
    EARLY_STOP = 20
    CHECKPOINT = 10
    DO_TESTING = False

    #! ========== Augmentation ==========
    sizeHW = (224, 224)
    train_trans_cpu = CustomCompose(
        [
            # transforms.ToTensor(), # already converted in the __getitem__()
            transforms.RandomChoice(
                [
                    RandomCrop(crop_size=sizeHW, p=1.0),
                    # RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
                ]
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_trans_cpu = transforms.Compose(
        [
            # transforms.ToTensor(), # already converted in the __getitem__()
            transforms.Resize(sizeHW, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    argumentation_order_ls = [
        # transforms.GaussianBlur([3, 3]),
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, hue=0.2, contrast=0.5, saturation=0.2)], p=0.75),
    ]

    train_iter_compose = IterativeCustomCompose([*argumentation_order_ls], target_size=sizeHW)
    test_iter_compose = IterativeCustomCompose([], target_size=sizeHW)

    #! ========== Datasets ==========
    dataset_cfg = DatasetConfig()
    dataset_cfg.num_epoch = NUM_EPOCH

    train_loader, val_loader, test_set = get_data_SetAndLoader(
        dataset_cfg=dataset_cfg,
        cv_set=CV_SET,
        dataset_rate=DATA_SPLIT_RATE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        train_transforms_cpu=train_trans_cpu,
        test_transforms_cpu=test_trans_cpu,
        label_isShadowFG=False,
        useTestAsVal=True,
    )

    #! ========== Network ==========

    se_model = UNetVgg16(9, 6)
    me_model = UNetVgg16(9, 1)
    sm2d_net = Model(se_model, me_model, useStandardNorm4Features=True).to(DEVICE)
    optimizer = optim.Adam(sm2d_net.parameters(), lr=1e-4)
    loss_func = Loss(reduce='mean')

    model_name = f'{sm2d_net.__class__.__name__}({se_model.__class__.__name__}-{me_model.__class__.__name__})'
    optimizer_name = f'{optimizer.__class__.__name__}-{optimizer.defaults["lr"]:.1e}'
    saveDir = f'out/{time.strftime("%m%d-%H%M")}_{model_name}_{optimizer_name}_{str(loss_func)}_BS-{BATCH_SIZE}'
    check2create_dir(saveDir)

    #! ========== Train Process ==========
    model_process = DL_Model(sm2d_net, optimizer, train_iter_compose, test_iter_compose, device=DEVICE, loss_func=loss_func)
    model_process.training(
        NUM_EPOCH, train_loader, val_loader, test_set if DO_TESTING else None, Path(saveDir), EARLY_STOP, CHECKPOINT
    )
