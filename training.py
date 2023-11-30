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

from models.unet import unet_vgg16 as FEModel
from models.unet import unet_vgg16 as MEModel
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
        FE_model: nn.Module | FEModel,
        ME_model: nn.Module | MEModel,
        optimizer: optim.Optimizer,
        train_transforms: IterativeCustomCompose,
        test_transforms: IterativeCustomCompose = None,
        acc_func: Callable = acc_func,
        loss_func: Callable = Loss(),
        eval_measure: EvalMeasure | nn.Module = EvalMeasure(0.5, Loss(reduction='none')),
        device: str = get_device(),
        is3dModel: bool = True,
    ) -> None:
        self.FE_model = FE_model
        self.ME_model = ME_model
        self.optimizer = optimizer
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.acc_func = acc_func
        self.loss_func = loss_func
        self.eval_measure = eval_measure
        self.device = device
        self.is3dModel = is3dModel

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

        self.FE_model.eval()
        self.ME_model.eval()

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
                    if torch.isnan(frame).any():
                        aa = 0
                    if i != 0:
                        frame, label, _ = self.test_transforms(frame, label, None)
                    else:
                        frame, label, features = self.test_transforms(frame, label, features)
                        bg_only_img = features[:, 0].unsqueeze(1)

                    if torch.isnan(frame).any():
                        aa = 0

                    combine_features = torch.hstack((features, bg_only_img))
                    combine_features: torch.Tensor
                    if torch.isnan(combine_features).any():
                        aa = 0
                    if not self.is3dModel:
                        frame = frame.squeeze(1)
                        if combine_features.dim() == 5:
                            combine_features = combine_features.reshape(
                                combine_features.shape[0],
                                combine_features.shape[1] * combine_features.shape[2],
                                *combine_features.shape[3:],
                            )

                    features = self.FE_model(combine_features)
                    if torch.isnan(features).any():
                        aa = 0

                    # std, mean = torch.std_mean(features, dim=0)
                    mean = features.mean(dim=0, keepdim=True)
                    std = features.std(dim=0, unbiased=False, keepdim=True)
                    features = (features - mean) / (std + 0.0001)
                    if torch.isnan(features).any():
                        aa = 0

                    combine_features = torch.hstack((features, frame))
                    pred: torch.Tensor = self.ME_model(combine_features)
                    loss: torch.Tensor = self.loss_func(pred, label)

                    if torch.isnan(pred).any():
                        aa = 0

                    pred_mask = torch.where(pred > self.eval_measure.thresh, 1, 0).type(dtype=torch.int32)
                    bg_only_img = frame * (1 - pred_mask)
                    videos_accumulator.accumulate(self.eval_measure(label, pred, pred_mask, video_id))
                    videos_accumulator.pixelLevel_matrix[-2] += loss.to('cpu')  # pixelLevel loss is different with others
                    videos_accumulator.pixelLevel_matrix[-1] += 1  # accumulative_times += 1
            self.summaries[-1].records(videos_accumulator)

    def validating(self, loader: DataLoader):
        videos_accumulator = OneEpochVideosAccumulator()

        self.FE_model.eval()
        self.ME_model.eval()

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
        summary_path = f'{saveDir}/summary'
        writer = SummaryWriter(summary_path)
        self.summaries = [
            SummaryRecord(writer, summary_path, num_epoch) for data in [train_loader, val_loader, test_set] if data is not None
        ]
        for data, name in zip([train_loader, val_loader, test_set], ['Train', 'Val', 'Test']):
            if data is not None:
                self.summaries.append(SummaryRecord(writer, summary_path, num_epoch, mode=name))

        best_acc_record, best_loss_records = self.best_records[: self.loss_idx], self.best_records[self.loss_idx :]

        isStop = False
        for self.epoch in range(num_epoch):
            BasicRecord.row_id = self.epoch
            measure_table = self.create_measure_table()
            videos_accumulator = OneEpochVideosAccumulator()

            isBest = False

            self.FE_model.train()
            self.ME_model.train()

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

            data_infos = [val_loader, test_set]
            checker_active_idx = 1 - data_infos.count(None)  # best record priority: test > val
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
                    self.save(self.FE_model, self.ME_model, str(saveDir / epoch_path))
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
            with torch.no_grad():
                frame, label = frames[:, step], labels[:, step]
                combine_features = torch.hstack((features, bg_only_imgs))

                if not self.is3dModel:
                    frame = frame.squeeze(1)
                    if combine_features.dim() == 5:
                        combine_features = combine_features.reshape(
                            combine_features.shape[0],
                            combine_features.shape[1] * combine_features.shape[2],
                            *combine_features.shape[3:],
                        )

            features = self.FE_model(combine_features)
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, unbiased=False, keepdim=True)
            features = (features - mean) / (std + 0.0001)
            if torch.isnan(features).any():
                a = 0

            combine_features = torch.hstack((features, frame))
            pred: torch.Tensor = self.ME_model(combine_features)
            loss: torch.Tensor = self.loss_func(pred, label)
            if torch.isnan(pred).any():
                a = 0

            with torch.no_grad():
                pred_mask = torch.where(pred > self.eval_measure.thresh, 1, 0).type(dtype=torch.int32)
                bg_only_imgs = frame * (1 - pred_mask)
                videos_accumulator.accumulate(self.eval_measure(label, pred, pred_mask, video_id))
                videos_accumulator.pixelLevel_matrix[-2] += loss.to('cpu')  # pixelLevel loss is different with others
                videos_accumulator.pixelLevel_matrix[-1] += 1  # accumulative_times += 1

        return loss

    def save(self, fe_model: FEModel | nn.Module, me_model: MEModel | nn.Module, path: str, isFull: bool = False):
        if isFull:
            torch.save((fe_model, me_model), f'{path}.pt')
            torch.save(self.optimizer, f'{path}_Optimizer.pickle')
        else:
            torch.save((fe_model.state_dict(), me_model.state_dict()), f'{path}.pt')
            torch.save(self.optimizer.state_dict(), f'{path}_Optimizer.pickle')


if __name__ == '__main__':
    import time

    from torch import optim
    from torchvision import transforms
    from utils.data_process import get_dataLoaders_and_testSet, DatasetConfig
    from utils.transforms import RandomCrop, RandomResizedCrop, CustomCompose
    from submodules.UsefulFileTools.FileOperator import check2create_dir

    sizeHW = (224, 224)
    train_trans_cpu = CustomCompose(
        [
            # transforms.ToTensor(), # already converted in the __getitem__()
            transforms.RandomChoice(
                [
                    RandomCrop(crop_size=sizeHW, p=1.0),
                    RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
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

    BATCH_SIZE = 12
    fe_model = FEModel(9, 6).to('cuda:0')
    me_model = MEModel(9, 1).to('cuda:0')
    optimizer = optim.Adam(list(fe_model.parameters()) + list(me_model.parameters()), lr=0.0001)

    train_loader, val_loader, test_set = get_dataLoaders_and_testSet(
        dataset_cfg=DatasetConfig(),
        cv_set=5,
        dataset_rate=0.7,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        train_transforms_cpu=train_trans_cpu,
        test_transforms_cpu=test_trans_cpu,
        label_isShadowFG=False,
    )

    saveDir = f'out/{time.strftime("%m%d-%H%M")}_{fe_model.__class__.__name__}-{me_model.__class__.__name__}_BS-{BATCH_SIZE}'
    check2create_dir(saveDir)

    model_process = DL_Model(fe_model, me_model, optimizer, train_iter_compose, test_iter_compose, device='cuda:0', is3dModel=False)
    model_process.training(3, train_loader, val_loader, test_set, saveDir=Path(saveDir), early_stop=3, checkpoint=3)
