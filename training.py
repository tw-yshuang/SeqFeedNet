import random
from pathlib import Path
from typing import Callable, List

from tqdm import tqdm
from rich.table import Table
from rich.console import Console
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.unet import unet_vgg16 as FEModel
from models.unet import unet_vgg16 as MEModel
from utils.data_process import CDNet2014Dataset
from utils.transforms import IterativeCustomCompose
from utils.evaluate.losses import CDNet2014_JaccardLoss as Loss
from utils.evaluate.accuracy import calculate_acc_metrics as acc_func
from utils.evaluate.eval_utils import ORDER_NAMES, SummaryRecord, BasicRecord, OneEpochVideosAccumulation, EvalMeasure
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
        train_transforms: IterativeCustomCompose,
        test_transforms: IterativeCustomCompose = None,
        acc_func: Callable = acc_func,
        loss_func: Callable = Loss(),
        eval_measure: EvalMeasure | nn.Module = EvalMeasure(0.5, Loss(reduction='none')),
        device: str = get_device(),
    ) -> None:
        self.FE_model = FE_model
        self.ME_model = ME_model
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
        self.best_records[-1] = 100

        self.train_summary: SummaryRecord
        self.val_summary: SummaryRecord
        self.test_summary: SummaryRecord

    def create_measure_table(self):
        measure_table = Table(show_header=True, header_style='bold magenta')

        measure_table.add_column("", style="dim")
        [measure_table.add_column(name, justify='right') for name in ORDER_NAMES]

        return measure_table

    def testing(self, loader: DataLoader):
        num_iter = 0
        self.model.eval()
        loss_record = torch.zeros_like(self.best_loss_record)
        acc_record = torch.zeros_like(self.best_acc_record)
        with torch.no_grad():
            for data, label, hit_idxs, isHits, _ in tqdm(loader):
                data, label = data.to(self.device), label.to(self.device)

                batch_coordXYs = torch.stack(
                    [label[:, self.model_operator.end_idx_orders[-2] :: 2], label[:, self.model_operator.end_idx_orders[-2] + 1 :: 2]],
                ).permute(
                    1, 0, 2
                )  # stack like: [[relatedX, ...], [relatedY, ...]]

                data, batch_coordXYs = self.test_transforms(data, batch_coordXYs)
                batch_coordXYs = batch_coordXYs.permute(1, 0, 2)
                label[:, self.model_operator.end_idx_orders[-2] :: 2] = batch_coordXYs[0]
                label[:, self.model_operator.end_idx_orders[-2] + 1 :: 2] = batch_coordXYs[1]

                pred = self.model(data)
                loss_record[:] += self.model_operator.update(pred, label, isTrain=False).cpu()
                acc_record[:] += self.acc_func(pred, label, hit_idxs, isHits).cpu()
                num_iter += 1

        loss_record /= num_iter
        acc_record /= num_iter

        return loss_record, acc_record

    def training(
        self,
        num_epoch: int,
        loader: DataLoader,
        val_loader: DataLoader = None,
        test_set: CDNet2014Dataset = None,
        saveDir: Path = PROJECT_DIR,
        useSummary: bool = True,
        early_stop: int = 50,
        checkpoint: int = 20,
        *args,
        **kwargs,
    ):
        data: torch.Tensor
        label: torch.Tensor
        hit_idxs: torch.Tensor
        isHits: torch.Tensor

        if useSummary:
            summary_path = f'{saveDir}/summary/'
            self.train_summary = SummaryRecord(SummaryWriter(summary_path), summary_path, num_epoch)
        #     pixelLevel_record = self.train_summary.pixelLevel_record
        # else:
        #     pixelLevel_record = BasicRecord('PixelLevel', num_epoch)

        isStop = False
        for self.epoch in range(num_epoch):
            measure_table = self.create_measure_table()

            isBest = False

            # self.FE_model.train()
            # self.ME_model.train()

            cate_id: int
            video_id: int
            features: torch.Tensor
            frames: torch.Tensor
            labels: torch.Tensor
            for (cate_id, video_id), features, frames, labels in tqdm(loader):
                features = features.to(self.device)
                frames = frames.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    features, frames, labels = self.train_transforms(features, frames, labels)

                bg_only_imgs = features[:, 0].unsqueeze(1)
                for step in range(frames.shape[1]):
                    frame, label = frames[:, step], labels[:, step]

                    combine_features = torch.hstack((features, bg_only_imgs))

                    features = self.FE_model(combine_features)

                    combine_features = torch.hstack((features, frame))
                    preds = self.ME_model(combine_features)
                    preds = self.loss_func(preds)

        #         pred = self.model(data)
        #         loss_records[self.epoch] += self.model_operator.update(pred, label).cpu()
        #         a_r = self.acc_func(pred, label, hit_idxs, isHits).cpu()
        #         num_missM_nan += 1 - (a_r[-1] // (a_r[-1] - 0.0000001))
        #         acc_records[self.epoch] += a_r
        #         num_iter += 1

        #     loss_records[self.epoch] /= num_iter
        #     acc_records[self.epoch, :-2] /= num_iter
        #     acc_records[self.epoch, -2:] /= num_iter - num_missM_nan

        #     loss_table.add_row('Train', *[f'{l:.3e}' for l in loss_records[self.epoch]])
        #     acc_table.add_row('Train', *[f'{a:.3f}' for a in acc_records[self.epoch]])

        #     if val_loader is not None:
        #         val_loss_records[self.epoch], val_acc_records[self.epoch] = self.validating(val_loader)

        #         loss_table.add_row('val', *[f'{l:.3e}' for l in val_loss_records[self.epoch]])
        #         acc_table.add_row('val', *[f'{a:.3f}' for a in val_acc_records[self.epoch]])

        #         best_loss_checker = self.best_loss_record > val_loss_records[self.epoch]
        #         self.best_loss_record[best_loss_checker] = val_loss_records[self.epoch, best_loss_checker]

        #         best_acc_checker = self.best_acc_record < val_acc_records[self.epoch]
        #         self.best_acc_record[best_acc_checker] = val_acc_records[self.epoch, best_acc_checker]

        #         if best_acc_checker.any() or best_loss_checker.any():
        #             self.best_epoch = self.epoch
        #             isBest = True

        #     self.console.print(loss_table)
        #     self.console.print(acc_table)

        #     # * Save Stage
        #     isCheckpoint = self.epoch % checkpoint == 0
        #     if self.best_epoch:
        #         save_path = f'lossSum-{val_loss_records[self.epoch, -1]:.3e}_accMean-{val_acc_records[self.epoch, -6]:.3f}.pt'
        #         isStop = early_stop == (self.epoch - self.best_epoch)
        #     else:
        #         save_path = f'lossSum-{loss_records[self.epoch, -1]:.3e}_accMean-{acc_records[self.epoch, -6]:.3f}.pt'

        #     save_path_heads: List[str] = []
        #     if isCheckpoint:
        #         save_path_heads.append(f'checkpoint_e{self.epoch:03}')
        #     if isBest:
        #         save_path_heads.extend(
        #             [f'bestLoss-{name}' for name, is_best in zip(self.loss_order_names, best_loss_checker) if is_best],
        #         )
        #         save_path_heads.extend(
        #             [f'bestAcc-{name}' for name, is_best in zip(self.acc_order_names, best_acc_checker) if is_best],
        #         )

        #     isStop += self.epoch + 1 == num_epoch
        #     if isStop:
        #         save_path_heads.append(f'final_e{self.epoch:03}_')

        #     for i, path_head in enumerate(save_path_heads):
        #         if i == 0:
        #             epoch_path = f'e{self.epoch:03}_{save_path}'
        #             self.model_operator.save(self.model, str(saveDir / epoch_path))
        #             print(f"Save Model: {str_format(str(epoch_path), fore='g')}")
        #             model_perform = ModelPerform(
        #                 self.loss_order_names,
        #                 self.acc_order_names,
        #                 loss_records[: self.epoch + 1],
        #                 acc_records[: self.epoch + 1],
        #                 val_loss_records[: self.epoch + 1],
        #                 val_acc_records[: self.epoch + 1],
        #             )
        #             model_perform.save(str(saveDir))

        #         path: Path = saveDir / f'{path_head}.pt'
        #         path.unlink(missing_ok=True)
        #         path.symlink_to(epoch_path)
        #         print(f"symlink: {str_format(str(path_head), fore='y'):<36} -> {epoch_path}")

        #     if isStop:
        #         print(str_format("Stop!!", fore='y'))
        #         break

        # if val_loader is None:
        #     return loss_records, acc_records
        # return loss_records, acc_records, val_loss_records, val_acc_records


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
                    transforms.RandomCrop(size=sizeHW),
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
        transforms.GaussianBlur([3, 3]),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, hue=0.2, contrast=0.5, saturation=0.2)], p=0.75),
    ]

    train_iter_compose = IterativeCustomCompose([*argumentation_order_ls], transform_img_size=sizeHW)
    test_iter_compose = None

    optimizer = optim.Adam
    lr = 1e-4
    BATCH_SIZE = 32
    fe_model = FEModel(9)
    me_model = MEModel(6)

    train_loader, val_loader, test_set = get_dataLoaders_and_testSet(
        dataset_cfg=DatasetConfig(),
        cv_set=5,
        dataset_rate=1,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        train_transforms_cpu=train_trans_cpu,
        test_transforms_cpu=test_trans_cpu,
        label_isShadowFG=False,
    )

    saveDir = f'out/{time.strftime("%m%d-%H%M")}_{fe_model.__class__.__name__}-{me_model.__class__.__name__}_BS-{BATCH_SIZE}'
    check2create_dir(saveDir)

    model_process = DL_Model(fe_model, me_model, train_iter_compose, test_iter_compose, device='cuda:0')
    model_process.training(3, train_loader, val_loader, test_set, saveDir=Path(saveDir), early_stop=3, checkpoint=3)
