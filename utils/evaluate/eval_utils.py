from typing import Dict, Tuple, Callable

import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.DataID_MatchTable import VID2ID, CAT2ID, ID2VID, ID2CAT
from utils.evaluate.accuracy import calculate_acc_metrics as acc_func
from utils.evaluate.losses import IOULoss4CDNet2014 as Loss
from submodules.UsefulFileTools.FileOperator import check2create_dir


ACC_NAMES = ['Prec', 'Recall', 'FNR', 'F_score', 'ACC']
LOSS_NAMES = [Loss.__repr__()]
ORDER_NAMES = [*ACC_NAMES, *LOSS_NAMES]


class EvalMeasure(nn.Module):
    def __init__(self, thresh: float, loss_func: Loss | nn.Module = Loss(reduction='none')):
        super(EvalMeasure, self).__init__()
        self.thresh = thresh
        self.loss_func = loss_func
        LOSS_NAMES[0] = str(loss_func)

    def forward(self, gts: torch.Tensor, preds: torch.Tensor, preds_mask: torch.Tensor, video_ids: torch.Tensor) -> torch.Tensor:
        '''
        gts, preds : 4-dimension -> batch * channel * im_height * im_width. (batch, channel, height, width)
        video_ids: 2-dimension -> batch * 1. (batch, vid_idx)
        result : 2-dimension -> batch * 5;  (batch, features) ; features-> (tp, fp, tn, fn, vid_idx)
        '''

        losses = self.loss_func(preds, gts)
        gts = gts.type(torch.int32)

        diff = gts ^ preds_mask
        same = diff ^ torch.ones_like(diff)
        gt_1 = gts == 1
        gt_0 = gts == 0

        tp, fp = gt_1 * same, gt_0 * diff
        tn, fn = gt_0 * same, gt_1 * diff
        batch = tp.size()[0]

        tp, fp = tp.view(batch, -1).sum(dim=1, keepdim=True), fp.view(batch, -1).sum(dim=1, keepdim=True)
        tn, fn = tn.view(batch, -1).sum(dim=1, keepdim=True), fn.view(batch, -1).sum(dim=1, keepdim=True)

        return torch.cat((tp, fp, tn, fn, losses, video_ids), dim=1)


class OneEpochVideosAccumulator:
    '''The `OneEpochVideosAccumulator` class accumulates and updates statistics for video features during one epoch of training.'''

    id2vid = ID2VID
    vid2id = VID2ID

    id2cat = ID2CAT
    cat2id = CAT2ID

    def __init__(self) -> None:
        self.vid_matrix: dict[int : torch.Tensor] = dict()
        # {id: (tp, fp, tn, fn, loss, accumulative_times)}
        self.batchLevel_matrix: torch.Tensor = torch.zeros(6, dtype=torch.float64, device='cpu')

    def accumulate(self, result: torch.Tensor) -> None:
        result = result.to('cpu')
        for feature in result:
            # feature: (tp, fp, tn, fn, loss, video_id)

            vid_matrix = self.vid_matrix.setdefault(int(feature[-1]), torch.zeros_like(self.batchLevel_matrix))
            vid_matrix[:-1] += feature[:-1]
            vid_matrix[-1] += 1

            self.batchLevel_matrix[:-2] += feature[:-2]
            self.batchLevel_matrix[-1] += 1


class BasicRecord:
    __row_id = 0

    def __init__(self, task_name: str, num_epochs: int = 0) -> None:
        self.task_name = task_name
        self.__acc_freq = 0
        self.__score_records = torch.zeros((num_epochs, len(ORDER_NAMES)), dtype=torch.float32)

    @classmethod
    def next_row(cls):
        cls.__row_id += 1

    @classmethod
    def update_row_id(cls, new_id: int):
        cls.__row_id = new_id

    @classmethod
    @property
    def row_id(cls):
        return cls.__row_id

    @staticmethod
    def convert2df(records: torch.Tensor, start_row: int = 0, end_row: int = None):
        record_dict = {name: records[start_row:end_row, i].tolist() for i, name in enumerate(ORDER_NAMES)}
        return pd.DataFrame(record_dict)

    def concatScoreRecords(self, score_records2: torch.Tensor, *args):
        self.__score_records = torch.vstack([self.__score_records, score_records2, *args])
        self.update_row_id(self.__score_records.shape[0])

        return self.__score_records

    def save(self, saveDir: str, start_row: int = 0, end_row: int = None):
        self.convert2df(self.__score_records, start_row, end_row).to_csv(f'{saveDir}/{self.task_name}.csv')

    def record(self, *args: Tuple[torch.Tensor]):
        self.__score_records[self.__row_id] = torch.tensor(args, dtype=torch.float32)

    def accumulate(self, single_result: torch.Tensor):
        self.__score_records[self.__row_id] += single_result
        self.__acc_freq += 1

    def finish_acc_and_record(self):
        self.record(*(self.__score_records[self.__row_id] / self.__acc_freq))
        self.__acc_freq = 0

    @property
    def last_scores(self):
        return self.__score_records[self.__row_id]

    def __repr__(self) -> str:
        return f'{self.task_name}(\n{self.__score_records[:self.__row_id]}\n)'


class SummaryRecord:
    def __init__(
        self,
        saveDir: str,
        num_epochs: int,
        writer: SummaryWriter | None = None,
        acc_func: Callable[[int | torch.IntTensor], torch.Tensor] = acc_func,
        mode: str = 'Train',
    ) -> None:
        self.saveDir = saveDir
        self.num_epochs = num_epochs
        self.writer = writer
        self.mode = mode
        self.acc_func = acc_func

        self.cate_dict: Dict[int, BasicRecord] = {}
        self.video_dict: Dict[int, BasicRecord] = {}

        self.overall = BasicRecord('Overall', num_epochs)  # internal manipulate, auto calculate
        self.batchLevel = BasicRecord('BatchLevel', num_epochs)  # external manipulate

    def records(self, videosAccumulator: OneEpochVideosAccumulator):
        vid: int
        k: torch.Tensor
        for vid, k in videosAccumulator.vid_matrix.items():
            cid = vid // 10

            video_record = self.video_dict.setdefault(vid, BasicRecord(ID2VID[vid], self.num_epochs))
            cate_record = self.cate_dict.setdefault(cid, BasicRecord(ID2CAT[cid], self.num_epochs))

            video_record.record(*self.acc_func(*k[:-2]), k[-2] / k[-1])
            cate_record.accumulate(video_record.last_scores)
            self.write2tensorboard(task_name=f'{ID2CAT[cid]}/{ID2VID[vid]}', scores=self.video_dict[vid].last_scores)

        for cate_record in self.cate_dict.values():
            cate_record.finish_acc_and_record()
            self.overall.accumulate(cate_record.last_scores)
            self.write2tensorboard(task_name=cate_record.task_name, scores=cate_record.last_scores)

        self.overall.finish_acc_and_record()
        self.write2tensorboard(task_name=self.overall.task_name, scores=self.overall.last_scores)

        batchLevel_loss = videosAccumulator.batchLevel_matrix[-2] / videosAccumulator.batchLevel_matrix[-1]
        self.batchLevel.record(*self.acc_func(*videosAccumulator.batchLevel_matrix[:-2]), batchLevel_loss)
        self.write2tensorboard(task_name=self.batchLevel.task_name, scores=self.batchLevel.last_scores)

    def write2tensorboard(self, task_name: str, scores: torch.Tensor):
        if self.writer is None:
            return

        for name, score in zip(ORDER_NAMES, scores):
            self.writer.add_scalar(f'{self.mode}/{task_name}/{name}', score, BasicRecord.row_id)

    def export2csv(self):
        main_saveDir = f'{self.saveDir}/{self.mode}'
        check2create_dir(main_saveDir)
        self.overall.save(main_saveDir)
        self.batchLevel.save(main_saveDir)

        for cate_record in self.cate_dict.values():
            cate_record.save(main_saveDir)

        for id, video_record in self.video_dict.items():
            sub_saveDir = f'{main_saveDir}/{ID2CAT[id // 10]}'
            check2create_dir(sub_saveDir)
            video_record.save(sub_saveDir)

    def __repr__(self) -> str:
        return f'{self.mode}-Records'


if __name__ == "__main__":
    import os
    import sys
    import cv2 as cv
    import numpy as np
    from pathlib import Path
    import torchvision.transforms as tvtf

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.data_preprocess import CDNet2014Preprocess

    def test():
        preprocess = CDNet2014Preprocess((224, 224))
        eval = EvalMeasure(0.5, Loss(reduction='none'))
        gts_pth = "/root/Work/fork-BGS/BSUV-Net-2.0/dataset/currentFr/baseline/highway/groundtruth"
        imgs_name = sorted(os.listdir(gts_pth))

        gts = list()
        video_ids = list()
        for i, img_name in enumerate(imgs_name[699:]):
            if img_name.split('.')[-1] != 'png':
                continue
            elif i == 5:
                break

            img = cv.imread(os.path.join(gts_pth, img_name), cv.IMREAD_GRAYSCALE)
            gt = preprocess(img)
            gts.append(tvtf.ToTensor()(gt.copy()))
            video_ids.append(11)
            # gt[gt == 1] = 255
            # gt[gt == -1] = 128
            # gt[gt == 0] = 0
            # cv.imwrite(f'./{i+1+699}.png', gt)

        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        gts = torch.cat(gts, dim=0).reshape(len(gts), 1, 224, 224).to(device=device)
        preds = torch.zeros_like(gts, device=device)
        preds_mask = torch.zeros_like(gts, dtype=torch.int32, device=device)
        video_ids = torch.tensor(video_ids).reshape(5, 1).to(device=device, dtype=torch.int32)

        with torch.no_grad():
            result = eval(gts, preds, preds_mask, video_ids)
        print(result)

        video_acc = OneEpochVideosAccumulator()
        video_acc.accumulate(result)
        print(video_acc.vid_matrix)

        return video_acc

    video_acc = test()

    # ! Training example
    trainwriter = SummaryWriter('./out/test/115')
    testwriter = SummaryWriter('./out/test/115')
    train_summary = SummaryRecord(trainwriter, saveDir='./out/test', num_epochs=50)
    test_summary = SummaryRecord(testwriter, saveDir='./out/test', num_epochs=50, mode='Test')

    train_summary.records(video_acc)
    test_summary.records(video_acc)

    for epoch in range(49):
        if len(test_summary.cate_dict) != 0:
            BasicRecord.next_row()
        train_video_acc = OneEpochVideosAccumulator()
        test_video_acc = OneEpochVideosAccumulator()
        for video_acc in [train_video_acc, test_video_acc]:
            result = torch.randint(0, 1000, size=(5, 6), dtype=torch.float32)
            result[:, -1] = torch.arange(12, 60, 10)
            video_acc.accumulate(result)
        train_summary.records(train_video_acc)
        test_summary.records(test_video_acc)

    print(train_summary.cate_dict)
    print(train_summary.video_dict)
    print(test_summary.cate_dict)
    print(test_summary.video_dict)

    #  ==============
    train_summary.records(video_acc)
    test_summary.records(video_acc)

    train_summary.export2csv()
    test_summary.export2csv()
