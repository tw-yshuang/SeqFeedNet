from typing import Dict, List, Callable

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.DataID_MatchTable import VID2ID, CAT2ID, ID2VID, ID2CAT
from utils.evaluate.accuracy import calculate_acc_metrics as acc_func
from utils.evaluate.losses import CDNet2014_JaccardLoss as Loss


ACC_NAMES = ['Prec', 'Recall', 'FNR', 'F_score', 'ACC']
LOSS_NAMES = ['Loss']
ORDER_NAMES = [*ACC_NAMES, *LOSS_NAMES]


class EvalMeasure(nn.Module):
    def __init__(self, thresh: float, loss_func: Loss | nn.Module = Loss(reduction='none')):
        super().__init__()
        self.thresh = thresh
        self.loss_func = loss_func

    def forward(self, gts: torch.Tensor, preds: torch.Tensor, vid_indices: torch.Tensor) -> torch.Tensor:
        '''
        gts, preds : 4-dimension -> batch * channel * im_height * im_width. (batch, channel, height, width)
        vid_indices: 2-dimension -> batch * 1. (batch, vid_idx)
        result : 2-dimension -> batch * 5;  (batch, features) ; features-> (tp, fp, tn, fn, vid_idx)
        '''

        losses = self.loss_func(preds, gts)

        gts = gts.type(torch.int32)
        preds = torch.where(preds > self.thresh, 1, 0).type(dtype=torch.int32)

        diff = gts ^ preds
        same = diff ^ torch.ones_like(diff)
        gt_1 = gts == 1
        gt_0 = gts == 0

        tp, fp = gt_1 * same, gt_0 * diff
        tn, fn = gt_0 * same, gt_1 * diff
        batch = tp.size()[0]

        tp, fp = tp.view(batch, -1).sum(dim=1, keepdim=True), fp.view(batch, -1).sum(dim=1, keepdim=True)
        tn, fn = tn.view(batch, -1).sum(dim=1, keepdim=True), fn.view(batch, -1).sum(dim=1, keepdim=True)

        return torch.cat((tp, fp, tn, fn, losses, vid_indices), dim=1)


class OneEpochVideosAccumulation:
    id2vid = ID2VID
    vid2id = VID2ID

    id2cat = ID2CAT
    cat2id = CAT2ID

    def __init__(self) -> None:
        self.vid_matrix: dict[int : torch.Tensor] = dict()

    def accumulate(self, result: torch.Tensor) -> None:
        result = result.to('cpu')
        for feature in result:
            vid_idx = int(feature[-1])
            if self.vid_matrix.get(vid_idx, None) is None:
                # {id: [tp, fp, tn, fn, losses, accumulative_times]}
                self.vid_matrix[vid_idx] = torch.zeros(6, dtype=torch.float64, device='cpu')

            vid_matrix = self.vid_matrix[vid_idx]
            vid_matrix[:-2] += feature[:-2]
            vid_matrix[-2] = (vid_matrix[-2] * vid_matrix[-1] + feature[-2]) / (vid_matrix[-1] + 1)
            vid_matrix[-1] += 1


class BasicRecord:
    row_id = 0

    def __init__(self, task_name: str, num_epoch: int = 0) -> None:
        self.task_name = task_name
        self.score_records = torch.zeros((num_epoch, len(ORDER_NAMES)), dtype=torch.float32)

    @classmethod
    def next_row(cls):
        cls.row_id += 1

    @classmethod
    def update_row_id(cls, new_id: int):
        cls.row_id = new_id

    @staticmethod
    def convert2df(records: torch.Tensor, start_row: int = 0, end_row: int = None):
        record_dict = {name: records[start_row:end_row, i].tolist() for i, name in enumerate(ORDER_NAMES)}
        return pd.DataFrame(record_dict)

    def concatScoreRecords(self, score_records2: torch.Tensor, *args):
        self.score_records = torch.vstack([self.score_records, score_records2, *args])
        self.update_row_id(self.score_records.shape[0])

        return self.score_records

    def save(self, saveDir: str, start_row: int = 0, end_row: int = None):
        self.convert2df(self.score_records, start_row, end_row).to_csv(f'{saveDir}/{self.task_name}.csv')

    def record(self, *args: torch.Tensor):
        self.score_records[self.row_id, :] = torch.tensor(args, dtype=torch.float32)

    @property
    def last_scores(self):
        return self.score_records[self.row_id]

    def __repr__(self) -> str:
        return f'{self.task_name}(\n{self.score_records[:self.row_id]}\n)'


class SummaryRecord:
    order_names = [*ACC_NAMES, *LOSS_NAMES]

    def __init__(
        self,
        writer: SummaryWriter,
        saveDir: str,
        num_epoch: int,
        mode: str = 'Train',
        acc_func: Callable[[int | torch.IntTensor], torch.Tensor] = acc_func,
    ) -> None:
        self.writer = writer
        self.saveDir = saveDir
        self.num_epoch = num_epoch
        self.mode = mode
        self.acc_func = acc_func

        self.cate_records: Dict[int, BasicRecord] = {}
        self.video_records: Dict[int, BasicRecord] = {}

    def records(self, videosAccumulation: OneEpochVideosAccumulation):
        vid: int
        k: torch.Tensor
        for vid, k in videosAccumulation.vid_matrix.items():
            self.video_records.setdefault(vid, BasicRecord(ID2VID[vid], self.num_epoch)).record(*self.acc_func(*k[:-2]), k[-2])
            self.write2tensorboard(task_name=f'{ID2CAT[vid // 10]}/{ID2VID[vid]}', scores=self.video_records[vid].last_scores)

        cid_freq = {}
        for vid, video_record in self.video_records.items():
            cid = vid // 10
            cid_freq[cid] = cid_freq.setdefault(cid, 0) + 1

            self.cate_records.setdefault(cid, BasicRecord(ID2CAT[cid], self.num_epoch))
            cate_record = self.cate_records[cid]
            cate_record.record(*((cate_record.last_scores * (cid_freq[cid] - 1) + video_record.last_scores) / cid_freq[cid]))
            self.write2tensorboard(task_name=str(ID2CAT[cid]), scores=self.cate_records[cid].last_scores)

    def write2tensorboard(self, task_name: str, scores: torch.Tensor):
        for name, score in zip(self.order_names, scores):
            self.writer.add_scalar(f'{self.mode}/{task_name}/{name}', score, BasicRecord.row_id)

    # ! need implement
    def export2csv(self):
        ...

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
        vid_indices = list()
        for i, img_name in enumerate(imgs_name[699:]):
            if img_name.split('.')[-1] != 'png':
                continue
            elif i == 5:
                break

            img = cv.imread(os.path.join(gts_pth, img_name), cv.IMREAD_GRAYSCALE)
            gt = preprocess(img)
            gts.append(tvtf.ToTensor()(gt.copy()))
            vid_indices.append(11)
            # gt[gt == 1] = 255
            # gt[gt == -1] = 128
            # gt[gt == 0] = 0
            # cv.imwrite(f'./{i+1+699}.png', gt)

        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        gts = torch.cat(gts, dim=0).reshape(len(gts), 1, 224, 224).to(device=device)
        preds = torch.zeros_like(gts, device=device)
        vid_indices = torch.tensor(vid_indices).reshape(5, 1).to(device=device, dtype=torch.int32)

        with torch.no_grad():
            result = eval(gts, preds, vid_indices)
        print(result)

        video_acc = OneEpochVideosAccumulation()
        video_acc.accumulate(result)
        print(video_acc.get_vid_ratio(11))
        print(video_acc.vid_matrix)

        return video_acc

    video_acc = test()

    # ! Training example
    writer = SummaryWriter('./out/test/114')
    train_summary = SummaryRecord(writer, saveDir='./out/test', num_epoch=50)
    test_summary = SummaryRecord(writer, saveDir='./out/test', num_epoch=50, mode='Test')

    train_summary.records(video_acc)
    test_summary.records(video_acc)

    for epoch in range(49):
        if len(test_summary.cate_records) != 0:
            BasicRecord.next_row()
        train_video_acc = OneEpochVideosAccumulation()
        test_video_acc = OneEpochVideosAccumulation()
        for video_acc in [train_video_acc, test_video_acc]:
            result = torch.randint(0, 1000, size=(5, 6), dtype=torch.float32)
            result[:, -1] = torch.arange(12, 60, 10)
            video_acc.accumulate(result)
        train_summary.records(train_video_acc)
        test_summary.records(test_video_acc)

    print(train_summary.cate_records)
    print(train_summary.video_records)
    print(test_summary.cate_records)
    print(test_summary.video_records)

    #  ==============
    train_summary.records(video_acc)
    test_summary.records(video_acc)
