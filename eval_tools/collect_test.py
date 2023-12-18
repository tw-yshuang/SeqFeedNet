import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from submodules.UsefulFileTools.FileOperator import get_filenames, check2create_dir


class TestResultCollector:
    def __init__(self, score_names: List[str], col_names: List[str], fromDir: str, toDir: str) -> None:
        self.fromDir = fromDir
        self.toDir = toDir
        self.score_names = score_names
        self.col_names = col_names
        self.col_dict = {name: [] for name in col_names}

    def collect(self, target: str):
        filenames = list(set(get_filenames(self.fromDir, f'**/Test/{target}.csv', withDirPath=False)))

        for filename in filenames:
            full_name, type_names = filename.split('/')[:2]
            name = full_name.split('_')[1]
            type_names = type_names.split('_')
            type_name, epoch = type_names[-2], type_names[-1][1:]

            test_csv = pd.read_csv(f'{dir_name}/{filename}')

            slice_flag = None if score_names[-1] in test_csv.columns else -1
            scores = test_csv.loc[0, score_names[:slice_flag]]

            loss_scores = getattr(scores, score_names[-1]) if slice_flag is None else [None] * scores.shape[0]

            for name, score in zip(
                col_names, [name, full_name, type_name, epoch, scores.Prec, scores.Recall, scores.F_score, scores.ACC, loss_scores]
            ):
                self.col_dict[name].append(score)

        save_dir = '/'.join(f'{out_dir}/{target}.csv'.split('/')[:-1])
        check2create_dir(save_dir)
        pd.DataFrame(self.col_dict).to_csv(f'{out_dir}/{target}.csv', index=False)


if __name__ == '__main__':
    dir_name = 'out'
    out_dir = 'out/Compare'
    check2create_dir(out_dir)

    target = 'Overall'

    score_names = ['Prec', 'Recall', 'F_score', 'ACC', 'IOULoss']
    col_names = ['Name', 'Full_Name', 'Type', 'Epoch', *score_names]

    collector = TestResultCollector(score_names, col_names, fromDir=dir_name, toDir=out_dir)
    collector.collect(target)
