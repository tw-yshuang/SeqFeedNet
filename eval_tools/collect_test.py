import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from submodules.UsefulFileTools.FileOperator import get_filenames

if __name__ == '__main__':
    dir_name = 'out'
    target = 'Overall'
    filenames = list(set(get_filenames(dir_name, f'**/Test/{target}.csv', withDirPath=False)))

    score_names = ['Prec', 'Recall', 'F_score', 'ACC', 'IOULoss']
    col_names = ['Name', 'Full_Name', 'Type', 'Epoch', *score_names]
    col_dict = {name: [] for name in col_names}

    for filename in filenames:
        full_name, type_name = filename.split('/')[:2]
        name = full_name.split('_')[1]
        type_name = type_name.split('_')[1]

        if 'result' in type_name:
            type_name = 'F1'
        if 'F' in type_name:
            pt_file = str(Path(f'{dir_name}/{full_name}/bestAcc-F_score.pt').readlink()).split('_')
        elif 'final' in type_name:
            pt_file = get_filenames(f'{dir_name}/{full_name}', 'final*.pt', withDirPath=False)[0].split('_')[1:]

        epoch = int(pt_file[0][1:])

        test_csv = pd.read_csv(f'{dir_name}/{filename}')
        scores = test_csv.loc[0, score_names]

        for name, score in zip(col_names, [name, full_name, type_name, epoch, scores.Prec, scores.Recall, scores.F_score, scores.ACC]):
            col_dict[name].append(score)

    # # clean duplicate data
    # items = list(zip(col_dict['Full_Name'], col_dict['Type']))
    # dup_idxs = np.where(pd.Series(items).duplicated())[0]
    # aa = 0

    pd.DataFrame(col_dict).to_csv(f'{dir_name}/Compare_{target}.csv', index=False)
