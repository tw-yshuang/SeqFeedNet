import sys
from typing import List
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from submodules.UsefulFileTools.WordOperator import str_format
from submodules.UsefulFileTools.FileOperator import get_filenames, check2create_dir

from cross_validation_config import datasets_tr


def get_best_overall(path: str):
    df = pd.read_csv(path)
    df = df[df['Type'] == 'bestAcc-F']
    df = df[~df['Name'].str.contains('.no1eb')]
    best_df = pd.DataFrame(columns=df.columns)
    for i in range(1, 5):
        cv_df = df[df['Name'].str.contains(f'cv{i}')]
        best_df = pd.concat([best_df, cv_df[cv_df['F_score'] == cv_df['F_score'].max()]], ignore_index=True)

    return best_df


def get_categories_f1(dir_name: str, col_names: List[str], select_names: List[str]):
    cate_df = pd.DataFrame(columns=['Name'])
    cate_df['Name'] = select_names

    for cate_name in col_names:
        df = pd.read_csv(f'{dir_name}/{cate_name}.csv')
        df = df[df['Type'] == 'bestAcc-F']
        df = df[df['Name'].str.contains('|'.join(select_names), na=False)]

        cate_df = cate_df.merge(df[['Name', 'F_score']], on='Name')
        cate_df.rename(columns={'F_score': cate_name}, inplace=True)

    return cate_df


if __name__ == '__main__':
    dir_name = 'out/Compare'
    overall_name = 'Overall.csv'

    best_df = get_best_overall(f'{dir_name}/{overall_name}')
    print(best_df)
    print(best_df['F_score'].mean())

    col_names = list(datasets_tr[0].keys())

    cate_df = get_categories_f1(dir_name, col_names, best_df['Name'].to_list())
    print(cate_df)

    best_df.to_csv(f'{dir_name}/Overall_f1.csv', index=False)
    cate_df.to_csv(f'{dir_name}/categories_f1.csv', index=False)
