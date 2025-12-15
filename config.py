import csv
from typing import Dict, List

from utils.DataID_MatchTable import VID2ID


class DatasetConfig:
    currentFrDir = 'Data/currentFr'
    emptyBgDir = 'Data/emptyBg'
    recentBgDir = 'Data/recentBg'
    moveObjCropDir = 'Data/moveObjCrop-exp1'
    cdnet2014SelectedCSV = 'CDNET2014_selected_frames_200.csv'
    frame_groups = 7
    gap_range = [3, 200]
    sample4oneVideo = 200

    cdnet2014SelectedMap: Dict[int, List[int]]

    def __init__(self, isModel3D=True) -> None:
        self.concat_axis = -1 if isModel3D else 0  # axis dependent by model input_channel
        self.cdnet2014SelectedMap = self.get_selected_frames(self.cdnet2014SelectedCSV)

    @staticmethod
    def get_selected_frames(csv_file):
        data = {}

        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)

            for row in reader:
                key = row[0]
                values = list(map(int, row[1].split(' ')))

                data[VID2ID[key.split('/')[-1]]] = values

        return data
