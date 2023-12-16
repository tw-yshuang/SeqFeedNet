import os, sys, subprocess, glob, ast, time
from typing import Dict, List
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from submodules.UsefulFileTools.FileOperator import get_filenames, str_format


class GPU_Provider:
    def __init__(self, GPUS: List[int]) -> None:
        self.GPUS = GPUS
        self.GPU_COUNT_DICT: Dict[int, int]
        self.init_gpus()
        self.__gpu = self.select_gpu()

    def init_gpus(self):
        self.GPU_COUNT_DICT = {gpu: 0 for gpu in self.GPUS}

    def select_gpu(self):
        i = 0
        gpus_len = len(self.GPUS)

        while True:
            gpu = self.GPUS[i % gpus_len]
            if self.GPU_COUNT_DICT[gpu] == 2:
                print(str_format("Wait 3.5 hr for GPU release computational power...", fore='y'))
                time.sleep(3.5 * 60 * 60)  # 3.5hr
                self.init_gpus()

            self.GPU_COUNT_DICT[gpu] += 1
            i += 1
            yield gpu

    def get(self):
        return next(self.__gpu)


def batch_testing4weights(task_dir: str, weight_tags: List[str], cross_validation: int, gpu: GPU_Provider):
    print(str_format(f"start testing directory: {task_dir}", fore='g'))

    filenames = sorted(map(os.path.basename, glob.glob(f'{task_dir}/*.pt')), key=len)
    max_len = len(filenames[-1])
    filenames = [filename for filename in filenames if len(filename) < max_len - 5]

    i = 0
    for weight_tag in weight_tags:
        weight_tag_info = weight_tag.split('_')
        tag_len = len(weight_tag_info[0])
        tag_filenames = sorted([filename for filename in filenames if filename.split('_')[0][-tag_len:] == weight_tag_info[0]])
        if len(weight_tag_info) == 2:  # resolve slice info
            slice_info = [None, None, None]
            tag_slice_info = weight_tag_info[1].split(':')
            slice_info[: len(tag_slice_info)] = map(ast.literal_eval, tag_slice_info)
            tag_filenames = tag_filenames[slice_info[0] : slice_info[1] : slice_info[2]]

        for tag_filename in tag_filenames:
            # print(f'python3 testing.py -cv {cross_validation} --device {gpu.get()} -weight {task_dir}/{tag_filename}')
            subprocess.Popen(
                f'python3 testing.py -cv {cross_validation} --device {gpu.get()} -weight {task_dir}/{tag_filename}',
                text=True,
                shell=True,
            )
            i += 1


if __name__ == '__main__':
    task_dict = {
        'dev/EmAsInp': [
            'out/1214-2031_EmAsInp.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'dev/EmAsInp.feaERD.1eEnNorm': [
            'out/1215-1036_EmAsInp.feaERD.1eEnNorm_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'EmAsInp.1eb': [
            'out/1215-1207_EmAsInp.1eb.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'EmAsInp.feaERD.1eEnNorm.1eb': [
            'out/1215-1230_EmAsInp.feaERD.1eEnNorm.1eb.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
    }

    weight_tags = [
        # 'F',
        'Loss',
        # 'final',
        'checkpoint_-3:None',
    ]

    cross_val = 2
    gpu = GPU_Provider([0, 2, 4, 5, 7])

    for branch, task_dirs in task_dict.items():
        subprocess.run(f'git checkout {branch}', text=True, shell=True)

        for task_dir in task_dirs:
            batch_testing4weights(task_dir, weight_tags, cross_val, gpu)
        time.sleep(5 * 60)
        print("Wait for previous branch...")
