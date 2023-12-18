import os, sys, subprocess, glob, time
from typing import Dict, List
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from submodules.UsefulFileTools.FileOperator import get_filenames, str_format


class GPU_Provider:
    ONE_TEST_SPEND_TIME = 25 * 60

    def __init__(self, GPUS: List[int], max_overlap: int = 5) -> None:
        self.GPUS = GPUS
        self.max_overlap = max_overlap
        self.delay_time = len(GPUS) * max_overlap * self.ONE_TEST_SPEND_TIME
        self.GPU_COUNT_DICT: Dict[int, int]
        self.init_time: float
        self.init_gpus()
        self.__gpu = self.select_gpu()

    def init_gpus(self):
        self.GPU_COUNT_DICT = {gpu: 0 for gpu in self.GPUS}
        self.init_time = time.time()

    def select_gpu(self):
        i = 0
        gpus_len = len(self.GPUS)

        while True:
            gpu = self.GPUS[i % gpus_len]
            if self.GPU_COUNT_DICT[gpu] == self.max_overlap:
                delay_time = self.delay_time - (time.time() - self.init_time)
                print(
                    str_format(
                        f"\n[{self.current_time_str}] Wait {delay_time/ 3600:.1f} hr for GPU release computational power...", fore='y'
                    )
                )
                time.sleep(delay_time)
                print(str_format(f"\n[{self.current_time_str}] GPU released computational power!!", fore='y'))
                self.init_gpus()

            self.GPU_COUNT_DICT[gpu] += 1
            i += 1
            yield gpu

    def get(self):
        return next(self.__gpu)

    @property
    def current_time_str(self):
        return time.strftime('%Y%m%d %H:%M', time.localtime())


def batch_testing4weights(task_dir: str, weight_tags: List[str], cross_validation: int, gpu_provider: GPU_Provider) -> bool:
    isExecute = False
    print(str_format(f"start testing directory: {task_dir}", fore='g'))

    filenames = sorted(map(os.path.basename, glob.glob(f'{task_dir}/*.pt')), key=len)
    max_len = len(filenames[-1])
    filenames = [filename for filename in filenames if len(filename) < max_len - 5]

    for weight_tag in weight_tags:
        weight_tag_info = weight_tag.split('_')
        tag_len = len(weight_tag_info[0])
        tag_filenames = sorted([filename for filename in filenames if filename.split('_')[0][-tag_len:] == weight_tag_info[0]])
        if len(weight_tag_info) == 2:  # resolve slice info
            slice_info = [None, None, None]
            tag_slice_info = weight_tag_info[1].split(':')
            slice_info[: len(tag_slice_info)] = [int(slice_info) if slice_info != '' else None for slice_info in tag_slice_info]
            tag_filenames = tag_filenames[slice_info[0] : slice_info[1] : slice_info[2]]

        for tag_filename in tag_filenames:
            pretrain_weight_path = f'{task_dir}/{tag_filename}'

            standard_testDirName = get_standard_testDirName(pretrain_weight_path)
            if Path(f'{standard_testDirName}/Test/Overall.csv').exists():
                print(str_format(f"Already tested {standard_testDirName}", fore="pink"))
                continue

            gpu = gpu_provider.get()
            command = f'python3 testing.py -cv {cross_validation} --device {gpu} -weight {pretrain_weight_path}'
            print(str_format(f"execute: {command}", fore="sky"))
            # print(f'python3 testing.py -cv {cross_validation} --device {gpu} -weight {pretrain_weight_path}')
            subprocess.Popen(command, text=True, shell=True)
            isExecute = True
    return isExecute


def get_standard_testDirName(pretrain_weight_path: str):
    path = Path(pretrain_weight_path)
    split_id = pretrain_weight_path.rfind('/') + 1
    testDirName = 'out' if pretrain_weight_path == '' else pretrain_weight_path[:split_id]
    if path.is_symlink():
        testDirName += f'{pretrain_weight_path[split_id:].split("_")[0]}_'
        path = str(path.readlink())
    else:
        path = pretrain_weight_path[split_id:]
    testDirName += path.split('_')[0]
    return testDirName


def merge_develop_branch(branch: str):
    try:
        subprocess.check_call(f'git checkout {branch}'.split())
        subprocess.check_call(f'git merge --no-ff --no-edit develop'.split())
    except subprocess.CalledProcessError as error_msg:
        print(f"{str_format('[CalledProcessError]', fore='r')} {error_msg}")
        exit()


if __name__ == '__main__':
    task_dict = {
        'develop': [
            'out/1211-0444_iouLoss.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'dev/label2bg': [
            'out/1211-1607_label2bg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'dev/dataset-em,1ref,1diff': [
            'out/1211-1611_feaERD.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'dev/1epochBackward': [
            'out/1211-1614_1eb.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-9_Set-2',
        ],
        'dev/feaERD.1eEnNorm': [
            'out/1211-1617_feaERD.1eEnNorm.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'com/feaERD.label2bg': [
            'out/1211-1626_feaERD.label2bg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'dev/label2bgRandom': [
            'out/1211-1604_label2bgRandom.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'com/1eb.label2bg': [
            'out/1211-1632_1eb.label2bg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-9_Set-2',
        ],
        'com/feaERD.1eb.label2bg': [
            'out/1211-1636_feaERD.1eb.label2bg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-9_Set-2',
        ],
        'dev/RecAsInp': [
            'out/1211-1621_RecAsInp.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'dev/EmAsInp': [
            'out/1214-2031_EmAsInp.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'com/EmAsInp.feaERD.1eEnNorm': [
            'out/1215-1036_EmAsInp.feaERD.1eEnNorm_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'com/EmAsInp.1eb': [
            'out/1215-1207_EmAsInp.1eb.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'com/EmAsInp.feaERD.1eEnNorm.1eb': [
            'out/1215-1230_EmAsInp.feaERD.1eEnNorm.1eb.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
    }

    # Flags:
    #          F: bestAcc-F_score,
    #       Loss: bestLoss-IOULoss,
    #       Prec: bestAcc-Prec
    #     Recall: bestAcc-Recall
    #        ACC: bestAcc-ACC,
    #        FNR: bestAcc-FNR,
    #   xxx(int): number of epoch
    # checkpoint: checkpoint_x:x:x, e.g. 'checkpoint_-3:', 'checkpoint_1:4:2'
    weight_tags = [
        'F',
        'Loss',
        'final',
        'checkpoint_-3:',
    ]

    cross_val = 2
    gpu_provider = GPU_Provider([0, 2, 4, 5, 7], max_overlap=5)

    for branch in task_dict.keys():
        merge_develop_branch(branch)

    for branch, task_dirs in task_dict.items():
        if task_dirs == []:
            continue

        print("")
        # print(f'git checkout {branch}')
        subprocess.run(f'git checkout {branch}', text=True, shell=True)
        time.sleep(3)

        isExecute = False
        for task_dir in task_dirs:
            if batch_testing4weights(task_dir, weight_tags, cross_val, gpu_provider) is True:
                isExecute = True
        if isExecute:
            time.sleep(5 * 60)
            print("Wait for previous branch...")
