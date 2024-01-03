import os, sys, subprocess, glob, time
from typing import List
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from submodules.UsefulFileTools.FileOperator import str_format

ENV = os.environ.copy()
ENV["LINES"] = "40"
ENV["COLUMNS"] = "203"


class GPU_Provider:
    ONE_TEST_SPEND_TIME = 5 * 60

    def __init__(self, GPUS: List[int], max_overlap: int = 5) -> None:
        self.GPUS = GPUS
        self.max_overlap = max_overlap
        self.delay_time = self.ONE_TEST_SPEND_TIME
        self.GPU_COUNT_DICT = {gpu: 0 for gpu in self.GPUS}
        self.__gpu = self.select_gpu()

    def get_num_testing_in_OneGPU(self, gpu_id: int):
        self.GPU_COUNT_DICT[gpu_id] = len(
            subprocess.run(
                f"nvitop -1 --only {gpu_id} | grep -o 'testing'",
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding='utf-8',
                env=ENV,
                timeout=30,
                shell=True,
                text=True,
            ).stdout.split()
        )
        return self.GPU_COUNT_DICT[gpu_id]

    def select_gpu(self):
        i = 0
        gpus_len = len(self.GPUS)

        while True:
            gpu = self.GPUS[i % gpus_len]
            while self.GPU_COUNT_DICT[gpu] >= self.max_overlap:
                time.sleep(5 * 60)  # process deploy to the GPU needs times
                if all([self.get_num_testing_in_OneGPU(gpu_id) >= self.max_overlap for gpu_id in self.GPUS]):
                    print(str_format(f"\n[{self.current_time_str}] Wait {self.delay_time / 60:.1f} mins for GPU release!!", fore='y'))
                    time.sleep(self.delay_time)
                else:
                    i += 1
                    gpu = self.GPUS[i % gpus_len]

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
    if len(filenames) == 0:
        print(str_format(f"[ModelNotFound] {task_dir}", fore='r'))
        return isExecute

    max_len = len(filenames[-1])
    filenames = [filename for filename in filenames if len(filename) < max_len - 5]

    for weight_tag in weight_tags:
        weight_tag_info = weight_tag.split('_')
        tag_len = len(weight_tag_info[0])
        tag_filenames = sorted([filename for filename in filenames if filename[:-3].split('_')[0][-tag_len:] == weight_tag_info[0]])
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
            'out/1224-0803_SE-3D.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.1.0e-02_IOULoss_BS-27_Set-2',
        ],
        'dev/BSUVNet-noFPM': [
            'out/1211-0325_bsuv.weight-decay.112_BSUVNet-noFPM_Adam1.0e-04_IOULoss_BS-48_Set-2',
            'out/1211-0348_bsuv.weight-decay.random.112_BSUVNet-noFPM_Adam1.0e-04_IOULoss_BS-48_Set-2',
            'out/1224-0153_bsuv.112_BSUVNet-noFPM_Adam1.0e-04.1.0e-02_IOULoss_BS-8_Set-2',
            'out/1224-0153_bsuv.224_BSUVNet-noFPM_Adam1.0e-04.1.0e-02_IOULoss_BS-8_Set-2',
            'out/0103-1811_bsuv.cv1.112_BSUVNet-noFPM_Adam1.0e-04.1.0e-02_IOULoss_BS-8_Set-1',
            'out/0103-1817_bsuv.cv2.112_BSUVNet-noFPM_Adam1.0e-04.1.0e-02_IOULoss_BS-8_Set-2',
            'out/0103-1816_bsuv.cv3.112_BSUVNet-noFPM_Adam1.0e-04.1.0e-02_IOULoss_BS-8_Set-3',
            'out/0103-1814_bsuv.cv4.112_BSUVNet-noFPM_Adam1.0e-04.1.0e-02_IOULoss_BS-8_Set-4',
        ],
        'dev/label2bg': [
            'out/1211-1607_label2bg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'dev/dataset-em,1ref,1diff': [
            'out/1223-1905_feaERD.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.0.0e+00_IOULoss_BS-48_Set-2',
        ],
        'dev/1epochBackward': [
            'out/1211-1614_1eb.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-9_Set-2',
        ],
        'dev/feaERD.1eEnNorm': [
            'out/1223-1906_feaERD.1eEnNorm.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.0.0e+00_IOULoss_BS-48_Set-2'
        ],
        'com/feaERD.label2bg': [
            'out/1223-1907_feaERD.label2bg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.0.0e+00_IOULoss_BS-48_Set-2',
        ],
        'dev/label2bgRandom': [
            'out/1211-1604_label2bgRandom.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'com/1eb.label2bg': [
            'out/1211-1632_1eb.label2bg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-9_Set-2',
        ],
        'com/feaERD.1eb.label2bg': [
            'out/1223-1915_feaERD.1eb.label2bg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.0.0e+00_IOULoss_BS-9_Set-2',
        ],
        'dev/RecAsInp': [
            'out/1211-1621_RecAsInp.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'dev/EmAsInp': [
            'out/1214-2031_EmAsInp.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2',
        ],
        'com/EmAsInp.feaERD.1eEnNorm': [
            'out/1223-1940_EmAsInp.feaERD.1eEnNorm.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.0.0e+00_IOULoss_BS-48_Set-2',
            'out/1228-0318_EmAsInp.feaERD.1eEnNorm.224_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-27_Set-2',
            'out/1228-0336_EmAsInp.feaERD.1eEnNorm.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
        ],
        'com/EmAsInp.1eb': [
            'out/1223-1943_EmAsInp.1eb.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.0.0e+00_IOULoss_BS-9_Set-2',
        ],
        'com/EmAsInp.feaERD.1eEnNorm.1eb': [
            'out/1223-1933_EmAsInp.feaERD.1eEnNorm.1eb.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.0.0e+00_IOULoss_BS-9_Set-2',
        ],
        'develop2': [
            'out/1223-2103_dev2.1e1ib4MEM.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.0.0e+00_IOULoss_BS-9_Set-2',
            'out/1225-2105_dev2.maxGAP100.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1225-2059_dev2.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1226-2232_dev2.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1227-1739_dev2.224_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1227-2016_dev2.maxGAP30.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1227-2123_dev2.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1230-1322_dev2.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1230-0920_dev2.maxGAP50.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1230-0921_dev2.maxGAP20.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1230-1322_dev2.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1230-2211_dev2.cv1.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-1',
            'out/1230-2212_dev2.cv3.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-3',
            'out/1230-2218_dev2.cv4.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-4',
            'out/1231-0022_dev2.cv1.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-1',
            'out/1231-0022_dev2.cv3.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-3',
            'out/1231-0023_dev2.cv4.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-4',
            'out/0103-0100_dev2.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
        ],
        'dev2/predInvAsBg': [
            'out/1223-1951_dev2.predInvAsBg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.0.0e+00_IOULoss_BS-9_Set-2',
        ],
        'dev2/1eb4SEM,1ib4MEM': [
            'out/1224-0916_dev2.1e1ib4MEM.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1226-2233_dev2.1e1ib4MEM.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1226-2233_dev2.1e1ib4MEM.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2/1228-0323_dev2.1e1ib4MEM.3dSEM.112.to200_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            # 'out/0101-0442_dev2.1e1ib4MEM.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            # 'out/0102-0043_dev2.1e1ib4MEM.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/0103-0105_dev2.1e1ib4MEM.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
        ],
        'com2/1e1ib4MEM.predInvAsBg': [
            'out/1230-0858_dev2.1e1ib4MEM.predInvAsBg.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1230-0902_dev2.1e1ib4MEM.predInvAsBg.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/1230-0900_dev2.1e1ib4MEM.predInvAsBg.maxGAP30.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/0101-0145_dev2.1e1ib4MEM.predInvAsBg.224_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/0101-0157_dev2.1e1ib4MEM.predInvAsBg.maxGAP20.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/0101-1935_dev2.1e1ib4MEM.predInvAsBg.maxGAP30.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/0101-0156_dev2.1e1ib4MEM.predInvAsBg.maxGAP50.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
            'out/0101-0211_dev2.1e1ib4MEM.predInvAsBg.cv1.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-1',
            'out/0101-0211_dev2.1e1ib4MEM.predInvAsBg.cv3.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-3',
            'out/0101-0212_dev2.1e1ib4MEM.predInvAsBg.cv4.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-4',
            'out/0103-0109_dev2.1e1ib4MEM.predInvAsBg.3dSEM.112_SMNet3to2D.UNet3D-UNetVgg16_Adam1.0e-04.wd0.0_IOULoss_BS-9_Set-2',
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
        # 'Prec',
        # 'Loss',
        'final',
        'checkpoint_-3:',
    ]

    gpu_provider = GPU_Provider([0], max_overlap=1)

    # subprocess.run(f"git stash", stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding='utf-8', timeout=30, shell=True)
    # for branch in task_dict.keys():
    #     merge_develop_branch(branch)
    # exit()

    for branch, task_dirs in task_dict.items():
        if task_dirs == []:
            continue

        print("")
        # print(f'git checkout {branch}')
        subprocess.run(f'git checkout {branch}', text=True, shell=True)
        time.sleep(3)

        isExecute = False
        for task_dir in task_dirs:
            cross_val = int(task_dir.split('Set-')[-1])
            if batch_testing4weights(task_dir, weight_tags, cross_val, gpu_provider) is True:
                isExecute = True
        if isExecute:
            time.sleep(5 * 60)
            print("Wait for previous branch...")
