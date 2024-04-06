class DatasetConfig:
    currentFrDir = 'Data/currentFr'
    emptyBgDir = 'Data/emptyBg'
    recentBgDir = 'Data/recentBg'
    frame_groups = 5
    gap_range = [2, 20]
    sample4oneVideo = 200
    num_epochs = 100

    def __init__(self, isModel3D=True) -> None:
        self.concat_axis = -1 if isModel3D else 0  # axis dependent by model input_channel
