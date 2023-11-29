class DatasetConfig:
    currentFrDir = 'Data/currentFr'
    emptyBgDir = 'Data/emptyBg'
    recentBgDir = 'Data/recentBg'
    next_stage = 5
    frame_groups = 5
    gap_range = [2, 200]
    sample4oneVideo = 200

    def __init__(self, isModel3D=True) -> None:
        self.concat_axis = -1 if isModel3D else 0  # axis dependent by model input_channel
