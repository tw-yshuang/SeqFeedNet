import torch

class CDNet2014_preprocess:
    # * Define FG Mode
    FG_MOVING_ONLY = 0
    FG_COUNT_SHADOW = 1

    # * Define gt pixel value meaning
    PXL_VAL_STATIC = 0
    PXL_VAL_SHADOW = 50
    PXL_VAL_NONROI = 85
    PXL_VAL_UNKNOWN = 170  # ! Belong NONROI
    PXL_VAL_MOVING = 255

    @classmethod
    def preprocessing(
        cls, gt: torch.Tensor, fg_mode
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 5
        gt_processed = torch.zeros_like(gt)
        gt_processed = -1
        match fg_mode:
            case cls.FG_MOVING_ONLY:
                gt_processed = torch.where(
                    gt >= cls.PXL_VAL_MOVING - eps, 1, gt_processed
                )
                gt_processed = torch.where(
                    gt <= cls.PXL_VAL_SHADOW + eps, 0, gt_processed
                )
            case cls.FG_COUNT_SHADOW:
                gt_processed = torch.where(
                    gt >= cls.PXL_VAL_MOVING - eps, 1, gt_processed
                )
                gt_processed = torch.where(
                    torch.bitwise_and(
                        gt <= cls.PXL_VAL_SHADOW + eps, gt >= cls.PXL_VAL_SHADOW - eps
                    ),
                    1,
                    gt_processed
                )
                gt_processed = torch.where(gt <= cls.PXL_VAL_STATIC + eps, 0 , gt_processed)
            case _:
                print("ERROR in fg_mode initiation.")
                exit()

        return gt_processed

# TODO
class LASIESTA_preprocess:
    # * Define gt pixel value meaning
    MOVING2STATIC = (255, 255, 255)
    STATIC = (0, 0, 0)
    UNKNOWN = (128, 128, 128) # Belong to NON roi
    OBJ1 = (255, 0, 0)
    OBJ2 = (0, 255, 0)
    OBJ3 = (255, 255, 0)

    @classmethod
    def preprocessing(
        cls, gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_preprocessed = torch.zeros_like(gt)
        gt_preprocessed = -1

        

        return gt_preprocessed