import torch
import torch.nn as nn


class SeqFeedNet(nn.Module):
    def __init__(self, se_model: nn.Module, fp_model: nn.Module, *args, **kwargs) -> None:
        super(SeqFeedNet, self).__init__(*args, **kwargs)

        self.si_encoder: nn.Module = SI_Encoder(3, 3)

        self.se_model = se_model
        self.fp_model = fp_model

    def forward(
        self,
        frame: torch.Tensor,
        empty_frame: torch.Tensor,
        features: torch.Tensor,
        bg_only_imgs: torch.Tensor,
        isDetachMEM: bool = False,
    ):
        frame = frame.squeeze(1)
        empty_frame = empty_frame.squeeze(1)

        combine_features: torch.Tensor
        if features.dim() != 5:
            combine_features = torch.hstack((features, bg_only_imgs))
        else:
            combine_features = torch.hstack((features, bg_only_imgs.unsqueeze(1)))
            combine_features = combine_features.reshape(
                combine_features.shape[0],
                combine_features.shape[1] * combine_features.shape[2],
                *combine_features.shape[3:],
            )

        features = self.se_model(combine_features)

        mask = self.fp_model(torch.hstack((features.detach() if isDetachMEM else features, frame, empty_frame)))

        return mask, frame, features


class SI_Encoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(SI_Encoder, self).__init__()
        # empty, reference, diff(reference - current), current frame encode
        self.cnn1 = nn.Sequential(nn.Conv3d(in_channel, in_channel, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(in_channel))
        self.se = SELayer(in_channel, reduction=3, is3D=True)
        self.cnn2 = nn.Sequential(nn.Conv3d(in_channel, out_channel, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(in_channel))

    def forward(self, features: torch.Tensor):
        features = self.cnn1(features)
        features = self.se(features)
        return self.cnn2(features)


# reference by https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=3, is3D=False):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if is3D else nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        avg_shape = x.shape[:-2]
        y: torch.Tensor = self.avg_pool(x).view(*avg_shape)
        y = self.fc(y).view(*avg_shape, 1, 1)
        return x * y.expand_as(x)
