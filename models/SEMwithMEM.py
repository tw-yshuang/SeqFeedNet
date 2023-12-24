import torch
import torch.nn as nn

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))


class StandardNorm(nn.Module):
    def __init__(self, noise_rate=1e-4, dim=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_rate = noise_rate
        self.dim = dim

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=self.dim, keepdim=True)
        std = x.std(self.dim, unbiased=False, keepdim=True)
        return (x - mean) / (std + torch.randn(1).to(x.device) * self.noise_rate)


class SMNet2D(nn.Module):
    def __init__(self, se_model: nn.Module, me_model: nn.Module, *args, **kwargs) -> None:
        super(SMNet2D, self).__init__(*args, **kwargs)

        self.se_model = se_model
        self.me_model = me_model

    def forward(self, frame: torch.Tensor, rec_frame: torch.Tensor, features: torch.Tensor, bg_only_imgs: torch.Tensor):
        frame = frame.squeeze(1)
        rec_frame = rec_frame.squeeze(1)

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

        mask = self.me_model(torch.hstack((features, frame, rec_frame)))

        return mask, frame, features


class SMNet3to2D(nn.Module):
    def __init__(self, se_model: nn.Module, me_model: nn.Module, *args, **kwargs) -> None:
        super(SMNet3to2D, self).__init__(*args, **kwargs)

        self.se_model = se_model
        self.me_model = me_model

    def forward(self, frame: torch.Tensor, features: torch.Tensor, bg_only_imgs: torch.Tensor):
        frame = frame.squeeze(1)
        bg_only_imgs = bg_only_imgs.unsqueeze(1)

        features = self.se_model(torch.hstack((features, bg_only_imgs)))

        b, c, d = features.shape[:3]
        mask = self.me_model(torch.hstack((features.reshape(b, c * d, *features.shape[3:]), frame)))

        return mask, frame, features
