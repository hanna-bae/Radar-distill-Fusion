from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sampler(nn.Module):

    def __init__(self, mode="bilinear", padding_mode="zeros"):
        """
        Initializes module
        Args:
            mode: string, Sampling mode [bilinear/nearest]
            padding_mode: string, Padding mode for outside grid values [zeros/border/reflection]
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

        if torch.__version__ >= '1.3':
            self.grid_sample = partial(F.grid_sample, align_corners=True)
        else:
            self.grid_sample = F.grid_sample

    def forward(self, input_features, grid):
        """
        Samples input using sampling grid
        Args:
            input_features: (B, C, D, H, W), Input frustum features
            grid: (B, X, Y, Z, 3), Sampling grids for input features
        Returns
            output_features: (B, C, X, Y, Z) Output voxel features
        """
        # Sample from grid
        #output = self.grid_sample(input=input_features, grid=grid, mode=self.mode, padding_mode=self.padding_mode)
        if torch.is_autocast_enabled() and input_features.dtype in (torch.float16, torch.bfloat16):
            if grid.dtype != input_features.dtype:
                grid = grid.to(dtype=input_features.dtype)
        else:
            # autocast off일 땐 둘 다 fp32로 맞춤
            if input_features.dtype != torch.float32:
                input_features = input_features.float()
            if grid.dtype != torch.float32:
                grid = grid.float()

        # --- autocast 컨텍스트에서 grid_sample 실행 ---
        with torch.cuda.amp.autocast(enabled=torch.is_autocast_enabled()):
            output = self.grid_sample(
                input=input_features.contiguous(),
                grid=grid.contiguous(),
                mode=self.mode,
                padding_mode=self.padding_mode
          )
        return output
