import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, input_channels=19, dropout = None):
        super().__init__()
        bias = False
        channels = 16
        stride = 4
        padding = 'same'
        kernel_size = 9
        dilation = 1

        self.kernel_size = kernel_size
        # Utilities
        self.up = nn.Upsample(scale_factor=stride)

        self.sigm = nn.Sigmoid()

        # First patch
        # Stage 0
        self.enc0 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels, out_channels=channels, dilation=dilation,
                stride=1, kernel_size=kernel_size, padding=padding, padding_mode='reflect'
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(inplace=True)
        )
        self.down = nn.MaxPool1d(kernel_size=stride, dilation=dilation)

        # Stage 1
        self.enc1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=2*channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode='reflect'
            ),
            nn.BatchNorm1d(num_features=2*channels),
            nn.ELU(inplace=True)
        )

        # Stage 2
        self.enc2 = nn.Sequential(
            nn.Conv1d(
                in_channels=2*channels, out_channels=4*channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode='reflect'
            ),
            nn.BatchNorm1d(num_features=4*channels),
            nn.ELU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(
                in_channels=(8+4)*channels, out_channels=4*channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding
            ),
            nn.BatchNorm1d(num_features=4*channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=4*channels, out_channels=4*channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding
            ),
            nn.BatchNorm1d(num_features=4*channels),
            nn.ELU(),
        )

        # Stage 3
        self.enc3 = nn.Sequential(
            nn.Conv1d(
                in_channels=4*channels, out_channels=8*channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode='reflect'
            ),
            nn.BatchNorm1d(num_features=8*channels),
            nn.ELU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(
                in_channels=(16+8)*channels, out_channels=8*channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding
            ),
            nn.BatchNorm1d(num_features=8*channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=8*channels, out_channels=8*channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding, padding_mode='reflect'
            ),
            nn.BatchNorm1d(num_features=8*channels),
            nn.ELU()
        )
        
        # Stage 4
        self.enc4 = nn.Sequential(
            nn.Conv1d(
                in_channels=8*channels, out_channels=16*channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode='reflect'
            ),
            nn.BatchNorm1d(num_features=16*channels),
            nn.ELU()
        )

        self.dec1 = nn.Sequential(
            nn.Conv1d(
                in_channels=(4+2) * channels, out_channels=2*channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding
            ),
            nn.BatchNorm1d(num_features=2*channels),
            nn.ELU(inplace=True)
        )

        self.dec0 = nn.Sequential(
            nn.Conv1d(
                in_channels=(2+1) * channels, out_channels=channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(inplace=True)
        )

        # Center prediction head
        self.center_logit = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=1, dilation=1,
                kernel_size=21, stride=1, padding=padding, #padding_mode='reflect'
            ),
            nn.MaxPool1d(kernel_size=21, stride=1)
        )

        # Duration prediction head
        self.duration_logit = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=1, dilation=1,
                kernel_size=21, stride=1, padding=padding, #padding_mode='reflect'
            ),
            nn.MaxPool1d(kernel_size=21, stride=1)
        )

    def forward(self, input, debug=False):
        # Base

        lvl0 = self.enc0(input)
        # Encoder
        x = self.down(lvl0)
        lvl1 = self.enc1(x) # 1x 

        x = self.down(lvl1) # 2x
        lvl2 = self.enc2(x) # 2x

        x = self.down(lvl2) # 4x
        lvl3 = self.enc3(x) # 4x

        x = self.down(lvl3) # 4x
        lvl4 = self.enc4(x)  # 4x

        up3 = self.up(lvl4) # 4x
        x = torch.cat((up3, lvl3), dim=1)
        out3 = self.dec3(x) # 2x

        up2 = self.up(out3) # 2x
        x = torch.cat((up2, lvl2), dim=1)
        out2 = self.dec2(x) # 2x

        up1 = self.up(out2) # 2x
        x = torch.cat((up1, lvl1), dim=1)
        out1 = self.dec1(x) # 2x
        
        up0 = self.up(out1) # 2x
        x = torch.cat((up0, lvl0), dim=1)
        out0 = self.dec0(x) # 2x

        # Outputs
        center_logit = self.center_logit(out0)
        center = self.sigm(center_logit)
        pad = 256 - 21 + 1
        center = center[:, :, pad//2:-pad//2]   # count only valid part from convolution

        x = self.duration_logit(out0)
        duration = self.sigm(x)
        duration = duration[:, :, pad//2:-pad//2]

        return center, duration
 