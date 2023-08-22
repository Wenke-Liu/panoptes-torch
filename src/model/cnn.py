import re
import torch
import torch.nn as nn

class ResBlock(nn.Module):

    def __init__(self, size, hidden = 64, stride = 1, dil = 1):
        super(ResBlock, self).__init__()
        pad_len = int((size - 1) / 2)
        self.res = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, padding = pad_len, 
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out


class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 1):
        super().__init__()
        pad_len = int((size - 1) / 2)
        self.scale = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, stride, pad_len),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        )
        self.res1 = ResBlock(size, hidden, stride, dil)
        self.res2 = ResBlock(size, hidden, stride, dil)

    def forward(self, x):
        scaled = self.scale(x)
        out = self.res2(self.res1(scaled))
        return out
    

class CNNEncoder(nn.Module):
    def __init__(self, in_channel=3, hidden = 64, filter_size = 3, num_blocks = 5):
        super().__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    nn.Conv2d(in_channel, hidden, 3, 1, 2),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        #self.conv_end = nn.Conv2d(hidden, 1, 2, 1, 0) # 3x3 convolution to merge to 1

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        #out = self.conv_end(x)
        # TODO: optimize architecture
        out = x
        #out = torch.mean(x, dim = (2, 3), keepdim = False) # Global pooling
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 1 #2 ** (i + 1)
            stride = 2    # 
            blocks.append(ResBlockDilated(self.filter_size, hidden = hidden, dil = dilation, stride = stride))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
