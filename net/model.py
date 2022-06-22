import torch, os
import torch.nn as nn

from typing import List
from sf import TDADataset
from torch.utils.data import DataLoader

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class TimeModel(nn.Module):
    def __init__(self, cnn_in: List[tuple], cnn_out: List[tuple]):
        super(TimeModel, self).__init__()

        self.cnn_in = self._create_conv_layer(cnn_in)
        self.cnn_out = self._create_conv_layer(cnn_out)
        self.fcs = self._create_fcs_layers() 

    @staticmethod
    def _calc_pool(i: tuple, size, stride, f):
        return ((i[0] - size) / stride, (i[1] - size) / stride, f)

    def _create_conv_layer(self, config: List[tuple]):
        layers = []
        in_channels = 3 # temp => RGB
        
        for x in config:
            if type(x) == tuple and type(x[0]) == int:
                b = CNNBlock(
                        in_channels,        # channels
                        out_channels=x[1],  # filters
                        kernel_size=x[0],   # kernel size
                        stride=x[2],        # stride
                        padding=x[3],       # padding
                    )

                layers += [b]
                in_channels = x[1]
                
            elif type(x) == tuple and type(x[0]) == str:
                if x[0] == "M":
                    layers += [nn.MaxPool2d(kernel_size=x[1], stride=x[2])]

        return nn.Sequential(*layers)

    def _create_fcs_layers(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 5) # time_increasement, x_change, y_change, w_change, h_change
        )

    def forward(self, a, b):
        xa = self.cnn_in(a)
        xb = self.cnn_out(b)
        print(xa.shape, xb.shape)

        x = torch.vstack((xa, xb))
        print(x.shape)
        x = self.fcs(x)

        print(x.shape, '\n')
        


IN_CNN = [
    # Tuple: (kernel_size, n_filters, stride, padding)
    (7, 64, 2, 3),
    ("M", 2, 2), # Max pooling, kernel_size, stride
    (3, 192, 1, 1),
    ("M", 2, 2),
    (1, 128, 1, 0)
]

OUT_CNN = [
    # Tuple: (kernel_size, n_filters, stride, padding)
    (7, 64, 2, 3),
    ("M", 2, 2), # Max pooling, kernel_size, stride
    (3, 192, 1, 1),
    ("M", 2, 2),
    (1, 128, 1, 0)
]

if __name__ == '__main__':
    m = TimeModel(IN_CNN, OUT_CNN)

    TEST_DB = os.path.join('D:/Projects/eleven/timesplit/annotator/static/data', "db.json")

    test_dataset = TDADataset(TEST_DB)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
        drop_last=False
    )
    
    print(len(test_dataset))

    for a, b, y in test_loader:
        a = a.to('cpu')
        b = b.to('cpu')
        out = m(a, b)

        break