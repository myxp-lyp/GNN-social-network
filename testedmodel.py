import numpy
import numpy as np
import torch
import torch.nn as nn
from encoder_decoder import *
class EnDecoder(nn.Module):
    def __init__(self):
        super(EnDecoder, self).__init__()
        # 定义Encoder
        self.Encoder = nn.Sequential(
            nn.Linear(1579, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
            nn.Tanh()
        )
        # 定义Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1579),
            nn.Sigmoid()
        )

        # 定义网路的前向传播路径

    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder

    def decode(self, encode_val):
        return self.Decoder(encode_val)

def cal_(decoder, x):
    x = x.tolist()
    idxs = []
    for row_index in range(len(x)):
        row = x[row_index]
        for col_index in range(len(row)):
            if x[row_index][col_index] == 1:
                idxs.append([row_index, col_index])
    vals = []
    for index in idxs:
        vals.append(decoder[index[0]][index[1]].item())
    print(f'min{np.min(vals)} max{np.max(vals)} median{np.median(vals)} mean{np.mean(vals)}')

if __name__ == '__main__':
    ld = get_loader(102400)
    model = torch.load('./files/models/edmodel0.0006637897818634783.pt')
    model.eval()
    for x, _ in ld:
        encoder, decoder = model(x)
        cal_(decoder,x)
        break
        numpy.save()