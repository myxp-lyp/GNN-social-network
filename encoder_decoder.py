import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.utils.data as Data
import numpy as np
import warnings

warnings.filterwarnings('ignore')


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


def get_loader(batch_size):
    df = pd.read_table('./files/data/mutil_label.txt', sep=' ', header=None)  # 会多读一列nan
    df.drop(df.columns[-1], axis=1, inplace=True)  # 将多读的一列nan删掉
    x = torch.tensor(np.array(df[df.columns[0:]]), dtype=torch.float)
    y = torch.tensor(torch.zeros(df.shape[0], 1))
    torch_dataset = TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return loader


if __name__ == '__main__':
    best_loss = 100000
    # 实例化一个编码器对象
    edmodel = EnDecoder()

    optimizer = torch.optim.Adam(edmodel.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    ld = get_loader(batch_size=256)
    epochs = 100

    for epoch in range(epochs):
        epoch_loss = []
        for x, _ in ld:
            _, output = edmodel(x)
            loss = loss_func(output, x)   # 自编码器的训练中，样本的特征向量既是输入，也是标签
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            # print(f'batch_loss:{loss.item()}')
        print(f'train_loss:{np.mean(epoch_loss)}')
        if best_loss > np.mean(epoch_loss):
            best_loss = np.mean(epoch_loss)
            name = f'edmodel{np.mean(epoch_loss)}.pt'
            print('saving...')
            torch.save(edmodel, './files/models/edmodels/'+name)
