import torch
from get_data import *
from torch_geometric.nn import GAT, GraphConv
import torch.nn.functional as F
import warnings
import torch.nn as nn
import matplotlib.pyplot as plt

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


class Net(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, label_dims, *, name='gat'):
        super(Net, self).__init__()
        # 有权重
        """
        使用简单图卷积时，权重在前向传播的过程中，和所有的边target节点做运算加权聚合特征
        使用图注意力卷积时，权重在前向传播的过程中作为边的属性与边注意力（边权重）做运算，然后与节点特征共同聚合成新特征
        """
        assert name in ['gcn', 'gat']
        if name == 'gat':
            self.conv1 = GAT(in_channels=in_channels, hidden_channels=hid_channels, out_channels=out_channels,
                             num_layers=1,
                             edge_dim=1)
        elif name == 'gcn':
            self.conv1 = GraphConv(in_channels=in_channels, out_channels=1)  #

        self.out_layers = torch.nn.ModuleList([torch.nn.Linear(out_channels, 1) for i in range(label_dims)])

    def forward(self, data):
        out = self.conv1(data.x, data.edge_index, data.edge_attr)

        res = torch.zeros(len(self.out_layers), data.x.shape[0], 1)
        for i, lin in enumerate(self.out_layers):
            res[i] = lin(out)
        return res


def metric(out, real_label, edmodel, *, cut=0.00001):
    print('计算召回率')
    # 通过out decoder获得预测的标签
    tp = 0  # 1-1
    fp = 0  # 0-1
    tn = 0  # 0-0
    fn = 0  # 1-0

    decode_val = edmodel.decode(out)
    # 转换为0和1
    t0 = torch.zeros_like(decode_val)
    t1 = torch.ones_like(decode_val)
    pred = torch.where(decode_val > cut, t1, t0)  # 预测标签
    acc = (pred == real_label).sum() / (pred.shape[0] * pred.shape[1])  # 准确率
    rec = ((pred == real_label) & (real_label == 1)).sum() / (real_label == 1).sum()
    return acc, rec


def train(edmodel):
    print('-' * 20)
    net.train()
    optim.zero_grad()
    out = net(dataset.data)[:, dataset.data.train_mask, :]  # 所有标签的输出（3个标签）
    label = dataset.data.y[dataset.data.train_mask]  # 所有节点的所有标签

    loss = 0
    """计算每一个标签的loss"""
    print('计算每个标签的loss')
    cnt = 0
    for l_index in range(out.shape[0]):
        cnt += 1
        l = out[l_index]  # 当前标签输出
        label_l = label[:, l_index]
        loss_l = loss_func(l.reshape(-1), label_l)
        loss += loss_l

    print(
        f'Epoch {epoch + 1}/{epochs} train_loss={loss.item() / label.shape[1]:.4f}')
    loss.backward()
    optim.step()
    return loss.item() / label.shape[1]


def eval(best_loss, *, cut):
    net.eval()
    with torch.no_grad():
        out = net(dataset.data)[:, dataset.data.test_mask, :]  # 所有标签的输出（3个标签）
        label = dataset.data.y[dataset.data.test_mask]
        real_label = dataset.Y_real[dataset.data.test_mask]
        loss = 0
        """计算每一个标签的loss"""
        for l_index in range(out.shape[0]):
            l = out[l_index]  # 当前标签概率输出
            label_l = label[:, l_index]
            loss_l = loss_func(l.reshape(-1), label_l)
            loss += loss_l
        # if best_loss > loss.item():
        #     best_loss = loss.item()
        #     print('saving model')
        #     model_name = '0416net' + str(best_loss) + '.pt'
        #     torch.save(net, './files/models/gatmodels/' + model_name)

        acc, rec = metric(out.squeeze(2).T, real_label, edmodel, cut=cut)
        print(
            f'Epoch {epoch + 1}/{epochs} ====> test_loss={loss.item() / label.shape[1]:.4f} test_recall:{rec:.4f} test_acc:{acc:.4f}')
    return loss.item() / label.shape[1], rec


if __name__ == '__main__':
    edmodel = torch.load('./files/models/edmodels/edmodel0.0006230373951096331.pt')
    edmodel.eval()

    dataset = MyDataset(rate=0.7)
    best_loss = 100000
    cut = 0.05

    epochs = 60
    num_class = 1
    hid_nums = 32
    net = Net(dataset.data.x.shape[1], hid_nums, num_class, dataset.data.y.shape[1])
    lr = 0.008
    L2 = 0.003  # 正则化防止过拟合
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=L2)
    loss_func = torch.nn.MSELoss()
    infos = []
    for epoch in range(epochs):
        train_loss = train(edmodel)
        if cut > 2.103515625e-05:
            cut /= 2
        test_loss, rec = eval(best_loss, cut=cut)
        best_loss = test_loss
        infos.append([train_loss, test_loss, rec])
    plt.figure()
    infos = np.array(infos)
    ax = plt.subplot(1, 2, 1)
    plt.plot(infos[:, 0], label='train_loss')
    plt.plot(infos[:, 1], label='test_loss')
    plt.legend()
    plt.title('loss trend')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax = plt.subplot(1, 2, 2)
    plt.plot(infos[:, 2], label='recall')
    plt.legend()
    plt.title('recall on test dataset')
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.tight_layout()
    plt.show()
