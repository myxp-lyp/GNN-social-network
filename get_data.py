import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


class MyDataset:
    def __init__(self, *, rate=0.7):
        """

        :param rate: 训练集的比例
        """
        print('构建数据集')
        self.W = pd.read_table('./files/data/edgeweight.txt', sep='\t', header=None)
        self.df = pd.read_csv('./files/data/X.csv')  # 包含所有节点
        self.labels_df = pd.read_table('./files/data/mutil_label.txt', sep=' ', header=None)  # 会多读一列nan
        self.labels_df.drop(self.labels_df.columns[-1], axis=1, inplace=True)  # 将多读的一列nan删掉

        self.keys, self.vals = self.get_edges()  # 只包含edge文件中有边关系的
        self.X = self.get_all_X(self.keys, self.vals)
        self.Y_real = self.get_all_Y(self.keys, self.vals)
        self.Y = torch.from_numpy(self.get_all_Y_encode())
        self.Y_real = torch.from_numpy(self.Y_real)
        assert self.X.shape[0] == self.Y.shape[0]

        self.COO = self.get_COO(self.keys, self.vals)
        self.edge_attr = self.get_all_weights()
        assert self.COO.shape[1] == self.edge_attr.shape[0]  # 确保边的数量相同

        """组装成图数据集"""
        self.X = torch.from_numpy(self.X[:, 1:]).to(torch.float)  # 第一列索引不要
        self.COO = torch.from_numpy(self.COO).to(torch.int64)
        self.edge_attr = torch.from_numpy(self.edge_attr).to(torch.float)
        cut = int(self.X.shape[0] * rate)
        self.train_mask = torch.tensor([True if i < cut else False for i in range(self.X.shape[0])], dtype=torch.bool)
        self.test_mask = torch.tensor([False if i < cut else True for i in range(self.X.shape[0])], dtype=torch.bool)

        self.data = Data(x=self.X, y=self.Y, edge_index=self.COO, edge_attr=self.edge_attr, train_mask=self.train_mask,
                         test_mask=self.test_mask)
        print('图数据集构建完成！')

    # 获取COO矩阵
    def get_COO(self, keys, vals):
        print('获取COO矩阵')
        assert len(keys) == len(vals)
        COO_src = np.array([])
        COO_dest = np.array([])
        for key, nbs in zip(keys, vals):
            if not -1 in nbs:  # 对有邻居节点的节点操作
                # 没有添加自环
                src = [key for i in range(len(nbs))]
                dest = nbs
                COO_src = np.concatenate([COO_src, src])
                COO_dest = np.concatenate([COO_dest, dest])
        return np.array([COO_src, COO_dest], dtype=np.int32)

    # 获取所有节点和邻居的对应数组
    def get_edges(self):
        print('获取所有邻居')
        """

        :return: keys 节点索引
        :return vals 节点对应的邻居集合
        """
        keys = []
        vals = []
        tb = pd.read_table('./files/data/edge.txt', sep='\n', on_bad_lines='skip', skip_blank_lines=False, header=None)
        nums = tb.shape[0]
        for index in np.arange(0, nums, 2):  # 一次遍历两行
            current_node_row = tb.loc[index, :]
            current_node = list(map(lambda x: int(x), current_node_row.item().strip().split(' ')))
            assert len(current_node) == 1
            keys.append(current_node[0])

            nbs_nodes_row = tb.loc[index + 1, :]
            if float == type(nbs_nodes_row.item()):  # 没有邻居节点
                vals.append([-1])  # 用-1表示没有邻居节点
            else:
                nbs_nodes = list(map(lambda x: int(x), nbs_nodes_row.item().strip().split(' ')))
                vals.append(nbs_nodes)
        assert len(keys) == len(vals)
        return keys, vals

    def get_all_Y_encode(self):
        y = np.load('./files/data/encoder_label.npy')
        return y

    # 获取src到所有邻居节点的边的权重
    def get_weight(self, src, keys, vals, w):
        print('获取所有权重')
        assert src in keys
        edge_cnt = 0  # 边计数
        for index in range(len(keys)):
            nbs = vals[index]  # 拿到邻居节点集合
            if not -1 in nbs:
                edge_cnt += len(nbs)  # 边计数增加
            if keys[index] == src:  # 找到当前节点
                # 计算起始索引
                start = edge_cnt - len(nbs) + 1
                break
        if -1 in nbs:  # 最终找到的邻居集合为空
            assert len(nbs) == len([0.0])
            return [0.0]
        if edge_cnt + 1 == len(w):  # 最后一组边
            assert len(nbs) == len(w[start:])
            return w[start:]
        else:
            assert len(nbs) == len(w[start:edge_cnt + 1])
            return w[start:edge_cnt + 1]

    # 将所有的边权重放在一个数组里
    def get_all_weights(self):
        w = self.W.loc[0, :].item().strip().split(' ')
        w = list(map(lambda x: float(x), w))  # 196158条边
        return np.array(w).reshape(-1, 1)

    # 获取所有节点的索引号
    def get_all_node_idx(self, keys, vals):
        all_nodes = set([])
        for key, val in zip(keys, vals):
            all_nodes.add(key)
            for item in val:
                if item != -1:
                    all_nodes.add(item)
        return list(all_nodes)

    # 获取所有的标签
    def get_all_Y(self, keys, vals):
        print('获取标签')
        assert self.labels_df.shape[0] == self.df.shape[0]
        # all_nodes = sorted(self.get_all_node_idx(keys, vals))
        # return self.labels_df.iloc[all_nodes, :]
        return np.array(self.labels_df)

    # 进行数据预处理获取所有节点的特征矩阵
    def get_all_X(self, keys, vals):
        print('获取特征矩阵')
        # all_nodes = sorted(self.get_all_node_idx(keys, vals))  # 索引号从小到大排序
        # df = self.df[self.df[self.df.columns[0]].isin(all_nodes)]
        # df_X = pd.concat([df[df.columns[0]], df[df.columns[2:]]], axis=1)  # 取出所有节点的X

        df_X = pd.concat([self.df[self.df.columns[0]], self.df[self.df.columns[2:]]], axis=1)  # 取出所有节点的X

        """进行归一化"""
        print('归一化')
        cat_cols = ['city', 'verified', 'province', 'gender', 'verified_type']  # 类别变量不归一化
        scale_cols = set(df_X.columns[1:]) - set(cat_cols)  # 节点编号列也不归一化
        df_scaled = (df_X[scale_cols] - df_X[scale_cols].min()) / (df_X[scale_cols].max() - df_X[scale_cols].min())
        df_scaled_X = pd.concat([df_X[df_X.columns[0]], df_scaled, df_X[cat_cols]], axis=1)  # 把编号列也加进去
        """onehot编码处理"""
        print('onehot编码')
        for col in cat_cols:
            onehot = pd.get_dummies(df_scaled_X[col])
            onehot.columns = [col + '_' + str(i + 1) for i in range(onehot.shape[1])]
            df_scaled_X.drop(col, axis=1, inplace=True)
            df_scaled_X = df_scaled_X.join(onehot)
        return np.array(df_scaled_X.loc[:, df_scaled_X.columns[1:]])  # 索引列不要


if __name__ == '__main__':
    mydataset = MyDataset(rate=0.7)
    print(mydataset.data)
