from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from typing import Dict, List, Union
import copy
import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.nn import TAGConv

from torch_geometric.data import HeteroData
import json
import random


class TAG(nn.Module):
    def __init__(self, device,in_channels, hidden_channels, out_channels,pyg):
        super(TAG, self).__init__()
        self.device = device
        self.bert_model = AutoModel.from_pretrained("codebert-base")  # 使用预训练的 CodeBERT 模型。
        self.node_types, self.edge_types = pyg.metadata()

        num_relations = len(self.edge_types)
        #init_sizes = [pyg[x].x.shape[1] for x in self.node_types]

        self.conv1 = TAGConv(in_channels, hidden_channels)
        self.conv2 = TAGConv(hidden_channels, out_channels)
        #self.lins = torch.nn.ModuleList()
        #for i in range(len(self.node_types)):
            #lin = nn.Linear(init_sizes[i], in_channels)
            #self.lins.append(lin)

    #def trans_dimensions(self, g):
        #data = copy.deepcopy(g)
        #for node_type, lin in zip(self.node_types, self.lins):
            #data[node_type].x = lin(torch.tensor(data[node_type].x,dtype=torch.float))

        #return data

    def forward(self, pyg,delIndexes):
#第一段代码的作用是根据给定的节点的 token ID，利用 BERT 模型生成节点的嵌入向量，并将这些嵌入向量赋值给图数据对象中对应节点的特征（x）。
        token_ids_dict = pyg.token_ids_dict#从 pyg 对象中获取 token_ids_dict。这是一个字典，其中包含了 “add_node” 和 “del_node” 的 token ID。
        if token_ids_dict["add_node"].numel() != 0:#检查 “add_node” 的 token ID 数量是否不为零。如果不为零，说明有一些节点需要添加
            pyg["add_node"].x = self.bert_model(#对于需要添加的节点，使用 BERT 模型生成嵌入向量。
                torch.tensor(
                    token_ids_dict["add_node"].tolist(),
                    dtype=torch.long,
                    device=self.device,
                )#将 token ID 转换为张量，并移动到指定的设备上（CPU 或 GPU）。
            )[0][:, 0, :]#self.bert_model(...)[0][:, 0, :] 是调用 BERT 模型并获取输出的第一个元素的第一个维度，这通常是每个输入序列的第一个 token 的输出，也就是 [CLS] token 的输出。
        if token_ids_dict["del_node"].numel() != 0:
            pyg["del_node"].x = self.bert_model(
                torch.tensor(
                    token_ids_dict["del_node"].tolist(),
                    dtype=torch.long,
                    device=self.device,
                )
            )[0][:, 0, :]
        if token_ids_dict["add_node"].numel() == 0:#如果 “add_node” 的 token ID 数量为零，说明没有节点需要添加。
            pyg["add_node"].x = torch.zeros(
                (0, 768), dtype=torch.float, device=self.device
            )#将 pyg["add_node"].x 设置为零张量
        if token_ids_dict["del_node"].numel() == 0:
            pyg["del_node"].x = torch.zeros(
                (0, 768), dtype=torch.float, device=self.device
            )


        addnum_nodes = pyg['add_node'].num_nodes
        delnum_nodes = pyg['del_node'].num_nodes
        #print("pyg:",pyg)
        #print("delIndexes:",delIndexes)
        #print("num_nodes(add_node):",addnum_nodes)
        #print("num_nodes(del_node):",delnum_nodes)
        #print("num_nodes:",pyg.num_nodes)
        data = pyg
        #data = self.trans_dimensions(data)#通过线性变换函数 lin 对节点特征进行转换，可以改变节点的表示维度，便于后续转化为同构图，因为只有所有节点的特征维度相同时才能进行合并
        homogeneous_data = data.to_homogeneous()
        #print("---------------------------------")
        #print("同构图data：",homogeneous_data)
        #print("同构图x：",homogeneous_data.x)
        #print("同构图num_nodes:",homogeneous_data.num_nodes)
        edge_index, edge_type = homogeneous_data.edge_index, homogeneous_data.edge_type
        x = self.conv1(homogeneous_data.x, edge_index)
        x = self.conv2(x, edge_index)
        
        #print("---------------------------------")
        #print("卷积后data：",homogeneous_data)
        #print("卷积后x:",x)

        x = x[-delnum_nodes:]

        #print("del_node.x:",x)
        #print("---------------------------------")
        return torch.index_select(x, 0, delIndexes)
    def predict(self, pyg, delIndexes):

        token_ids_dict = pyg.token_ids_dict#从 pyg 对象中获取 token_ids_dict。这是一个字典，其中包含了 “add_node” 和 “del_node” 的 token ID。
        if token_ids_dict["add_node"].numel() != 0:#检查 “add_node” 的 token ID 数量是否不为零。如果不为零，说明有一些节点需要添加
            pyg["add_node"].x = self.bert_model(#对于需要添加的节点，使用 BERT 模型生成嵌入向量。
                torch.tensor(
                    token_ids_dict["add_node"].tolist(),
                    dtype=torch.long,
                    device=self.device,
                )#将 token ID 转换为张量，并移动到指定的设备上（CPU 或 GPU）。
            )[0][:, 0, :]#self.bert_model(...)[0][:, 0, :] 是调用 BERT 模型并获取输出的第一个元素的第一个维度，这通常是每个输入序列的第一个 token 的输出，也就是 [CLS] token 的输出。
        if token_ids_dict["del_node"].numel() != 0:
            pyg["del_node"].x = self.bert_model(
                torch.tensor(
                    token_ids_dict["del_node"].tolist(),
                    dtype=torch.long,
                    device=self.device,
                )
            )[0][:, 0, :]
        if token_ids_dict["add_node"].numel() == 0:#如果 “add_node” 的 token ID 数量为零，说明没有节点需要添加。
            pyg["add_node"].x = torch.zeros(
                (0, 768), dtype=torch.float, device=self.device
            )#将 pyg["add_node"].x 设置为零张量
        if token_ids_dict["del_node"].numel() == 0:
            pyg["del_node"].x = torch.zeros(
                (0, 768), dtype=torch.float, device=self.device
            )

        addnum_nodes = pyg['add_node'].num_nodes
        delnum_nodes = pyg['del_node'].num_nodes
        data = pyg
        #data=self.trans_dimensions(data)
        homogeneous_data = data.to_homogeneous()
        edge_index, edge_type = homogeneous_data.edge_index, homogeneous_data.edge_type
        x = self.conv1(homogeneous_data.x, edge_index)
        x = self.conv2(x, edge_index)
        x = x[-delnum_nodes:]
        return torch.index_select(x, 0, delIndexes)

class rankNet(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
        )#这是一个顺序模型，包含了四个线性层，每个线性层后面都没有激活函数。这四个线性层的输入和输出维度分别是 (num_features, 32)、(32, 16)、(16, 8) 和 (8, 1)。

        self.output = nn.Sigmoid()#这是一个 Sigmoid 函数，用于将模型的输出转换为 0 到 1 之间的值。

    def forward(self, input1, input2):
        s1 = self.model(input1)
        s2 = self.model(input2)#计算出两个分数 s1 和 s2
        return self.output(s1 - s2)#返回 s1 - s2 经过 Sigmoid 函数处理后的结果。
        #这个设计是 RankNet 排序模型的一个典型特征，它试图预测 input1 和 input2 中哪一个的排名更高。
    def predict(self, input):
        return self.model(input)#直接返回 self.model(input) 的结果，即没有经过 Sigmoid 函数处理的分数。

