seed = 2023
import os

os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random

random.seed(seed)
import numpy as np

np.random.seed(seed)
import torch

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from typing import Dict, List, Union

import torch.nn.functional as F
from torch import nn
import argparse
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from torch_geometric.data import HeteroData
import json
import sys

from itertools import chain
from loss import *
from genPyG import *
from genPairs import *
from genBatch import *
from model import *
from eval import *
from genMiniGraphs import genAllMiniGraphs

from torch.autograd import Variable

from tqdm import tqdm
from torch.distributions.beta import Beta

# co-teaching+
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 5e-6)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.1)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = 0.1)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--num_gradual', type = int, default = 5, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=5)
# parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_workers', type=int, default=0, help='how many subprocesses to use for data loading')    # 多线程导致代码重复执行
parser.add_argument('--num_iter_per_epoch', type=int, default=40)
parser.add_argument('--epoch_decay_start', type=int, default=12)

args = parser.parse_args()

learning_rate = args.lr

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate


mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)    # 每个epoch去掉的样本比例, exponent=1时适用



# %%
def get_all_data():
    with open("miniGraphs.json") as f:
        miniGraphs = json.load(f)  # 打开名为 “miniGraphs.json” 的文件，并将其内容加载到 miniGraphs 变量中。

    dataset1 = json.load(open("dataset1.json"))  # 打开dataset1.json,导入到变量dataset1
    dataset2 = json.load(open("dataset2.json"))
    dataset3 = json.load(open("dataset3.json"))

    # 返回包含以上所有数据的元组
    return (miniGraphs, dataset1, dataset2, dataset3)


# %%
def init_model(device, pyg):
    criterion = BCEFocalLoss(reduction='none')  # 二元交叉熵损失函数，一种常用于二分类问题的损失函数。
    TAGModel = TAG(device, 768, 1 , 768 * 2, pyg)  # 初始化了一个 HAN 模型。
    TAGModel = TAGModel.to(device)  # 将模型移动到指定的设备上
    rankNetModel = rankNet(768 * 2)  # 初始化了一个 rankNet 模型，输入维度是 768 * 2。
    rankNetModel = rankNetModel.to(device)  # 将 rankNetModel 移动到指定的设备上

    optimizer = torch.optim.Adam(
        chain(TAGModel.parameters(), rankNetModel.parameters()), lr = learning_rate
    )  # 定义了一个优化器，使用的是 Adam 算法。优化器会更新 hanModel 和 rankNetModel 的参数以最小化损失函数。学习率设置为 5e-6。

    return TAGModel, rankNetModel, optimizer, criterion


# %%
def divide_lst(lst, n, k):  # 这个函数可以用于将一个列表分割成多个子列表，其中前 k - 1 个子列表的长度为 n，最后一个子列表包含剩余的元素。
    cnt = 0
    all_list = []
    for i in range(0, len(lst), n):  # 遍历 lst，步长为 n。每次循环处理一段长度为 n 的子列表。
        if cnt < k - 1:
            all_list.append(lst[i: i + n])
        else:
            all_list.append(lst[i:])
            break
        cnt = cnt + 1
    return all_list


# %%
def get_sub_minigraphs(fdirs, all_minigraphs):  # 这个函数可以用于从一个大的图集合中提取出一部分子图。
    sub_minigraphs = {}
    for fdir in fdirs:
        sub_minigraphs[fdir] = all_minigraphs[fdir]
    return sub_minigraphs


# %%
# used for k cross fold validation
def divide_minigraphs(all_minigraphs, k):  # 这个函数的主要目的是将 all_minigraphs 中的所有子图分割成 k 个部分。
    all_fdirs = []
    for fdir in all_minigraphs.keys():
        all_fdirs.append(fdir)  # 遍历 all_minigraphs 中的每一个键 fdir，并将其添加到 all_fdirs 中。
    # random.shuffle(all_fdirs)

    all_sub_minigraphs = []
    all_sub_fdirs = []
    for sub_fdirs in divide_lst(all_fdirs, int(len(all_fdirs) / k),
                                k):  # 将 all_fdirs 分割成 k 个部分，每个部分包含 int(len(all_fdirs) / k) 个元素。然后，遍历每一个部分 sub_fdirs。
        if len(sub_fdirs) == 0:
            continue
        all_sub_fdirs.append(sub_fdirs)
        all_sub_minigraphs.append(get_sub_minigraphs(sub_fdirs, all_minigraphs))

    return all_sub_minigraphs, all_sub_fdirs


# %%
def get_all_batchlist(mini_graphs, k, max_pair):  # 这个函数的主要目的是将 mini_graphs 中的所有子图分割成 k 个部分，然后为每个部分生成一组批次列表。
    all_batch_list = []
    pair_cnt = 0
    all_sub_minigraphs, all_sub_fdirs = divide_minigraphs(mini_graphs, k)

    for sub_minigraph in all_sub_minigraphs:
        all_pairs = get_all_pairs(sub_minigraph, max_pair)  # 使用 get_all_pairs 函数为 sub_minigraph 生成一组对，最多 max_pair 对。
        pair_cnt = pair_cnt + len(all_pairs)  # 更新对的总数 pair_cnt。
        batch_list = combinePair(all_pairs, 128)  # 使用 combinePair 函数将 all_pairs 组合成一组批次，每个批次包含 128 对。
        all_batch_list.append(batch_list)  # 将 batch_list 添加到 all_batch_list 中。

    return all_batch_list, all_sub_fdirs, pair_cnt





def mixup_data(x, y, alpha=5.0):
    # 将 alpha 转换为浮点类型
    lam = Beta(torch.tensor(alpha, dtype=torch.float), torch.tensor(alpha, dtype=torch.float)).sample() if alpha > 0 else 1
    index = torch.randperm(x.size()[0]).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



def mixup_criterion(pred, y_a, y_b, criterion, lam=0.5):
    loss = (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()
    return loss


# %%
# %%
def train_batchlist(batches, TAGModel, rankNetModel, optimizer, criterion, device, epoch):
    # 这个函数的主要目的是训练模型并计算损失。
    all_loss = []  # 初始化一个空列表 all_loss，用于存储每个批次的损失。
    # 将 TAGModel 和 rankNetModel 设置为训练模式。
    TAGModel.train()
    rankNetModel.train()

    for batch in batches:
        pyg1 = batch.pyg1.clone().to(device)  # 将 batch 中的 pyg1 和 pyg2 克隆并移动到指定的设备上（CPU 或 GPU）。
        pyg2 = batch.pyg2.clone().to(device)

        del_index1 = batch.del_index1.to(device)
        del_index2 = batch.del_index2.to(device)

        probs = batch.probs.to(device)
        
        x = TAGModel(pyg1, del_index1)  # 通过 TAGModel 计算 pyg1 和 pyg2 的输出。
        y = TAGModel(pyg2, del_index2)

        optimizer.zero_grad()
        preds = rankNetModel(x, y)
        #loss = criterion(preds, probs)#使用损失函数 criterion 计算 preds 和 probs 之间的损失。

        loss_con = loss_sort(preds, probs, rate_schedule[epoch], criterion)
        
        #inputs, targets_a, targets_b, lam = mixup_data(preds, probs)
        #inputs = inputs.clone().detach().requires_grad_(True)
        #logits = inputs.to(device, dtype=torch.float32)
        #logits = logits.reshape(-1, logits.size(-1))
        #targets_a = targets_a.to(device, dtype=torch.long)
        #targets_b = targets_b.to(device, dtype=torch.long)
        #targets_a = targets_a.to(device, dtype=torch.float32)  # 转换为 Float 类型
        #targets_b = targets_b.to(device, dtype=torch.float32)
        #targets_a = targets_a.flatten()
        #targets_b = targets_b.flatten()
        #print(f"logits shape: {logits.shape}")
        #print(f"targets_a: {targets_a}")
        #print(f"targets_b: {targets_b}")
        #loss_sup = mixup_criterion(inputs, targets_a, targets_b, criterion, lam)
        #loss = 0.4 * loss_sup + loss
        loss = loss_con

        loss.backward()
        optimizer.step()

        all_loss.append(loss.cpu().detach().item())  # 将损失转移到 CPU，从计算图中分离出来，然后转换为 Python 数字，并添加到 all_loss 中。

    return sum(all_loss)  # 函数返回所有批次的总损失。


# %%
# #这个函数的主要目的是在给定的设备上验证模型的性能，并计算损失。
def validate_batchlist(batches, TAGModel, rankNetModel, criterion, device):
    all_loss = []
    TAGModel.eval()
    rankNetModel.eval()  # 将 hanModel 和 rankNetModel 设置为评估模式。

    for batch in batches:
        with torch.no_grad():  # 在这个上下文管理器中，PyTorch 不会计算和存储梯度，这可以节省内存并加速计算，特别是在评估模式下
            pyg1 = batch.pyg1.clone().to(device)
            pyg2 = batch.pyg2.clone().to(device)

            del_index1 = batch.del_index1.to(device)
            del_index2 = batch.del_index2.to(device)

            probs = batch.probs.to(device)
            x = TAGModel(pyg1, del_index1)  # 通过 hanModel 计算 pyg1 和 pyg2 的输出。
            y = TAGModel(pyg2, del_index2)

            preds = rankNetModel(x, y)
            loss_con = criterion(preds, probs)
            inputs, targets_a, targets_b, lam = mixup_data(preds, probs, alpha=5.0)
            preds = preds.clone().detach().requires_grad_(True)
            logits = preds.to(device, dtype=torch.float32)
            logits = logits.reshape(-1, logits.size(-1))
            targets_a = targets_a.to(device, dtype=torch.long)
            targets_b = targets_b.to(device, dtype=torch.long)
            targets_a = targets_a.flatten()
            targets_b = targets_b.flatten()
            # print(f"logits shape: {logits.shape}")
            # print(f"targets_a: {targets_a}")
            # print(f"targets_b: {targets_b}")
            loss_sup = mixup_criterion(logits, targets_a, targets_b, lam)
            loss = 0.4 * loss_sup +  loss_con
            all_loss.append(loss.cpu().detach().item())  # 将损失转移到 CPU，从计算图中分离出来，然后转换为 Python 数字，并添加到 all_loss 中。

    return sum(all_loss)  # 函数返回所有批次的总损失。


def do_cross_fold_valid(device, K):
    all_mini_graphs, dataset1, dataset2, dataset3 = get_all_data()  # 使用 get_all_data 函数获取所有的数据和小图
    all_data = []

    high_ranking_folders = {}

    all_data.extend(dataset1)
    all_data.extend(dataset2)
    all_data.extend(dataset3)
    # print(all_data)
    # print(np.array(all_data).shape)

    random.shuffle(all_data)

    all_data_list = divide_lst(all_data, int(len(all_data) * 0.1),
                               K)  # 使用 divide_lst 函数将 all_data 分割成 K 个部分，每个部分包含 len(all_data) * 0.1 个元素。
    for i in range(0, len(all_data_list)):
        testset = all_data_list[i]  # 对于 all_data_list 中的每个部分，将其作为测试集，其余部分作为训练集。
        trainset = []

        for j in range(len(all_data_list)):
            if j != i:
                trainset.extend(all_data_list[j])

        random.shuffle(trainset)

        max_pair = 100
        # 从 all_mini_graphs 中提取出 trainset 指定的子图。
        mini_graphs = get_sub_minigraphs(trainset, all_mini_graphs)

        all_batch_list, all_sub_fdirs, pair_cnt = get_all_batchlist(
            mini_graphs, 1, max_pair=max_pair
        )
        # 获取一些可能用于后续处理的映射和数据。
        all_true_cid_map = get_true_cid_map(all_data)
        dir_to_minigraphs = get_dir_to_minigraphs(
            get_sub_minigraphs(all_data, all_mini_graphs)
        )
        # print(trainset)
        # print(dir_to_minigraphs)

        TAGModel, rankNetModel, optimizer, criterion = init_model(
            device, all_batch_list[0][0].pyg1
        )  # 初始化模型和训练设置。

        epochs = 20

        all_info = []
        test_info = []
        for epoch in range(epochs):
            total_train_loss = 0
            total_tp1 = 0
            total_fp1 = 0
            total_tp2 = 0
            total_fp2 = 0
            total_tp3 = 0
            total_fp3 = 0
            total_t = 0
            
            adjust_learning_rate( optimizer, epoch)
            
            total_train_loss = total_train_loss + train_batchlist(
                all_batch_list[0], TAGModel, rankNetModel, optimizer, criterion, device, epoch
            )

            eval(trainset, dir_to_minigraphs, TAGModel, rankNetModel, device)

            tp1, fp1, t = eval_top(
                trainset,
                dir_to_minigraphs,
                TAGModel,
                rankNetModel,
                device,
                all_true_cid_map,
                1,
            )
            tp2, fp2, t = eval_top(
                trainset,
                dir_to_minigraphs,
                TAGModel,
                rankNetModel,
                device,
                all_true_cid_map,
                2,
            )
            tp3, fp3, t = eval_top(
                trainset,
                dir_to_minigraphs,
                TAGModel,
                rankNetModel,
                device,
                all_true_cid_map,
                3,
            )
            total_t = total_t + t
            total_tp1 = total_tp1 + tp1
            total_fp1 = total_fp1 + fp1
            total_tp2 = total_tp2 + tp2
            total_fp2 = total_fp2 + fp2
            total_tp3 = total_tp3 + tp3
            total_fp3 = total_fp3 + fp3
            cur_f1_score = (
                                   2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
                           ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
            info = {}
            info["epoch"] = epoch
            info["pair_cnt"] = pair_cnt
            info["train_loss"] = total_train_loss
            info["train_f1_score"] = (
                                             2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
                                     ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
            info["train_top1_f1_precision"] = total_tp1 / (total_tp1 + total_fp1)
            info["train_top1_f1_recall"] = total_tp1 / total_t
            info["train_top2_f1_precision"] = total_tp2 / (total_tp2 + total_fp2)
            info["train_top2_f1_recall"] = total_tp2 / total_t
            info["train_top3_f1_precision"] = total_tp3 / (total_tp3 + total_fp3)
            info["train_top3_f1_recall"] = total_tp3 / total_t

            total_tp1 = 0
            total_fp1 = 0
            total_tp2 = 0
            total_fp2 = 0
            total_tp3 = 0
            total_fp3 = 0
            total_t = 0
            eval(testset, dir_to_minigraphs, TAGModel, rankNetModel, device)
            tp1, fp1, t = eval_top(
                testset,
                dir_to_minigraphs,
                TAGModel,
                rankNetModel,
                device,
                all_true_cid_map,
                1,
            )
            tp2, fp2, t = eval_top(
                testset,
                dir_to_minigraphs,
                TAGModel,
                rankNetModel,
                device,
                all_true_cid_map,
                2,
            )
            tp3, fp3, t = eval_top(
                testset,
                dir_to_minigraphs,
                TAGModel,
                rankNetModel,
                device,
                all_true_cid_map,
                3,
            )
            total_t = total_t + t
            total_tp1 = total_tp1 + tp1
            total_fp1 = total_fp1 + fp1
            total_tp2 = total_tp2 + tp2
            total_fp2 = total_fp2 + fp2
            total_tp3 = total_tp3 + tp3
            total_fp3 = total_fp3 + fp3
            cur_f1_score = (
                                   2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
                           ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
            info["test_f1_score"] = (
                                            2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
                                    ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
            info["test_top1_f1_precision"] = total_tp1 / (total_tp1 + total_fp1)
            info["test_top1_f1_recall"] = total_tp1 / total_t
            info["test_top2_f1_precision"] = total_tp2 / (total_tp2 + total_fp2)
            info["test_top2_f1_recall"] = total_tp2 / total_t
            info["test_top3_f1_precision"] = total_tp3 / (total_tp3 + total_fp3)
            info["test_top3_f1_recall"] = total_tp3 / total_t
            info["test recall@top1"] = eval_recall_topk(testset, dir_to_minigraphs, 1)
            info["test recall@top2"] = eval_recall_topk(testset, dir_to_minigraphs, 2)
            info["test recall@top3"] = eval_recall_topk(testset, dir_to_minigraphs, 3)
            info["mean_first_rank"] = eval_mean_first_rank(testset, dir_to_minigraphs)
            all_info.append(info)

            print("recall1:",info["test recall@top1"],"recall2:",info["test recall@top2"],"recall3:",info["test recall@top3"],"MFR:",info["mean_first_rank"])
            
            if epoch == 19 :
                with open(f"./crossfold_result/{i}_{epoch}.json", "w") as f:
                    json.dump(all_info, f)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = ''
    genAllMiniGraphs()
    do_cross_fold_valid(device, 10)
