import sys
import os
import shutil
import numpy as np
from scipy.fft import dct, idct
import pandas as pd
from datetime import datetime
import time
import torch
import torch.nn as nn
import Metrics
# import Utils
from GWN_SCPT_14_adpAdj_mask_infill import *
# from st_gt_model import STGraphTransformer
import unseen_nodes
from graph_new_sensor import generate_quotient_graph, generate_graphs, feature_extract, load_dataset, get_subgraph, get_additional_info
from torch_geometric.utils.convert import from_networkx
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
import matplotlib
import networkx as nx
from sklearn.preprocessing import MinMaxScaler # P
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["NO_PROXY"] = "overpass-api.de"
import matplotlib.pyplot as plt
import random

def set_random_seed(seed=42):
    random.seed(seed)                      # Python 内置
    np.random.seed(seed)                   # Numpy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch 单 GPU
    torch.cuda.manual_seed_all(seed)       # PyTorch 多 GPU
    torch.backends.cudnn.deterministic = True   # 确定性卷积
    torch.backends.cudnn.benchmark = False      # 关闭 benchmark
class StandardScaler: #device
    def __init__(self):
        self.u = None
        self.z = None
    def fit_transform(self, x):
        self.u = x.mean()
        self.z = x.std()
        print("mean")
        print(self.u)
        print("std")
        print(self.z)
        return (x-self.u)/self.z
    def inverse_transform(self, x):
        return x * self.z + self.u


def getXSYS_estimation(data, mode, missing_ratio=0.4, missing_ratio_test=0.4):
    """
    TRAIN: 仍按给定 missing_ratio 随机掩码，但后续空间切分只会取真实节点(不含虚拟节点)用于训练；
    TEST : 不再对真实节点随机掩码；仅对虚拟(新增)节点恒置缺失。
    """
    TRAIN_NUM = int(data.shape[0] * P.TRAINRATIO)
    XS, YS, MS = [], [], []
    virtual_nodes = getattr(P, "VIRTUAL_NODES", [])  # 新增节点列表

    if mode == 'TRAIN':
        for i in range(TRAIN_NUM):
            x = data[i:i+1, :].copy()
            y = data[i:i+1, :].copy()
            m = np.ones_like(x)

            # 随机遮挡（真实节点列随时间随机），虚拟节点随后强制置缺
            mask = np.random.rand(*x.shape) < missing_ratio
            x[mask] = 0.0
            m[mask] = 0.0

            # 强制虚拟节点恒为缺失（但训练/验证阶段不会索引到这些列）
            if len(virtual_nodes) > 0:
                x[:, virtual_nodes] = 0.0
                m[:, virtual_nodes] = 0.0

            XS.append(x); YS.append(y); MS.append(m)

    elif mode == 'TEST':
        for i in range(TRAIN_NUM, data.shape[0]):
            x = data[i:i+1, :].copy()
            y = data[i:i+1, :].copy()
            m = np.ones_like(x)

            # 测试阶段：不再对真实节点随机掩码，**仅**对虚拟节点置缺
            if len(virtual_nodes) > 0:
                x[:, virtual_nodes] = 0.0
                m[:, virtual_nodes] = 0.0
            #print(m)

            XS.append(x); YS.append(y); MS.append(m)

    XS, YS, MS = np.array(XS), np.array(YS), np.array(MS)
    XS = XS[:, :, :, np.newaxis].transpose(0, 3, 2, 1)  # [B, 1, N, 1]
    YS = YS[:, :, :, np.newaxis]                        # [B, 1, N, 1]
    MS = MS[:, :, :, np.newaxis].transpose(0, 3, 2, 1)  # [B, 1, N, 1]
    return XS, YS, MS

def setups_estimation(missing_ratio=0.4):
    if not os.path.exists(P.PATH):
        os.makedirs(P.PATH)
    if P.seed_SS == -1:
        P.seed_SS = P.seed
    torch.manual_seed(P.seed)
    torch.cuda.manual_seed(P.seed)
    np.random.seed(P.seed)
    if P.IS_EPOCH_1:
        P.EPOCH = 1
        P.PRETRN_EPOCH = 1
    print(P.KEYWORD, 'data splits (estimation task)', time.ctime())

    # ========== 时间切分 ==========
    trainXS, trainYS, trainMS = getXSYS_estimation(data, 'TRAIN')
    testXS, testYS, testMS = getXSYS_estimation(data, 'TEST')

    if P.IS_DESEASONED:
        trainXS_ds, _, trainMS_ds = getXSYS_estimation(data_ds, 'TRAIN')
        testXS_ds, _, testMS_ds = getXSYS_estimation(data_ds, 'TEST')
        trainXS = np.concatenate((trainXS, trainXS_ds), axis=1)
        testXS = np.concatenate((testXS, testXS_ds), axis=1)
        trainMS = np.concatenate((trainMS, trainMS_ds), axis=1)
        testMS = np.concatenate((testMS, testMS_ds), axis=1)

    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (1 - P.TRAINVALSPLIT))
    XS_torch_trn = trainXS[:P.train_size]
    YS_torch_trn = trainYS[:P.train_size]
    MS_torch_trn = trainMS[:P.train_size]
    XS_torch_val = trainXS[P.train_size:]
    YS_torch_val = trainYS[P.train_size:]
    MS_torch_val = trainMS[P.train_size:]

    # ========== 空间切分 ==========
    all_nodes = list(range(data.shape[1]))   # 全部节点索引
    #print(all_nodes)
    virtual_nodes = getattr(P, "VIRTUAL_NODES", [])  # 你需要在配置里加一个虚拟节点列表，例如 [101,102,103]

    real_nodes = [n for n in all_nodes if n not in virtual_nodes]

    # 只在真实节点上做划分
    spatialSplit_real = unseen_nodes.SpatialSplit(len(real_nodes),
                                                  r_trn=P.R_TRN,
                                                  r_val=0.25,
                                                  r_tst=0.05,
                                                  seed=P.seed_SS)
    spatialSplit_real.i_tst = np.arange(len(real_nodes))
    # 还原回全局索引
    i_trn = [real_nodes[i] for i in spatialSplit_real.i_trn]
    i_val = [real_nodes[i] for i in spatialSplit_real.i_val]
    i_tst = [real_nodes[i] for i in spatialSplit_real.i_tst] + virtual_nodes  # 把虚拟节点强制加入测试集

    # 构造一个伪 SpatialSplit 对象，兼容后面代码
    class SplitObj:
        pass
    spatialSplit_unseen = SplitObj()
    spatialSplit_unseen.i_trn = np.array(i_trn)
    spatialSplit_unseen.i_val = np.array(i_val)
    spatialSplit_unseen.i_tst = np.array(i_tst)

    # 全节点划分（这里 val/tst 还是覆盖所有节点）
    spatialSplit_allNod = unseen_nodes.SpatialSplit(data.shape[1],
                                                    r_trn=P.R_TRN,
                                                    r_val=min(1.0, P.R_TRN * 8 / 7),
                                                    r_tst=1.0,
                                                    seed=P.seed_SS)

    # ========== 转为 Tensor ==========
    XS_torch_train = torch.Tensor(XS_torch_trn[:, :, spatialSplit_unseen.i_trn, :])
    YS_torch_train = torch.Tensor(YS_torch_trn[:, :, spatialSplit_unseen.i_trn, :])
    MS_torch_train = torch.Tensor(MS_torch_trn[:, :, spatialSplit_unseen.i_trn, :])

    XS_torch_val_u = torch.Tensor(XS_torch_val[:, :, spatialSplit_unseen.i_val, :])
    YS_torch_val_u = torch.Tensor(YS_torch_val[:, :, spatialSplit_unseen.i_val, :])
    MS_torch_val_u = torch.Tensor(MS_torch_val[:, :, spatialSplit_unseen.i_val, :])

    XS_torch_val_a = torch.Tensor(XS_torch_val[:, :, spatialSplit_allNod.i_val, :])
    YS_torch_val_a = torch.Tensor(YS_torch_val[:, :, spatialSplit_allNod.i_val, :])
    MS_torch_val_a = torch.Tensor(MS_torch_val[:, :, spatialSplit_allNod.i_val, :])

    XS_torch_tst_u = torch.Tensor(testXS[:, :, spatialSplit_unseen.i_tst, :])
    YS_torch_tst_u = torch.Tensor(testYS[:, :, spatialSplit_unseen.i_tst, :])
    MS_torch_tst_u = torch.Tensor(testMS[:, :, spatialSplit_unseen.i_tst, :])

    XS_torch_tst_a = torch.Tensor(testXS[:, :, spatialSplit_allNod.i_tst, :])
    YS_torch_tst_a = torch.Tensor(testYS[:, :, spatialSplit_allNod.i_tst, :])
    MS_torch_tst_a = torch.Tensor(testMS[:, :, spatialSplit_allNod.i_tst, :])

    # ========== 构造 Dataset 和 DataLoader ==========
    train_data = torch.utils.data.TensorDataset(XS_torch_train, YS_torch_train, MS_torch_train)
    val_u_data = torch.utils.data.TensorDataset(XS_torch_val_u, YS_torch_val_u, MS_torch_val_u)
    val_a_data = torch.utils.data.TensorDataset(XS_torch_val_a, YS_torch_val_a, MS_torch_val_a)
    tst_u_data = torch.utils.data.TensorDataset(XS_torch_tst_u, YS_torch_tst_u, MS_torch_tst_u)
    tst_a_data = torch.utils.data.TensorDataset(XS_torch_tst_a, YS_torch_tst_a, MS_torch_tst_a)

    num_workers = 8
    pin_memory = True if device.type == 'cuda' else False

    train_iter = torch.utils.data.DataLoader(train_data, P.BATCHSIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.BATCHSIZE, shuffle=False)
    val_a_iter = torch.utils.data.DataLoader(val_a_data, P.BATCHSIZE, shuffle=False)
    tst_u_iter = torch.utils.data.DataLoader(tst_u_data, P.BATCHSIZE, shuffle=False)
    tst_a_iter = torch.utils.data.DataLoader(tst_a_data, P.BATCHSIZE, shuffle=False)

    # ========== 邻接矩阵 ==========
    adj_mx = load_adj(P.ADJPATH, P.ADJTYPE, P.DATANAME)
    print(adj_mx)
    adj_train = [torch.tensor(i[i_trn, :][:, i_trn]).to(device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[i_val, :][:, i_val]).to(device) for i in adj_mx]
    adj_val_a = [torch.tensor(i[spatialSplit_allNod.i_val, :][:, spatialSplit_allNod.i_val]).to(device) for i in adj_mx]
    adj_tst_u = [torch.tensor(i[i_tst, :][:, i_tst]).to(device) for i in adj_mx]
    adj_tst_a = [torch.tensor(i[spatialSplit_allNod.i_tst, :][:, spatialSplit_allNod.i_tst]).to(device) for i in adj_mx]

    # ========== 预训练节点抽样 ==========
    pretrn_iter = random.sample(i_trn, min(P.BATCHSIZE, len(i_trn)))
    preval_iter = i_val

    mapping_tst_u = {old: new for new, old in enumerate(i_tst)}

    return pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
        train_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a, mapping_tst_u
# Custom TensorDataset that returns indices
class TensorDatasetWithIndices(TensorDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)  # Retrieve the original data (features, targets)
        return index, data  # Return the index along with the data


def pre_evaluateModel(model, data_iter, Q1, Q2):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x in data_iter:
            dataset_keys = {i: k for i, k in enumerate(load_dataset(P.DATANAME).keys())}
            Q1_s, Q2_s = get_subgraph(Q1, dataset_keys[x], P.SUBGRAPH_SIZE), get_subgraph(Q2, dataset_keys[x], P.SUBGRAPH_SIZE)
            fQ1, fQ2 = feature_extract(Q1_s, P.FEATURES).float().to(device), feature_extract(Q2_s, P.FEATURES).float().to(device) # 64x4 tensor
            nQ1, nQ2 = from_networkx(Q1_s).to(device), from_networkx(Q2_s).to(device)

            l = model.contrast(fQ1, fQ2, nQ1.edge_index, nQ2.edge_index)
            l_sum += l.item() * P.SUBGRAPH_SIZE
            n += P.SUBGRAPH_SIZE
        return l_sum / n

def network_calls():
    Q, nearest_node, clusters, gdf_nodes, gdf_edges, traffic, hull = generate_quotient_graph(P.QUOTIENT_GRAPH_RADIUS, P.DATANAME)
    info = get_additional_info(hull)
    return

def pretrainModel(name, mode, pretrain_iter, preval_iter):
    print('pretrainModel Started ...', time.ctime())
    # model = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
    # this is a 207x4 matrix
    model = Geometric_Encoder(P.TEMPERATURE, P.FEATURES, P.GRAPH_NORM, P.HIDDEN).to(device)
    min_val_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=P.PRE_LEARN, weight_decay=P.weight_decay)
    s_time = datetime.now()
    Q, nearest_node, clusters, gdf_nodes, gdf_edges, traffic, hull = generate_quotient_graph(P.QUOTIENT_GRAPH_RADIUS, P.DATANAME)
    info = get_additional_info(hull)
    Q_nearest, _ = generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges, info, nearest=True)
    scaler = MinMaxScaler()
    scaler.fit(feature_extract(Q_nearest, P.FEATURES))

    for epoch in range(P.PRETRN_EPOCH):
        # unseen stuff trainModel here
        Q1, Q2 = generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges, info) # gives 2 networkx graphs 
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        # this used to be the data for BATCH_SIZE nodes (all data)
        # this should now be the features for BATCH_SIZE nodes (all features)
        # slice the 207x4 feature matrix into a BATCH_SIZEx4 feature matrix

        # pretrain_iter = len(0, 7, 108, 34, ...) = 100
        for x in pretrain_iter:
            dataset_keys = {i: k for i, k in enumerate(load_dataset(P.DATANAME).keys())}
            Q1_s, Q2_s = get_subgraph(Q1, dataset_keys[x], P.SUBGRAPH_SIZE), get_subgraph(Q2, dataset_keys[x], P.SUBGRAPH_SIZE)
            # x = len([0 7 108 34 ...]) = 64
            # dataset_keys = {i: k for i, k in enumerate(load_dataset().keys())}
            # indices = list(map(lambda k: dataset_keys[k], x))
            # # [0 -> 734108]
            # Q1_s = Q1.subgraph(indices).copy()
            # Q2_s = Q2.subgraph(indices).copy()
            # print(Q1, Q1_s, Q2, Q2_s)
            fQ1, fQ2 = torch.from_numpy(scaler.transform(feature_extract(Q1_s, P.FEATURES))).float().to(device), \
            torch.from_numpy(scaler.transform(feature_extract(Q2_s, P.FEATURES))).float().to(device) # 64x4 tensor
            # Q1 -> fQ1: feature matrix
            # Q1 -> nQ1: edge index, GCN doesn't like adjacency matrices
            nQ1, nQ2 = from_networkx(Q1_s).to(device), from_networkx(Q2_s).to(device)

            # fig = matplotlib.pyplot.figure()
            # nx.draw(Q1_s, pos=positions1)
            # fig.savefig("graph1.png")
            
            # fig = matplotlib.pyplot.figure()
            # nx.draw(Q2_s, pos=positions2)
            # fig.savefig("graph2.png")

            # return

            # print(fQ1, fQ2)
            # print(nQ1, nQ2)
            # x = [0, 15, 32, 79]
            # fQ1[x] = [[0.7, 0.3, 0.8, 0.5], [0.6, 0.3, 0.8, 0.5], ...] [64 x 4]
            optimizer.zero_grad()
            # loss = model.contrast([0.7, 0.3, 0.8, 0.5], [0.7, 0.3, 0.8, 0.5])
            loss = model.contrast(fQ1, fQ2, nQ1.edge_index, nQ2.edge_index)
            # loss = model.contrast(x[0].to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * P.SUBGRAPH_SIZE
            n += P.SUBGRAPH_SIZE
        train_loss = loss_sum / n
        val_loss = pre_evaluateModel(model, preval_iter, Q1, Q2)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            try:
                model_path = P.PATH + '/' + name + '.pt'
                # 尝试保存一个临时文件
                with open(model_path, 'w') as f:
                    f.write('test')
                print("权限正常，可以保存文件")
            except PermissionError:
                print(f"错误：没有写入权限，无法保存文件到 {model_path}")
            except Exception as e:
                print(f"其他错误：{e}")
            torch.save(model.state_dict(), P.PATH + '/' + name + '.pt')
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(P.PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
    e_time = datetime.now()
    print('PRETIME DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    print('pretrainModel Ended ...', time.ctime())

def getModel(name, device):
    model = gwnet(device, num_nodes=P.N_NODE, in_dim=P.CHANNEL, adp_adj=P.adp_adj, sga=P.is_SGA).to(device)
    return model
# def getModel(name, device):
#     model = STGraphTransformer(
#         in_dim=P.CHANNEL,
#         d_model=64,
#         n_heads=4,
#         n_layers=2,
#         dropout=0.0
#     ).to(device)
#     return model

def masked_loss(y_pred, y_true, mask):
    # mask: [B, 1, N, T]  (1=可见, 0=缺失)
    miss_mask = 1 - mask[:, 0, :, :].permute(0, 2, 1)  # → [B, T, N], 1=缺失
    loss = (y_pred - y_true) ** 2
    loss = loss * miss_mask
    return loss.sum() / (miss_mask.sum() + 1e-6)


def graph_constructor_helper():
    Q, nearest_node, clusters, gdf_nodes, gdf_edges, traffic, hull = generate_quotient_graph(P.QUOTIENT_GRAPH_RADIUS, P.DATANAME)
    info = get_additional_info(hull)
    Q1, _ = generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges, info, nearest=True) # gives 2 networkx graphs 
    dataset_keys = {i: k for i, k in enumerate(load_dataset(P.DATANAME).keys())}
    fQ1 = feature_extract(Q1, P.FEATURES).float().to(device)
    # Q1 -> fQ1: feature matrix
    # Q1 -> nQ1: edge index, GCN doesn't like adjacency matrices
    nQ1 = from_networkx(Q1)
    return fQ1, nQ1
def trainModel_estimation_with_pretrain(name,
                                        train_iter, val_u_iter, val_a_iter,
                                        adj_train, adj_val_u, adj_val_a,
                                        spatialSplit_unseen, spatialSplit_allNod):
    print('trainModel (Estimation + Pretrain) Started ...', time.ctime())
    model = getModel(name, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=P.LEARN, weight_decay=P.weight_decay)
    criterion = masked_loss
    s_time = datetime.now()

    # === 根据 P.IS_PRETRN 决定是否加载预训练几何编码器 ===
    if P.IS_PRETRN:
        print("Using pre-trained geometric encoder...")
        encoder = Geometric_Encoder(P.TEMPERATURE, P.FEATURES, P.GRAPH_NORM, P.HIDDEN).to(device)
        encoder.load_state_dict(torch.load(P.PATH + '/encoder.pt', map_location=device))
        encoder.eval()
        with torch.no_grad():
            fQ1, nQ1 = graph_constructor_helper()
            all_node_embed = encoder(fQ1.to(device), nQ1.edge_index.to(device)).T.detach()  # [hidden_dim, N]
            train_embed = all_node_embed[:, spatialSplit_unseen.i_trn]
            val_u_embed = all_node_embed[:, spatialSplit_unseen.i_val]
            val_a_embed = all_node_embed[:, spatialSplit_allNod.i_val]
    else:
        print("No pre-training: using zero embeddings.")
        hidden_dim = 32
        train_embed = torch.zeros((hidden_dim, len(spatialSplit_unseen.i_trn)), device=device)
        val_u_embed = torch.zeros((hidden_dim, len(spatialSplit_unseen.i_val)), device=device)
        val_a_embed = torch.zeros((hidden_dim, len(spatialSplit_allNod.i_val)), device=device)

    print('train_embed', train_embed.shape)
    print('val_u_embed', val_u_embed.shape)
    print('val_a_embed', val_a_embed.shape)

    min_val_loss = float('inf')

    for epoch in range(P.EPOCH):
        model.train()
        epoch_loss, n = 0.0, 0
        start_time = datetime.now()

        for x, y, mask in train_iter:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            y = y.squeeze(-1)
            y_pred = model(x, adj_train, train_embed)
            loss = criterion(y_pred, y, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        train_loss = epoch_loss / n
        val_u_loss = evaluateModel_estimation_with_pretrain(model, val_u_iter, val_u_embed, adj_val_u)
        val_a_loss = evaluateModel_estimation_with_pretrain(model, val_a_iter, val_a_embed, adj_val_a)

        if val_u_loss < min_val_loss:
            min_val_loss = val_u_loss
            torch.save(model.state_dict(), f"{P.PATH}/{name}_best.pt")

        epoch_time = (datetime.now() - start_time).seconds
        print(f"Epoch {epoch}, Time {epoch_time}s, Train Loss: {train_loss:.6f}, "
              f"Val_U Loss: {val_u_loss:.6f}, Val_A Loss: {val_a_loss:.6f}")
        with open(f"{P.PATH}/{name}_log.txt", 'a') as f:
            f.write(f"epoch,{epoch},time,{epoch_time},train_loss,{train_loss:.10f},"
                    f"val_u_loss,{val_u_loss:.10f},val_a_loss,{val_a_loss:.10f}\n")

    print("TRAINING FINISHED. Best val_u loss:", min_val_loss)
    print('MODEL TRAINING DURATION:', datetime.now() - s_time)

    train_score = evaluateModel_estimation_with_pretrain(model, train_iter, train_embed, adj_train)
    print(f"{name}, MAE on train: {train_score:.6f}")
    with open(f"{P.PATH}/{name}_prediction_scores.txt", 'a') as f:
        f.write(f"{name}, estimation, MAE on train, {train_score:.10f}, {train_score:.10f}\n")


def testModel_estimation_with_pretrain(name, mode, test_iter, node_indices, scaler, adj_tst_u, mapping,node_indices_plt):
    """
    node_indices: 原始节点 ID 列表
    mapping: {原始ID -> 子图索引}
    """
    print('Model Testing (Estimation)', mode, 'Started ...', time.ctime())

    # === 加载模型 ===
    model = getModel(name, device)
    model.load_state_dict(torch.load(f"{P.PATH}/{name}_best.pt", map_location=device))
    model.to(device)
    model.eval()

    # === 节点嵌入 ===
    if P.IS_PRETRN:
        print("Using pre-trained geometric encoder for testing...")
        encoder = Geometric_Encoder(P.TEMPERATURE, P.FEATURES, P.GRAPH_NORM, P.HIDDEN).to(device)
        encoder.load_state_dict(torch.load(f"{P.PATH}/encoder.pt", map_location=device))
        encoder.eval()
        with torch.no_grad():
            fQ1, nQ1 = graph_constructor_helper()
            full_embed = encoder(fQ1.to(device), nQ1.edge_index.to(device)).T.detach()   # [H, N]
            node_embed = full_embed[:, node_indices]                                    # [H, N_sub]
    else:
        print("No pre-training: using zero embeddings for testing.")
        hidden_dim = 32
        node_embed = torch.zeros((hidden_dim, len(node_indices)), device=device)

    adj_tst_u = [a.to(device) for a in adj_tst_u]

    Y_true_list, Y_pred_list, MISS_list = [], [], []

    with torch.no_grad():
        for batch in test_iter:
            if len(batch) != 3:
                print("[WARN] unexpected batch format:", type(batch), len(batch))
                continue
            x, y, ms = batch
            x = x.to(device, non_blocking=True)    # [B, C, N, T]
            y = y.to(device, non_blocking=True)    # [B, T, N, 1]
            ms = ms.to(device, non_blocking=True)  # [B, 1, N, T]

            y = y.squeeze(-1)                      # [B, T, N]
            y_pred = model(x, adj_tst_u, node_embed)  # [B, T, N]

            # [B, N, T]
            y_pred = y_pred.permute(0, 2, 1)
            y      = y.permute(0, 2, 1)

            miss_mask = 1.0 - ms[:, 0, :, :]       # [B, N, T]

            Y_true_list.append(y.cpu().numpy())
            Y_pred_list.append(y_pred.cpu().numpy())
            MISS_list.append(miss_mask.cpu().numpy())

    Y_true = np.concatenate(Y_true_list, axis=0)
    Y_pred = np.concatenate(Y_pred_list, axis=0)
    MISS   = np.concatenate(MISS_list,   axis=0)

    #print("[DBG] shapes  "  Y_pred:", Y_pred.shape)
    assert Y_true.shape == Y_pred.shape == MISS.shape

    # === 反标准化 ===
    def safe_inverse_transform_3d(arr, scaler):
        B, N, T = arr.shape
        arr_flat = arr.reshape(-1, N) # New data.shape
        print(arr_flat)
        try:
            # 如果是 sklearn 标准化器
            if hasattr(scaler, "mean_"):
                arr_flat = scaler.inverse_transform(arr_flat)
                print(arr_flat)
            else:
                # 自定义 StandardScaler
                arr_flat = arr_flat * scaler.z + scaler.u
        except Exception as e:
            print("[WARN] inverse_transform failed, fallback to identity:", e)
        return arr_flat.reshape(B, N, T)

    Y_true = safe_inverse_transform_3d(Y_true, scaler)
    #print(Y_true)
    Y_pred = safe_inverse_transform_3d(Y_pred, scaler)
    print("Y_true stats after inverse:")
    print("min:", Y_true.min(), "max:", Y_true.max(), "mean:", Y_true.mean())

    # === 指标计算（仅缺失处） ===
    eps = 1e-6
    #abs_err = np.abs(Y_true - Y_pred) * MISS
    #sq_err  = ((Y_true - Y_pred) ** 2) * MISS
    #den     = MISS.sum() + eps

    #MAE  = abs_err.sum() / den
    #RMSE = np.sqrt(sq_err.sum() / den)
    #MAPE = (np.abs((Y_true - Y_pred) / (np.abs(Y_true) + eps)) * MISS).sum() / den

    #print('*' * 40)
    #print(f"{name}, {mode}, Masked MAE: {MAE:.6f}, RMSE: {RMSE:.6f}, MAPE: {MAPE:.6f}")

    # === 保存结果 ===
    #np.save(f"{P.PATH}/{P.MODELNAME}_{mode}_{name}_prediction.npy", Y_pred)
    #np.save(f"{P.PATH}/{P.MODELNAME}_{mode}_{name}_groundtruth.npy", Y_true)
    #np.save(f"{P.PATH}/{P.MODELNAME}_{mode}_{name}_missmask.npy", MISS)

    #with open(f"{P.PATH}/{name}_prediction_scores.txt", 'a') as f:
    #    f.write(f"{name}, {mode}, Masked MAE, {MAE:.10f}, RMSE, {RMSE:.10f}, MAPE, {MAPE:.10f}\n")

    # === 热图函数 ===
    def plot_heatmap(matrix, nodes, title, save_path):
        plt.figure(figsize=(10, 6))
        #plt.imshow(matrix.T, aspect="auto", cmap="viridis", origin="lower")
        plt.imshow(matrix.T, aspect="auto", cmap="viridis", origin="lower", interpolation='none')  # 关闭插值
        plt.colorbar(label="Speed")
        plt.xlabel("Time step")
        plt.ylabel("Node ID")
        plt.yticks(range(len(nodes)), nodes)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_heatmap_1(matrix, nodes, title, save_path):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        im = plt.imshow(matrix.T, aspect="auto", cmap="viridis", origin="lower")
        plt.colorbar(im, label="Speed")
        plt.xlabel("Time step")
        plt.ylabel("Node ID")

        # 调整字体大小
        plt.yticks(range(len(nodes)), nodes, fontsize=6)

        # === 让新增节点标红 ===
        virtual_nodes = set(getattr(P, "VIRTUAL_NODES", []))
        ax = plt.gca()
        for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
            try:
                node_id = int(label.get_text())
                if node_id in virtual_nodes:
                    label.set_color("red")
                    label.set_fontweight("bold")
            except ValueError:
                pass  # 防止无法解析为整数的标签出错

        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    # === 选几个节点画图（原始ID显示，局部索引取值） ===
    subset_nodes_raw = list(node_indices_plt)  # 人为指定的顺序
    subset_nodes = [mapping[n] for n in subset_nodes_raw if n in mapping]

    B, N, T = Y_pred.shape
    # 预测和掩码子集
    pred_sub = Y_pred[:, subset_nodes, :]   # [B, N_sub, T]
    mask_sub = MISS[:, subset_nodes, :]     # [B, N_sub, T]

    # 如果有真实值
    # (假如前面没保存Y_true，这里需要取消注释)
    true_sub = Y_true[:, subset_nodes, :]  # [B, N_sub, T]

    # 展平成 (B*T, N_sub)
    pred_sub = pred_sub.reshape(-1, len(subset_nodes))
    mask_sub = mask_sub.reshape(-1, len(subset_nodes))
    true_sub = true_sub.reshape(-1, len(subset_nodes))

    # === 按节点区分：虚拟节点画预测，真实节点画真实 ===
    final_matrix = np.zeros_like(pred_sub) * np.nan  # 先全NaN
    for j, node_id in enumerate(subset_nodes_raw):
        if node_id in getattr(P, "VIRTUAL_NODES", []):
            # 新增节点 → 画预测值（缺失处）
            final_matrix[:, j] = np.where(mask_sub[:, j] == 1, pred_sub[:, j], np.nan)
        else:
            # 真实节点 → 画真实值（如果Y_true有的话）
            final_matrix[:, j] = true_sub[:, j]
            # 如果你没保存Y_true，那这里可以直接用预测值但不mask
            #final_matrix[:, j] = pred_sub[:, j]

    # === 画图 ===
    plot_heatmap(final_matrix, subset_nodes_raw,
                "Mixed GroundTruth & Predicted",
                f"{P.PATH}/heatmap_mixed.png")
    print('Model Testing Ended ...', time.ctime())

def evaluateModel_estimation_with_pretrain(model, data_iter, node_embed, adj):
    """
    评估时只在“被遮挡”的位置计算损失：
      - 原始 MS: 1=可见, 0=缺失
      - 我们用 miss_mask = 1 - MS 来在缺失处评估
    另外，采用“全数据集统一加权”的方式计算平均损失：
      loss = sum((y_pred - y_true)^2 * miss_mask) / sum(miss_mask)
    """
    model.eval()
    num_sum = 0.0   # 分子：累积的加权误差
    den_sum = 0.0   # 分母：累积的有效掩码个数
    with torch.no_grad():
        for x, y, ms in data_iter:
            x, y, ms = x.to(device), y.to(device), ms.to(device)

            # y: [B, 1, N, 1] -> [B, T, N]，这里 T=1
            y = y.squeeze(-1)            # [B, 1, N]
            # 模型输出: [B, T, N]（你的模型已按此返回）
            y_pred = model(x, adj, node_embed)  # [B, T, N]

            # miss_mask: 只取第0个通道即可，适配单/多通道；[B, 1, N, 1] -> [B, T, N]
            miss_mask = 1.0 - ms[:, 0, :, :]      # [B, N, T]
            miss_mask = miss_mask.permute(0, 2, 1)  # -> [B, T, N]

            # 累积到整体
            sq_err = (y_pred - y) ** 2            # [B, T, N]
            num_sum += (sq_err * miss_mask).sum().item()
            den_sum += miss_mask.sum().item()

    return num_sum / (den_sum + 1e-6)

################# Parameter Setting #######################
P = type('Parameters', (object,), {})()
P.TIMESTEP_IN =1
P.TIMESTEP_OUT = 1
P.CHANNEL = 1
P.BATCHSIZE = 64 # 64
P.LEARN = 0.0003
P.PRETRN_EPOCH = 100
P.EPOCH = 100# 100
P.TRAINRATIO = 0.8 # TRAIN + VAL  USE_MASK
P.TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
P.ADJTYPE = 'doubletransition'
P.MODELNAME = 'GraphWaveNet'
# P.MODELNAME = 'GraphTransformer'
P.FEATURES = 4
P.SUBGRAPH_SIZE = 64
P.QUOTIENT_GRAPH_RADIUS = 0.01
P.NETWORK_CALLS = 0
P.PRE_LEARN = 0.0001
P.GRAPH_NORM = False
P.HIDDEN = 320

data = None
data_ds = None
scaler = None
###########################################################
def get_argv():
    ''' # ARGV
    0: .py file
    1: IS_PRETRN
    2: R_TRN
    3: IS_EPOCH_1
    4: seed
    5: TEMPERATURE
    6: dataset
    7: seed_ss # spatial split
    8: IS_DESEASONED
    9: weight_decay
    10: adp_adj
    11: is_SGA
    12: FEATURES
    '''
    print('sys.argv', sys.argv)
    P.IS_PRETRN = bool(int(sys.argv[1])) if len(sys.argv) >= 2 else True
    P.R_TRN = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.7
    P.IS_EPOCH_1 = bool(int(sys.argv[3])) if len(sys.argv) >= 4 else False
    P.seed = int(sys.argv[4]) if len(sys.argv) >= 5 else 100
    P.TEMPERATURE = float(sys.argv[5]) if len(sys.argv) >= 6 else 1.0
    P.DATANAME = sys.argv[6] if len(sys.argv) >= 7 else 'METRLA'
    P.seed_SS = int(sys.argv[7]) if len(sys.argv) >= 8 else -1
    P.IS_DESEASONED = bool(int(sys.argv[8])) if len(sys.argv) >= 9 else True
    P.weight_decay = float(sys.argv[9]) if len(sys.argv) >= 10 else 0.0
    P.adp_adj = bool(int(sys.argv[10])) if len(sys.argv) >= 11 else True
    P.is_SGA = bool(int(sys.argv[11])) if len(sys.argv) >= 12 else True
    P.FEATURES = int(sys.argv[12]) if len(sys.argv) >= 13 else 4
    P.SUBGRAPH_SIZE = int(sys.argv[13]) if len(sys.argv) >= 14 else 64
    P.QUOTIENT_GRAPH_RADIUS = float(sys.argv[14]) if len(sys.argv) >= 15 else 0.01
    P.PRETRN_EPOCH = int(sys.argv[15]) if len(sys.argv) >= 16 else 100
    P.EPOCH = int(sys.argv[16]) if len(sys.argv) >= 17 else 100
    P.NETWORK_CALLS = bool(int(sys.argv[17])) if len(sys.argv) >= 18 else 0
    P.PRE_LEARN = float(sys.argv[18]) if len(sys.argv) >= 19 else P.LEARN
    P.GRAPH_NORM = bool(int(sys.argv[19])) if len(sys.argv) >= 20 else True
    P.HIDDEN = int(sys.argv[20]) if len(sys.argv) >= 21 else 320

device = torch.device('cuda:0') 
#device = torch.device("cpu")
###########################################################
def main():
    script_start_time = datetime.now()
    get_argv()
    set_random_seed(P.seed)
    # === 路径设置 ===
    P.KEYWORD = 'est_' + P.DATANAME + '_' + P.MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M") + '_' + str(os.getpid())
    P.PATH = '../save/' + P.KEYWORD

    global data
    global data_ds
    global scaler_main   # 原始数据的 scaler
    global scaler_ds     # 去季节化分量的 scaler（可选）

    scaler = None
    scaler_ds = None

    # === 数据加载 ===
    if P.DATANAME == 'METRLA':
        P.FLOWPATH = '../METRLA/metr-la.h5'
        P.n_dct_coeff = 3918
        P.ADJPATH = '../METRLA/adj_mat_with_newnodes.pkl'
        P.N_NODE = 207
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMSBAY':
        P.FLOWPATH = '../PEMSBAY/pems-bay.h5'
        P.n_dct_coeff = 4107
        P.ADJPATH = '../PEMSBAY/adj_mx_bay.pkl'
        P.N_NODE = 325
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMSD7M':
        P.FLOWPATH = '../PEMSD7M/V_228.csv'
        P.n_dct_coeff = 860
        P.ADJPATH = '../PEMSD7M/W_228.csv'
        P.N_NODE = 228
        data = pd.read_csv(P.FLOWPATH, index_col=[0]).values
    elif P.DATANAME == 'PEMS11160':
        P.BATCHSIZE = 16
        P.EPOCH = 20
        P.FLOWPATH = '../PEMS11160/pems12kSPEED2m.npy'
        P.n_dct_coeff = 2179
        P.ADJPATH = '../PEMS11160/adj_mat.pkl'
        P.N_NODE = 11160
        with open(P.FLOWPATH, 'rb') as f:
            data = np.load(f)
    else:
        raise ValueError("Unsupported dataset name")
    #TODO！！！
    print(data.shape)
    print(data)
    #P.VIRTUAL_NODES = []
    #node_indices_plt =[150, 207, 87, 208, 172, 209, 148, 210, 197]
    node_indices_plt =[150, 207, 87, 208, 172, 209, 148, 210, 197,211, 130, 212, 43, 213, 82, 214, 76, 215, 60, 216, 47, 217, 52, 218, 164, 219, 73, 220, 70, 221, 102, 222, 103, 223, 68, 224, 98, 225, 22, 226, 20, 227, 32, 228, 96, 229, 206, 230, 127, 231, 155]
        #159, 232, 163, 233, 160, 234, 187, 235, 12,
        #193, 236, 191, 237, 4, 238, 15, 239, 33, 240, 144, 241, 93, 242, 6]'''
    P.VIRTUAL_NODES = list(range(207, 246))
    #P.VIRTUAL_NODES = list(range(207, 208))
    
    print(P.VIRTUAL_NODES)

    if hasattr(P, "VIRTUAL_NODES") and len(P.VIRTUAL_NODES) > 0:
        num_virtual = len(P.VIRTUAL_NODES)
        T = data.shape[0]
        virtual_data = np.zeros((T, num_virtual), dtype=data.dtype)  # 或 np.full((T, num_virtual), np.nan)
        data = np.hstack([data, virtual_data])  # 拼接到原始 data
        print(f"[INFO] Added {num_virtual} virtual nodes. New data.shape = {data.shape}")

        # 更新 N_NODE
        P.N_NODE = data.shape[1]


    # === 网络拓扑可视化？ ===
    if P.NETWORK_CALLS:
        network_calls()
        return

    # === 去季节化（可选）===
    if P.IS_DESEASONED:
        P.CHANNEL = 2
        data_ = dct(data, axis=0)
        data_[P.n_dct_coeff:, :] = 0
        data_ds = data - idct(data_, axis=0)
    else:
        data_ds = None

    # === 标准化（分开保存两个 scaler，避免覆盖）===
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print("scaler")
    print(scaler)

    if P.IS_DESEASONED:
        scaler2 = StandardScaler()
        data_ds = scaler2.fit_transform(data_ds)

    print('data.shape', data.shape)
    if P.IS_DESEASONED:
        print('[DEBUG] use deseasoned branch: CHANNEL=2')
    
    # === 数据构造（状态估计版本）===
    # 注意：这里把 missing_ratio 往下透传，确保掩码率真的是 0.2（你的 setups_estimation 需要支持该参数）
    pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
    train_iter, val_iter, val_a_iter, test_iter, tst_a_iter, \
    adj_train, adj_val, adj_val_a, adj_test, adj_tst_a,mapping_tst_u = setups_estimation(missing_ratio=0.2)

    if P.IS_PRETRN:
        print(P.KEYWORD, 'pretraining started', time.ctime())
        pretrainModel('encoder', 'pretrain', pretrn_iter, preval_iter)
    else:
        print(P.KEYWORD, 'No pre-training')

    # === 状态估计任务训练 ===
    print(P.KEYWORD, 'training started', time.ctime())
    trainModel_estimation_with_pretrain(
        P.MODELNAME,
        train_iter,        # 训练集（unseen 的训练节点） 0.4
        val_iter,          # 验证（unseen 节点）
        val_a_iter,        # 验证（all-nodes） 12
        adj_train, adj_val, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod  # 映射
    )
    #node_indices_plt = [104, 89, 207, 34, 186]

    # === 状态估计任务测试（unseen 节点）===
    print(P.KEYWORD, 'testing started', time.ctime())
    testModel_estimation_with_pretrain(
        name=P.MODELNAME,
        mode="tst_u",
        test_iter=test_iter,                         # 测试（unseen 节点）
        node_indices=spatialSplit_unseen.i_tst,      # 与 adj_test 对齐
        scaler=scaler,                          # ★ 用原始数据的 scaler 反标准化 0.4
        adj_tst_u=adj_test ,mapping=mapping_tst_u  ,   node_indices_plt=node_indices_plt                     # 子图邻接（unseen 测试）
    )

    print('SCRIPT DURATION', datetime.now() - script_start_time)
    

if __name__ == '__main__':
    main()