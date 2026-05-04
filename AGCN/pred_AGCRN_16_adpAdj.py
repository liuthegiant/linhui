import sys
import os
import shutil
import numpy as np
from scipy.fft import dct, idct
import pandas as pd
from datetime import datetime
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import Metrics
# import Utils
from GWN_SCPT_14_adpAdj_future12step import *
from AGCRN import AGCRN
import unseen_nodes
from graph import generate_quotient_graph, generate_graphs, feature_extract, load_dataset, get_subgraph, get_additional_info
from torch_geometric.utils.convert import from_networkx
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
import statistics
import matplotlib
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

class StandardScaler:
    def __init__(self):
        self.u = None
        self.z = None
    def fit_transform(self, x):
        self.u = x.mean()
        self.z = x.std()
        return (x-self.u)/self.z
    def inverse_transform(self, x):
        return x * self.z + self.u

# Wall-clock / inference timings for paper-style tables (one run per process).
PAPER_TIMING: dict = {}


def reset_paper_timing() -> None:
    PAPER_TIMING.clear()


def _timing_stats_sec(times: list[float]) -> dict:
    """Mean / sample stdev / min / max / sum / count for batch-level timings (seconds)."""
    if not times:
        return {
            "mean_sec": 0.0,
            "stdev_sec": 0.0,
            "min_sec": 0.0,
            "max_sec": 0.0,
            "sum_sec": 0.0,
            "n": 0,
        }
    n = len(times)
    s = sum(times)
    mu = s / n
    if n < 2:
        sd = 0.0
    else:
        sd = statistics.stdev(times)
    return {
        "mean_sec": float(mu),
        "stdev_sec": float(sd),
        "min_sec": float(min(times)),
        "max_sec": float(max(times)),
        "sum_sec": float(s),
        "n": int(n),
    }


def _epoch_stats_sec(epoch_secs: list[float]) -> dict:
    if not epoch_secs:
        return {"mean_sec": 0.0, "stdev_sec": 0.0, "min_sec": 0.0, "max_sec": 0.0, "n": 0}
    n = len(epoch_secs)
    if n < 2:
        sd = 0.0
    else:
        sd = statistics.stdev(epoch_secs)
    return {
        "mean_sec": float(statistics.mean(epoch_secs)),
        "stdev_sec": float(sd),
        "min_sec": float(min(epoch_secs)),
        "max_sec": float(max(epoch_secs)),
        "n": int(n),
    }


def _put_iter_stats(prefix: str, times: list[float]) -> None:
    for k, v in _timing_stats_sec(times).items():
        PAPER_TIMING[f"{prefix}_{k}"] = v


def _trainable_params(*modules) -> int:
    n = 0
    for m in modules:
        if m is None:
            continue
        n += sum(p.numel() for p in m.parameters() if p.requires_grad)
    return n


def emit_paper_timing_summary(script_start: datetime) -> None:
    """Print and save seconds / params / throughput for cross-paper comparison."""
    wall = (datetime.now() - script_start).total_seconds()
    PAPER_TIMING["script_wall_sec"] = float(wall)
    geo = float(PAPER_TIMING.get("pretrain_geo_sec", 0.0))
    tmp = float(PAPER_TIMING.get("pretrain_temporal_sec", 0.0))
    PAPER_TIMING["pretrain_total_sec"] = geo + tmp
    core = (
        PAPER_TIMING["pretrain_total_sec"]
        + float(PAPER_TIMING.get("main_train_sec", 0.0))
        + float(PAPER_TIMING.get("test_u_all_sec", 0.0))
        + float(PAPER_TIMING.get("test_a_all_sec", 0.0))
    )
    PAPER_TIMING["core_pipeline_sec"] = float(core)

    def _fmt_ep(label: str, lst: list) -> str:
        if not lst:
            return f"  {label}: (n/a)"
        st = _epoch_stats_sec([float(x) for x in lst])
        return (
            f"  {label}: n={st['n']}, mean={st['mean_sec']:.3f}s, "
            f"stdev={st['stdev_sec']:.3f}s, min={st['min_sec']:.3f}s, max={st['max_sec']:.3f}s"
        )

    def _fmt_it(label: str, d: dict) -> str:
        if not d or d.get("n", 0) == 0:
            return f"  {label}: (n/a)"
        return (
            f"  {label}: n={d['n']}, mean={d['mean_sec']:.4f}s, stdev={d['stdev_sec']:.4f}s, "
            f"sum={d['sum_sec']:.3f}s, min={d['min_sec']:.4f}s, max={d['max_sec']:.4f}s"
        )

    geo_ep = PAPER_TIMING.get("pretrain_geo_epoch_sec_list") or []
    tmp_ep = PAPER_TIMING.get("pretrain_temporal_epoch_sec_list") or []
    main_ep = PAPER_TIMING.get("main_train_epoch_sec_list") or []
    if geo_ep:
        PAPER_TIMING["pretrain_geo_epoch_stats"] = _epoch_stats_sec([float(x) for x in geo_ep])
    if tmp_ep:
        PAPER_TIMING["pretrain_temporal_epoch_stats"] = _epoch_stats_sec([float(x) for x in tmp_ep])
    if main_ep:
        PAPER_TIMING["main_train_epoch_stats"] = _epoch_stats_sec([float(x) for x in main_ep])

    path = P.PATH + "/paper_timing.json"
    os.makedirs(P.PATH, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(PAPER_TIMING, f, indent=2, sort_keys=True)

    lines = [
        "=== PAPER_TIMING (wall-clock seconds unless noted) ===",
        f"pretrain_geo_sec: {PAPER_TIMING.get('pretrain_geo_sec', 0):.3f}",
        f"pretrain_temporal_sec: {PAPER_TIMING.get('pretrain_temporal_sec', 0):.3f}",
        f"pretrain_total_sec: {PAPER_TIMING.get('pretrain_total_sec', 0):.3f}",
        f"core_pipeline_sec (pretrain+main_train+test_u+test_a): {PAPER_TIMING.get('core_pipeline_sec', 0):.3f}",
        f"main_train_sec: {PAPER_TIMING.get('main_train_sec', 0):.3f}",
        f"main_train_epoch_time_mean_sec: {PAPER_TIMING.get('main_train_epoch_time_mean_sec', 0):.3f}",
        f"forecast_model_params (trainable): {int(PAPER_TIMING.get('forecast_model_params', 0))}",
        f"fusion_module_params (trainable): {int(PAPER_TIMING.get('fusion_module_params', 0))}",
        f"test_u_all_sec (load+infer+metrics): {PAPER_TIMING.get('test_u_all_sec', 0):.3f}",
        f"test_u_eval_forward_sec: {PAPER_TIMING.get('test_u_eval_forward_sec', 0):.4f}",
        f"test_u_predict_forward_sec: {PAPER_TIMING.get('test_u_predict_forward_sec', 0):.4f}",
        f"test_u_predict_throughput_samples_per_s: {PAPER_TIMING.get('test_u_predict_throughput_samples_per_s', 0):.2f}",
        f"test_a_all_sec (load+infer+metrics): {PAPER_TIMING.get('test_a_all_sec', 0):.3f}",
        f"test_a_eval_forward_sec: {PAPER_TIMING.get('test_a_eval_forward_sec', 0):.4f}",
        f"test_a_predict_forward_sec: {PAPER_TIMING.get('test_a_predict_forward_sec', 0):.4f}",
        f"test_a_predict_throughput_samples_per_s: {PAPER_TIMING.get('test_a_predict_throughput_samples_per_s', 0):.2f}",
        f"script_wall_sec: {PAPER_TIMING.get('script_wall_sec', 0):.3f}",
        f"(saved {path})",
    ]

    def _iter_dict(prefix: str) -> dict:
        if f"{prefix}_mean_sec" not in PAPER_TIMING:
            return {}
        keys = ("mean_sec", "stdev_sec", "min_sec", "max_sec", "sum_sec", "n")
        out = {}
        for k in keys:
            kk = f"{prefix}_n" if k == "n" else f"{prefix}_{k}"
            out[k] = PAPER_TIMING.get(kk, 0)
        return out

    time_sum_lines = [
        "=== TIME SUMMARY (training & inference) ===",
        "[Overall — wall clock]",
        f"  script_wall_sec: {PAPER_TIMING.get('script_wall_sec', 0):.3f}",
        f"  pretrain_total_sec: {PAPER_TIMING.get('pretrain_total_sec', 0):.3f}",
        f"  main_train_sec: {PAPER_TIMING.get('main_train_sec', 0):.3f}",
        f"  test_u_all_sec: {PAPER_TIMING.get('test_u_all_sec', 0):.3f}",
        f"  test_a_all_sec: {PAPER_TIMING.get('test_a_all_sec', 0):.3f}",
        "",
        "[Epoch — pretrain geometric]",
        _fmt_ep("per-epoch duration", geo_ep),
        "",
        "[Iteration — pretrain geometric, per inner step]",
        _fmt_it("batch/step", _iter_dict("pretrain_geo_iter")),
        "",
        "[Epoch — pretrain temporal]",
        _fmt_ep("per-epoch duration", tmp_ep),
        "",
        "[Iteration — pretrain temporal, per mini-batch]",
        _fmt_it("batch", _iter_dict("pretrain_temporal_iter")),
        "",
        "[Epoch — main forecast training]",
        _fmt_ep("per-epoch duration", main_ep),
        "",
        "[Iteration — main forecast training, per train batch]",
        _fmt_it("batch", _iter_dict("main_train_iter")),
        "",
        "[Inference test_u — per DataLoader batch]",
        _fmt_it("eval (validation forward)", _iter_dict("test_u_eval_batch")),
        _fmt_it("predict (test forward)", _iter_dict("test_u_predict_batch")),
        "",
        "[Inference test_a — per DataLoader batch]",
        _fmt_it("eval (validation forward)", _iter_dict("test_a_eval_batch")),
        _fmt_it("predict (test forward)", _iter_dict("test_a_predict_batch")),
    ]
    time_block = "\n".join(time_sum_lines)
    print("\n".join(lines))
    print()
    print(time_block)
    print("PAPER_TIMING_JSON", json.dumps(PAPER_TIMING, sort_keys=True))
    block = "\n".join(lines) + "\n\n" + time_block + "\n"
    with open(P.PATH + "/paper_timing.txt", "w", encoding="utf-8") as f:
        f.write(block)


def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * P.TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - P.TIMESTEP_OUT - P.TIMESTEP_IN + 1):
            x = data[i:i+P.TIMESTEP_IN, :]
            y = data[i+P.TIMESTEP_IN:i+P.TIMESTEP_IN+P.TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - P.TIMESTEP_IN,  data.shape[0] - P.TIMESTEP_OUT - P.TIMESTEP_IN + 1):
            x = data[i:i+P.TIMESTEP_IN, :]
            y = data[i+P.TIMESTEP_IN:i+P.TIMESTEP_IN+P.TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

# Custom TensorDataset that returns indices
class TensorDatasetWithIndices(TensorDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)  # Retrieve the original data (features, targets)
        return index, data  # Return the index along with the data

def setups():
    # make save folder
    if not os.path.exists(P.PATH):
        os.makedirs(P.PATH)
    # seed
    if P.seed_SS == -1:
        P.seed_SS = P.seed
    torch.manual_seed(P.seed)
    torch.cuda.manual_seed(P.seed)
    np.random.seed(P.seed)
    # epoch
    if P.IS_EPOCH_1:
        P.EPOCH = 1
        P.PRETRN_EPOCH = 1
    print(P.KEYWORD, 'data splits', time.ctime())
    # test split temporal
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    testXS, testYS = getXSYS(data, 'TEST')
    if P.IS_DESEASONED:
        trainXS_ds, trainYS = getXSYS(data_ds, 'TRAIN') # all the Y are de-seasoned
        testXS_ds, testYS = getXSYS(data_ds, 'TEST') # all the Y are de-seasoned
        trainXS = np.concatenate((trainXS, trainXS_ds), axis=1) # the Xs are combined between normal and de-seasoned
        testXS = np.concatenate((testXS, testXS_ds), axis=1) # the Xs are combined between normal and de-seasoned
    # trn val split
    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (1-P.TRAINVALSPLIT))
    XS_torch_trn = trainXS[:P.train_size,:,:,:]
    YS_torch_trn = trainYS[:P.train_size,:,:,:]
    XS_torch_val = trainXS[P.train_size:P.trainval_size,:,:,:]
    YS_torch_val = trainYS[P.train_size:P.trainval_size,:,:,:]
    # spatial split
    spatialSplit_unseen = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=.1, r_tst=.2, seed=P.seed_SS)
    spatialSplit_allNod = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=min(1.0,P.R_TRN*8/7), r_tst=1.0, seed=P.seed_SS)
    print('spatialSplit_unseen', spatialSplit_unseen)
    print(spatialSplit_unseen.i_trn)
    print(spatialSplit_unseen.i_val)
    print(spatialSplit_unseen.i_tst)
    print('spatialSplit_allNod', spatialSplit_allNod)
    print(spatialSplit_allNod.i_trn)
    print(spatialSplit_allNod.i_val)
    print(spatialSplit_allNod.i_tst)
    XS_torch_train = torch.Tensor(XS_torch_trn[:,:,spatialSplit_unseen.i_trn,:])
    YS_torch_train = torch.Tensor(YS_torch_trn[:,:,spatialSplit_unseen.i_trn,:])
    XS_torch_val_u = torch.Tensor(XS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    YS_torch_val_u = torch.Tensor(YS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    XS_torch_val_a = torch.Tensor(XS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    YS_torch_val_a = torch.Tensor(YS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    XS_torch_tst_u = torch.Tensor(testXS[:,:,spatialSplit_unseen.i_tst,:])
    YS_torch_tst_u = torch.Tensor(testYS[:,:,spatialSplit_unseen.i_tst,:])
    XS_torch_tst_a = torch.Tensor(testXS[:,:,spatialSplit_allNod.i_tst,:])
    YS_torch_tst_a = torch.Tensor(testYS[:,:,spatialSplit_allNod.i_tst,:])
    print('train.shape', XS_torch_train.shape, YS_torch_train.shape)
    print('val_u.shape', XS_torch_val_u.shape, YS_torch_val_u.shape)
    print('val_a.shape', XS_torch_val_a.shape, YS_torch_val_a.shape)
    print('tst_u.shape', XS_torch_tst_u.shape, YS_torch_tst_u.shape)
    print('tst_a.shape', XS_torch_tst_a.shape, YS_torch_tst_a.shape)
    # torch dataset
    train_data = torch.utils.data.TensorDataset(XS_torch_train, YS_torch_train)
    # 207 x K x D
    val_u_data = torch.utils.data.TensorDataset(XS_torch_val_u, YS_torch_val_u)
    val_a_data = torch.utils.data.TensorDataset(XS_torch_val_a, YS_torch_val_a)
    tst_u_data = torch.utils.data.TensorDataset(XS_torch_tst_u, YS_torch_tst_u)
    tst_a_data = torch.utils.data.TensorDataset(XS_torch_tst_a, YS_torch_tst_a)
    # torch dataloader
    train_iter = torch.utils.data.DataLoader(train_data, P.BATCHSIZE, shuffle=True)
    # [64 x K x D, 64 x K x D, ...]
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.BATCHSIZE, shuffle=False)
    val_a_iter = torch.utils.data.DataLoader(val_a_data, P.BATCHSIZE, shuffle=False)
    tst_u_iter = torch.utils.data.DataLoader(tst_u_data, P.BATCHSIZE, shuffle=False)
    tst_a_iter = torch.utils.data.DataLoader(tst_a_data, P.BATCHSIZE, shuffle=False)
    # adj matrix spatial split
    adj_mx = load_adj(P.ADJPATH, P.ADJTYPE, P.DATANAME)
    adj_train = [torch.tensor(i[spatialSplit_unseen.i_trn,:][:,spatialSplit_unseen.i_trn]).to(device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[spatialSplit_unseen.i_val,:][:,spatialSplit_unseen.i_val]).to(device) for i in adj_mx]
    adj_val_a = [torch.tensor(i[spatialSplit_allNod.i_val,:][:,spatialSplit_allNod.i_val]).to(device) for i in adj_mx]
    adj_tst_u = [torch.tensor(i[spatialSplit_unseen.i_tst,:][:,spatialSplit_unseen.i_tst]).to(device) for i in adj_mx]
    adj_tst_a = [torch.tensor(i[spatialSplit_allNod.i_tst,:][:,spatialSplit_allNod.i_tst]).to(device) for i in adj_mx]
    print('adj_train', len(adj_train), adj_train[0].shape, adj_train[1].shape)
    print('adj_val_u', len(adj_val_u), adj_val_u[0].shape, adj_val_u[1].shape)
    print('adj_val_a', len(adj_val_a), adj_val_a[0].shape, adj_val_a[1].shape)
    print('adj_tst_u', len(adj_tst_u), adj_tst_u[0].shape, adj_tst_u[1].shape)
    print('adj_tst_a', len(adj_tst_a), adj_tst_a[0].shape, adj_tst_a[1].shape)
    # PRETRAIN data loader
    # pretrn_iter = [random.sample(list(spatialSplit_unseen.i_trn), P.BATCHSIZE) for _ in range(10)]
    # this doesn't have to be tied to metr la nodes necessarily.
    # all we need is some random assortment of OSM nodes, with density and scale roughly matching the METR-LA dataset.
    pretrn_iter = random.sample(list(spatialSplit_unseen.i_trn), P.BATCHSIZE)
    preval_iter = list(spatialSplit_unseen.i_val)
    # print('pretrn_iter.dataset.tensors[0].shape', pretrn_iter.dataset.tensors[0].shape)
    # print('preval_iter.dataset.tensors[0].shape', preval_iter.dataset.tensors[0].shape)
    # print
    for k, v in vars(P).items():
        print(k,v)
    return pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
        train_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a

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

def pre_evaluateModel_temporal(model, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x in data_iter:
            l = model.contrast(x[0].to(device))
            l_sum += l.item() * x[0].shape[0]
            n += x[0].shape[0]
    return l_sum / max(n, 1)

def network_calls():
    Q, nearest_node, clusters, gdf_nodes, gdf_edges, traffic, hull = generate_quotient_graph(P.QUOTIENT_GRAPH_RADIUS, P.DATANAME)
    info = get_additional_info(hull)
    return

def pretrainModel(name, mode, pretrain_iter, preval_iter):
    print('pretrainModel Started ...', time.ctime())
    geo_epoch_secs: list[float] = []
    geo_batch_times: list[float] = []
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
            _t_step0 = time.perf_counter()
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
            geo_batch_times.append(time.perf_counter() - _t_step0)
        train_loss = loss_sum / n
        val_loss = pre_evaluateModel(model, preval_iter, Q1, Q2)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), P.PATH + '/' + name + '.pt')
        endtime = datetime.now()
        geo_epoch_secs.append((endtime - starttime).total_seconds())
        epoch_time = int(geo_epoch_secs[-1])
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(P.PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
    e_time = datetime.now()
    print('PRETIME DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    PAPER_TIMING['pretrain_geo_sec'] = float((e_time - s_time).total_seconds())
    PAPER_TIMING["pretrain_geo_epoch_sec_list"] = geo_epoch_secs
    _put_iter_stats("pretrain_geo_iter", geo_batch_times)
    print('pretrainModel Ended ...', time.ctime())

def pretrainModel_temporal(name, spatialSplit_unseen):
    print('pretrainModel_temporal Started ...', time.ctime())
    temporal_epoch_secs: list[float] = []
    temporal_batch_times: list[float] = []
    model = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
    min_val_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=P.PRE_LEARN, weight_decay=P.weight_decay)
    s_time = datetime.now()

    source = _get_temporal_source_data()
    trn = torch.tensor(source[:P.train_size, spatialSplit_unseen.i_trn]).float().T
    val = torch.tensor(source[:P.train_size, spatialSplit_unseen.i_val]).float().T
    trn_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(trn), P.BATCHSIZE, shuffle=True
    )
    val_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val), P.BATCHSIZE, shuffle=False
    )

    for epoch in range(P.PRETRN_EPOCH):
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x in trn_iter:
            _tb0 = time.perf_counter()
            optimizer.zero_grad()
            loss = model.contrast(x[0].to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x[0].shape[0]
            n += x[0].shape[0]
            temporal_batch_times.append(time.perf_counter() - _tb0)
        train_loss = loss_sum / max(n, 1)
        val_loss = pre_evaluateModel_temporal(model, val_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(P.PRETRAIN_CKPT_DIR, name + '.pt'))
        endtime = datetime.now()
        temporal_epoch_secs.append((endtime - starttime).total_seconds())
        epoch_time = int(temporal_epoch_secs[-1])
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(os.path.join(P.PRETRAIN_CKPT_DIR, name + '_log.txt'), 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % (
                "epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss
            ))
    e_time = datetime.now()
    print('PRETIME DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    PAPER_TIMING['pretrain_temporal_sec'] = float((e_time - s_time).total_seconds())
    PAPER_TIMING["pretrain_temporal_epoch_sec_list"] = temporal_epoch_secs
    _put_iter_stats("pretrain_temporal_iter", temporal_batch_times)
    print('pretrainModel_temporal Ended ...', time.ctime())


class AGCRNGWNCallAdapter(nn.Module):
    """
    AGCRN 输入为 (B, T, N, C)，本脚本其余部分沿用 GWN 的 (B, C, N, T) 与 model(x, adj, embed)。
    预训练得到的 embed 仍参与融合逻辑；AGCRN 本体使用可学习 node_embeddings，不使用 adj/embed。

    GWN 对节点数不敏感；AGCRN 的 node_embeddings 维度固定。此处将各 split 的节点数统一到
    P.AGCRN_NUM_NODES_MAX：不足则沿节点维零填充，并用 node_mask 在图卷积中屏蔽填充节点，
    输出再裁回真实节点数以与标签对齐。
    """

    def __init__(self, agcrn: AGCRN):
        super().__init__()
        self.agcrn = agcrn

    def forward(self, x, adj, embed):
        del adj, embed
        b, c, n, _t = x.shape
        nmax = int(self.agcrn.num_node)
        if n > nmax:
            raise ValueError(f'batch nodes {n} > AGCRN_NUM_NODES_MAX {nmax}')
        if n < nmax:
            pad = x.new_zeros(b, c, nmax - n, _t)
            x = torch.cat([x, pad], dim=2)
        node_mask = x.new_zeros(nmax)
        node_mask[:n] = 1.0
        source = x.permute(0, 3, 2, 1).contiguous()
        dummy_targets = torch.zeros(
            b,
            self.agcrn.horizon,
            nmax,
            self.agcrn.output_dim,
            device=x.device,
            dtype=x.dtype,
        )
        out = self.agcrn(source, dummy_targets, node_mask=node_mask)
        if n < nmax:
            out = out[:, :, :n, :].contiguous()
        return out


def getModel(name, device):
    args = type('AGCRNArgs', (), {})()
    args.num_nodes = int(P.AGCRN_NUM_NODES_MAX)
    args.input_dim = P.CHANNEL
    args.rnn_units = P.AGCRN_RNN_UNITS
    args.output_dim = 1
    args.horizon = P.TIMESTEP_OUT
    args.num_layers = P.AGCRN_NUM_LAYERS
    args.default_graph = True
    args.embed_dim = P.AGCRN_EMBED_DIM
    args.cheb_k = P.AGCRN_CHEB_K
    return AGCRNGWNCallAdapter(AGCRN(args).to(device))

def evaluateModel(model, criterion, data_iter, adj, embed, return_batch_times=False):
    model.eval()
    torch.cuda.empty_cache()
    l_sum, n = 0.0, 0
    batch_times: list[float] | None = [] if return_batch_times else None
    with torch.no_grad():
        for x, y in data_iter:
            t0 = time.perf_counter()
            y_pred = model(x.to(device), adj, embed)
            if batch_times is not None:
                batch_times.append(time.perf_counter() - t0)
            l = criterion(y_pred, y.to(device))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
    out = l_sum / n
    if return_batch_times:
        return out, batch_times  # type: ignore[return-value]
    return out

def predictModel(model, data_iter, adj, embed, return_batch_times=False):
    YS_pred = []
    model.eval()
    batch_times: list[float] | None = [] if return_batch_times else None
    with torch.no_grad():
        for x, y in data_iter:
            t0 = time.perf_counter()
            YS_pred_batch = model(x.to(device), adj, embed)
            if batch_times is not None:
                batch_times.append(time.perf_counter() - t0)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    if return_batch_times:
        return YS_pred, batch_times  # type: ignore[return-value]
    return YS_pred

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

def trainModel(name, mode,
        train_iter, val_u_iter, val_a_iter,
        adj_train, adj_val_u, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod):
    print('trainModel Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', P.TIMESTEP_IN, P.TIMESTEP_OUT)
    model = getModel(name, device)
    min_val_u_loss = np.inf
    min_val_a_loss = np.inf
    criterion = nn.L1Loss()
    gate_module = None
    temp_full_embed, geo_full_embed = None, None
    if P.IS_PRETRN and P.PRETRN_MODE == 'dual' and P.FUSION_MODE in ('learned', 'residual', 'residual_gated', 'temporal_delta'):
        temp_full_embed = _get_temporal_full_embed()
        geo_full_embed = _get_geometric_full_embed()
        if P.FUSION_MODE == 'learned':
            gate_module = LearnableFusionGate(
                embed_dim=temp_full_embed.shape[0],
                hidden_dim=P.GATE_HIDDEN
            ).to(device)
        else:
            if P.FUSION_MODE in ('residual', 'residual_gated'):
                gate_module = ResidualDeltaFusion(
                    embed_dim=temp_full_embed.shape[0],
                    hidden_dim=P.GATE_HIDDEN,
                    use_gate=(P.FUSION_MODE == 'residual_gated'),
                ).to(device)
            else:
                gate_module = TemporalBaseDeltaFusion(
                    embed_dim=temp_full_embed.shape[0],
                    hidden_dim=P.GATE_HIDDEN,
                    use_projection=True,
                ).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(gate_module.parameters()),
            lr=P.LEARN,
            weight_decay=P.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=P.LEARN, weight_decay=P.weight_decay)
    s_time = datetime.now()
    print('Model Training Started ...', s_time)
    if P.IS_PRETRN and gate_module is None:
        full_embed = _get_full_embed_by_mode()
        train_embed = full_embed[:, spatialSplit_unseen.i_trn]
        val_u_embed = full_embed[:, spatialSplit_unseen.i_val]
        val_a_embed = full_embed[:, spatialSplit_allNod.i_val]
    else:
        train_embed = torch.zeros(32, train_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_u_embed = torch.zeros(32, val_u_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_a_embed = torch.zeros(32, val_a_iter.dataset.tensors[0].shape[2]).to(device).detach()
    print('train_embed', train_embed.shape, train_embed.mean(), train_embed.std())
    print('val_u_embed', val_u_embed.shape, val_u_embed.mean(), val_u_embed.std())
    print('val_a_embed', val_a_embed.shape, val_a_embed.mean(), val_a_embed.std())
    epoch_secs: list[float] = []
    main_train_batch_times: list[float] = []
    for epoch in range(P.EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        if gate_module is not None:
            gate_module.train()
        for x, y in train_iter:
            _t_batch0 = time.perf_counter()
            optimizer.zero_grad()
            if gate_module is not None:
                train_full_embed, train_gate = _fuse_embeddings(
                    temp_full_embed, geo_full_embed,
                    gate_module=gate_module,
                    return_gate=True,
                    print_gate_stats=False
                )
                train_embed = train_full_embed[:, spatialSplit_unseen.i_trn]
            y_pred = model(x.to(device), adj_train, train_embed)
            loss = criterion(y_pred, y.to(device))
            if gate_module is not None:
                # For learned gate, optionally add entropy regularization.
                if isinstance(gate_module, LearnableFusionGate) and P.GATE_REG > 0 and train_gate is not None:
                    g = train_gate[:, spatialSplit_unseen.i_trn]
                    gate_entropy = -(g * torch.log(g + 1e-8) + (1.0 - g) * torch.log(1.0 - g + 1e-8)).mean()
                    # maximize gate entropy to avoid single-branch collapse
                    loss = loss - P.GATE_REG * gate_entropy
                # For temporal_delta fusion, add a small L2 penalty on delta.
                if isinstance(gate_module, TemporalBaseDeltaFusion) and P.DELTA_REG > 0 and train_gate is not None:
                    delta = train_gate[:, spatialSplit_unseen.i_trn]
                    loss = loss + P.DELTA_REG * (delta.pow(2).mean())
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
            main_train_batch_times.append(time.perf_counter() - _t_batch0)
            # print('n', n)
        train_loss = loss_sum / n
        if gate_module is not None:
            gate_module.eval()
            with torch.no_grad():
                full_embed_eval, full_gate_eval = _fuse_embeddings(
                    temp_full_embed, geo_full_embed,
                    gate_module=gate_module,
                    return_gate=True,
                    print_gate_stats=False
                )
            train_embed = full_embed_eval[:, spatialSplit_unseen.i_trn]
            val_u_embed = full_embed_eval[:, spatialSplit_unseen.i_val]
            val_a_embed = full_embed_eval[:, spatialSplit_allNod.i_val]
        val_u_loss = evaluateModel(model, criterion, val_u_iter, adj_val_u, val_u_embed)
        val_a_loss = evaluateModel(model, criterion, val_a_iter, adj_val_a, val_a_embed)
        if val_u_loss < min_val_u_loss:
            min_val_u_loss = val_u_loss
            torch.save(model.state_dict(), P.PATH + '/' + name + '_u.pt')
            if gate_module is not None:
                if isinstance(gate_module, LearnableFusionGate):
                    torch.save(gate_module.state_dict(), P.PATH + '/' + name + '_gate_u.pt')
                else:
                    torch.save(gate_module.state_dict(), P.PATH + '/' + name + '_fusion_u.pt')
        if val_a_loss < min_val_a_loss:
            min_val_a_loss = val_a_loss
            torch.save(model.state_dict(), P.PATH + '/' + name + '_a.pt')
            if gate_module is not None:
                if isinstance(gate_module, LearnableFusionGate):
                    torch.save(gate_module.state_dict(), P.PATH + '/' + name + '_gate_a.pt')
                else:
                    torch.save(gate_module.state_dict(), P.PATH + '/' + name + '_fusion_a.pt')
        endtime = datetime.now()
        epoch_secs.append((endtime - starttime).total_seconds())
        epoch_time = int(epoch_secs[-1])
        print("epoch", epoch,
            "time used:",epoch_time," seconds ",
            "train loss:", train_loss,
            "validation unseen nodes loss:", val_u_loss,
            "validation all nodes loss:", val_a_loss)
        if gate_module is not None:
            if full_gate_eval is not None:
                # temporal_delta 时此处为 delta（几何分支残差），非 sigmoid 门；初值与 gamma 很小会导致长期接近 0。
                aux_name = 'delta(epoch)' if isinstance(gate_module, TemporalBaseDeltaFusion) else 'gate(epoch)'
                print(aux_name + ' mean/std/min/max',
                      full_gate_eval.mean().item(),
                      full_gate_eval.std().item(),
                      full_gate_eval.min().item(),
                      full_gate_eval.max().item())
        with open(P.PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f, %s, %.10f\n" % \
                ("epoch", epoch,
                 "time used:",epoch_time," seconds ",
                 "train loss:", train_loss,
                 "validation unseen nodes loss:", val_u_loss,
                 "validation all nodes loss:", val_a_loss))
    e_time = datetime.now()
    print('MODEL TRAINING DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    PAPER_TIMING['main_train_sec'] = float((e_time - s_time).total_seconds())
    PAPER_TIMING['main_train_epoch_time_mean_sec'] = float(sum(epoch_secs) / len(epoch_secs)) if epoch_secs else 0.0
    PAPER_TIMING["main_train_epoch_sec_list"] = epoch_secs
    _put_iter_stats("main_train_iter", main_train_batch_times)
    PAPER_TIMING['forecast_model_params'] = _trainable_params(model)
    PAPER_TIMING['fusion_module_params'] = _trainable_params(gate_module)
    torch_score = evaluateModel(model, criterion, train_iter, adj_train, train_embed)
    with open(P.PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, %s, %.10e, %.10f\n" % (name, mode, 'MAE on train', torch_score, torch_score))
    print('*' * 40)
    print("%s, %s, %s, %.10e, %.10f" % (name, mode, 'MAE on train', torch_score, torch_score))
    print('min_val_u_loss', min_val_u_loss)
    print('min_val_a_loss', min_val_a_loss)
    print('trainModel Ended ...', time.ctime())

def testModel(name, mode, test_iter, adj_tst, spatialsplit):
    criterion = nn.L1Loss()
    _t_all0 = time.perf_counter()
    print('Model Testing', mode, 'Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', P.TIMESTEP_IN, P.TIMESTEP_OUT)
    model = getModel(name, device)
    model.load_state_dict(torch.load(P.PATH+ '/' + name +mode[-2:]+ '.pt'))
    s_time = datetime.now()
    
    print('Model Infer Start ...', s_time)
    if P.IS_PRETRN:
        if P.PRETRN_MODE == 'dual' and P.FUSION_MODE in ('learned', 'residual', 'residual_gated', 'temporal_delta'):
            temp_full_embed = _get_temporal_full_embed()
            geo_full_embed = _get_geometric_full_embed()
            if P.FUSION_MODE == 'learned':
                gate_module = LearnableFusionGate(
                    embed_dim=temp_full_embed.shape[0],
                    hidden_dim=P.GATE_HIDDEN
                ).to(device)
                gate_ckpt = P.PATH + '/' + name + '_gate' + mode[-2:] + '.pt'
                ckpt_path = gate_ckpt
            elif P.FUSION_MODE in ('residual', 'residual_gated'):
                gate_module = ResidualDeltaFusion(
                    embed_dim=temp_full_embed.shape[0],
                    hidden_dim=P.GATE_HIDDEN,
                    use_gate=(P.FUSION_MODE == 'residual_gated'),
                ).to(device)
                fusion_ckpt = P.PATH + '/' + name + '_fusion' + mode[-2:] + '.pt'
                ckpt_path = fusion_ckpt
            else:
                gate_module = TemporalBaseDeltaFusion(
                    embed_dim=temp_full_embed.shape[0],
                    hidden_dim=P.GATE_HIDDEN,
                    use_projection=True,
                ).to(device)
                fusion_ckpt = P.PATH + '/' + name + '_fusion' + mode[-2:] + '.pt'
                ckpt_path = fusion_ckpt

            if os.path.exists(ckpt_path):
                gate_module.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                print('fusion checkpoint not found, fallback to randomly initialized module:', ckpt_path)
            gate_module.eval()
            with torch.no_grad():
                full_embed = _fuse_embeddings(
                    temp_full_embed, geo_full_embed,
                    gate_module=gate_module,
                    print_gate_stats=False
                )
        else:
            full_embed = _get_full_embed_by_mode()
        tst_embed = full_embed[:, spatialsplit.i_tst]
    else:
        tst_embed = torch.zeros(32, test_iter.dataset.tensors[0].shape[2]).to(device).detach()

    _t_eval0 = time.perf_counter()
    torch_score, eval_batch_times = evaluateModel(
        model, criterion, test_iter, adj_tst, tst_embed, return_batch_times=True
    )
    _t_eval1 = time.perf_counter()
    e_time = datetime.now()
    print('Model Infer End ...', e_time)
    
    print('MODEL INFER DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    YS_pred, predict_batch_times = predictModel(
        model, test_iter, adj_tst, tst_embed, return_batch_times=True
    )
    _put_iter_stats(f"{mode}_eval_batch", eval_batch_times)
    _put_iter_stats(f"{mode}_predict_batch", predict_batch_times)
    _t_pred1 = time.perf_counter()
    YS = test_iter.dataset.tensors[1].cpu().numpy()
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    original_shape = np.squeeze(YS).shape
    YS = scaler.inverse_transform(np.squeeze(YS).reshape(-1, YS.shape[2])).reshape(original_shape)
    YS_pred  = scaler.inverse_transform(np.squeeze(YS_pred).reshape(-1, YS_pred.shape[2])).reshape(original_shape)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(P.PATH + '/' + P.MODELNAME + '_' + mode + '_' + name +'_prediction.npy', YS_pred)
    np.save(P.PATH + '/' + P.MODELNAME + '_' + mode + '_' + name +'_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    f = open(P.PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(P.TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    n_test = len(test_iter.dataset)
    pred_sec = max(_t_pred1 - _t_eval1, 1e-12)
    PAPER_TIMING[f'{mode}_eval_forward_sec'] = float(_t_eval1 - _t_eval0)
    PAPER_TIMING[f'{mode}_predict_forward_sec'] = float(_t_pred1 - _t_eval1)
    PAPER_TIMING[f'{mode}_predict_throughput_samples_per_s'] = float(n_test / pred_sec)
    PAPER_TIMING[f'{mode}_num_test_samples'] = int(n_test)
    PAPER_TIMING[f'{mode}_all_sec'] = float(time.perf_counter() - _t_all0)
    print('Model Testing Ended ...', time.ctime())

################# Parameter Setting #######################
P = type('Parameters', (object,), {})()
P.TIMESTEP_IN = 12
P.TIMESTEP_OUT = 12
P.CHANNEL = 1
P.BATCHSIZE = 64 # 64
P.LEARN = 0.001
P.PRETRN_EPOCH = 10
P.EPOCH = 10 # 100
P.TRAINRATIO = 0.8 # TRAIN + VAL
P.TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
P.ADJTYPE = 'doubletransition'
P.MODELNAME = 'AGCRN'
P.FEATURES = 4
P.SUBGRAPH_SIZE = 64
P.QUOTIENT_GRAPH_RADIUS = 0.01
P.NETWORK_CALLS = 0
P.PRE_LEARN = 0.0001
P.GRAPH_NORM = False
P.HIDDEN = 320
P.AGCRN_RNN_UNITS = 64
P.AGCRN_NUM_LAYERS = 2
P.AGCRN_EMBED_DIM = 10
P.AGCRN_CHEB_K = 2

data = None
data_ds = None
scaler = None
###########################################################
def _load_state_or_fail(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

def _get_temporal_source_data():
    # Temporal encoder uses deseasoned series when available.
    if P.IS_DESEASONED and data_ds is not None:
        return data_ds
    return data

def _get_geometric_full_embed():
    encoder = Geometric_Encoder(P.TEMPERATURE, P.FEATURES, P.GRAPH_NORM, P.HIDDEN).to(device)
    ckpt = os.path.join(P.PRETRAIN_CKPT_DIR, P.GEO_ENCODER_NAME + '.pt')
    encoder = _load_state_or_fail(encoder, ckpt)
    with torch.no_grad():
        fQ1, nQ1 = graph_constructor_helper()
        full_embed = encoder(fQ1.to(device), nQ1.edge_index.to(device)).T.detach()  # [32, N]
    return full_embed

def _get_temporal_full_embed():
    encoder = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
    ckpt = os.path.join(P.PRETRAIN_CKPT_DIR, P.TEMP_ENCODER_NAME + '.pt')
    encoder = _load_state_or_fail(encoder, ckpt)
    source = _get_temporal_source_data()
    with torch.no_grad():
        # source: [T, N] -> encoder input: [N, T]
        full_embed = encoder(torch.tensor(source[:P.trainval_size, :]).float().to(device).T).T.detach()
    return full_embed

class LearnableFusionGate(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embed_temporal, embed_geometric):
        # input: [D, N], output logits: [1, N]
        z_t = embed_temporal.T
        z_g = embed_geometric.T
        z = torch.cat([z_t, z_g, z_t - z_g, z_t * z_g], dim=1)
        return self.net(z).T

class ResidualDeltaFusion(nn.Module):
    """
    Residual/Delta fusion:
      z = z1 + g(z1, z2)
    Optional gated variant:
      z = z1 + alpha(z1, z2) ⊙ g(z1, z2),   alpha ∈ [0,1] (node-wise scalar, broadcast over D)

    Inputs/outputs are [D, N].
    """
    def __init__(self, embed_dim=32, hidden_dim=64, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        self.delta_net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z1, z2):
        # z1/z2: [D, N] -> features: [N, 4D]
        z1t = z1.T
        z2t = z2.T
        feat = torch.cat([z1t, z2t, z1t - z2t, z1t * z2t], dim=1)
        delta = self.delta_net(feat).T  # [D, N]
        if not self.use_gate:
            return z1 + delta, None
        gate_logits = self.gate_net(feat).T  # [1, N]
        return z1 + delta, gate_logits

class TemporalBaseDeltaFusion(nn.Module):
    """
    Use temporal embedding as base (best branch in your experiments),
    add a learnable delta from geometric embedding:
        fused = z_temporal + gamma * delta(z_temporal, z_geometric)
    with both gamma and delta head initialized to 0, so the initial fused
    embedding equals temporal-only.

    Inputs/outputs are [D, N].
    """
    def __init__(self, embed_dim=32, hidden_dim=64, use_projection=True):
        super().__init__()
        self.use_projection = use_projection
        if use_projection:
            self.proj_t = nn.Linear(embed_dim, embed_dim, bias=False)
            self.proj_g = nn.Linear(embed_dim, embed_dim, bias=False)
        else:
            self.proj_t = None
            self.proj_g = None

        self.delta_net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Ensure delta starts from 0 -> fused starts from temporal.
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

        # Node-wise gamma in [0,1], estimated from the same features.
        self.gamma_net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Bias negative so that initial sigmoid(gamma_logits) is close to 0.
        nn.init.constant_(self.gamma_net[-1].bias, -4.0)

    def forward(self, z_t, z_g):
        # z_t/z_g: [D, N] -> [N, D]
        ztt = z_t.T
        zgg = z_g.T
        if self.use_projection:
            ztt = self.proj_t(ztt)
            zgg = self.proj_g(zgg)
        feat = torch.cat([ztt, zgg, ztt - zgg, ztt * zgg], dim=1)  # [N, 4D]
        delta = self.delta_net(feat).T  # [D, N]
        gamma_logits = self.gamma_net(feat).T  # [1, N]
        gamma = torch.sigmoid(gamma_logits)
        fused = z_t + gamma * delta
        return fused, delta

def _normalize_embeddings(embed_temporal, embed_geometric):
    if not P.FUSION_NORM:
        return embed_temporal, embed_geometric
    d = embed_temporal.shape[0]
    z_t = F.layer_norm(embed_temporal.T, (d,)).T
    z_g = F.layer_norm(embed_geometric.T, (d,)).T
    return z_t, z_g

def _fuse_embeddings(embed_temporal, embed_geometric, gate_module=None, return_gate=False, print_gate_stats=True):
    embed_temporal, embed_geometric = _normalize_embeddings(embed_temporal, embed_geometric)
    if P.FUSION_MODE == 'weighted':
        alpha = P.FUSE_ALPHA
        fused = alpha * embed_temporal + (1.0 - alpha) * embed_geometric
        return (fused, None) if return_gate else fused
    if P.FUSION_MODE == 'gated':
        # Node-wise static gate from embedding energy.
        # gate in [0,1], shape [1, N], then broadcast over feature dimension.
        e_t = embed_temporal.abs().mean(dim=0, keepdim=True)
        e_g = embed_geometric.abs().mean(dim=0, keepdim=True)
        gate = torch.sigmoid(P.GATE_SCALE * (e_t - e_g) + P.GATE_BIAS)
        if print_gate_stats:
            print('gate stats', gate.mean().item(), gate.std().item(), gate.min().item(), gate.max().item())
        fused = gate * embed_temporal + (1.0 - gate) * embed_geometric
        return (fused, gate) if return_gate else fused
    if P.FUSION_MODE == 'learned':
        if gate_module is None:
            raise ValueError('learned fusion requires gate_module')
        gate_logits = gate_module(embed_temporal, embed_geometric)
        gate = torch.sigmoid(P.GATE_SCALE * gate_logits + P.GATE_BIAS)
        if print_gate_stats:
            print('gate stats', gate.mean().item(), gate.std().item(), gate.min().item(), gate.max().item())
        fused = gate * embed_temporal + (1.0 - gate) * embed_geometric
        return (fused, gate) if return_gate else fused
    if P.FUSION_MODE == 'temporal_delta':
        if gate_module is None:
            raise ValueError('temporal_delta fusion requires gate_module')
        fused, delta = gate_module(embed_temporal, embed_geometric)
        return (fused, delta) if return_gate else fused
    if P.FUSION_MODE in ('residual', 'residual_gated'):
        if gate_module is None:
            raise ValueError('residual fusion requires gate_module')
        # z1 is temporal (stronger main branch), z2 is geometric (complement).
        fused_base, gate_logits = gate_module(embed_temporal, embed_geometric)
        if P.FUSION_MODE == 'residual':
            # gate_module may return logits; ignore and behave as ungated residual.
            fused = fused_base
            return (fused, None) if return_gate else fused
        # gated residual: alpha is node-wise scalar in [0,1]
        if gate_logits is None:
            raise ValueError('residual_gated requires gate logits')
        gate = torch.sigmoid(P.GATE_SCALE * gate_logits + P.GATE_BIAS)  # [1, N]
        # fused_base currently equals z1 + delta (see module); we need to reapply gate on delta.
        # Recompute delta by subtracting z1; keep stable residual form.
        delta = fused_base - embed_temporal
        fused = embed_temporal + gate * delta
        if print_gate_stats:
            print('gate stats', gate.mean().item(), gate.std().item(), gate.min().item(), gate.max().item())
        return (fused, gate) if return_gate else fused
    raise ValueError(f'Unknown FUSION_MODE: {P.FUSION_MODE}')

def _get_full_embed_by_mode(gate_module=None, return_gate=False, print_gate_stats=True):
    mode = P.PRETRN_MODE
    if mode == 'geo':
        embed = _get_geometric_full_embed()
        return (embed, None) if return_gate else embed
    if mode == 'temporal':
        embed = _get_temporal_full_embed()
        return (embed, None) if return_gate else embed
    if mode == 'dual':
        temp_full_embed = _get_temporal_full_embed()
        geo_full_embed = _get_geometric_full_embed()
        return _fuse_embeddings(
            temp_full_embed, geo_full_embed,
            gate_module=gate_module,
            return_gate=return_gate,
            print_gate_stats=print_gate_stats,
        )
    raise ValueError(f'Unknown PRETRN_MODE: {mode}')

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
    P.weight_decay = float(sys.argv[9]) if len(sys.argv) >= 10 else 0.01
    P.adp_adj = bool(int(sys.argv[10])) if len(sys.argv) >= 11 else True
    P.is_SGA = bool(int(sys.argv[11])) if len(sys.argv) >= 12 else True
    P.FEATURES = int(sys.argv[12]) if len(sys.argv) >= 13 else 2
    P.SUBGRAPH_SIZE = int(sys.argv[13]) if len(sys.argv) >= 14 else 64
    P.QUOTIENT_GRAPH_RADIUS = float(sys.argv[14]) if len(sys.argv) >= 15 else 0.01
    P.PRETRN_EPOCH = int(sys.argv[15]) if len(sys.argv) >= 16 else 100
    P.EPOCH = int(sys.argv[16]) if len(sys.argv) >= 17 else 100
    P.NETWORK_CALLS = bool(int(sys.argv[17])) if len(sys.argv) >= 18 else 0
    P.PRE_LEARN = float(sys.argv[18]) if len(sys.argv) >= 19 else P.LEARN
    P.GRAPH_NORM = bool(int(sys.argv[19])) if len(sys.argv) >= 20 else True
    P.HIDDEN = int(sys.argv[20]) if len(sys.argv) >= 21 else 320
    P.IS_DUAL_PRETRN = bool(int(sys.argv[21])) if len(sys.argv) >= 22 else False
    P.FUSE_ALPHA = float(sys.argv[22]) if len(sys.argv) >= 23 else 0.5
    P.SKIP_PRETRAIN = bool(int(sys.argv[23])) if len(sys.argv) >= 24 else False
    P.TEMP_ENCODER_NAME = sys.argv[24] if len(sys.argv) >= 25 else 'encoder'
    P.GEO_ENCODER_NAME = sys.argv[25] if len(sys.argv) >= 26 else ('encoderg' if P.IS_DUAL_PRETRN else 'encoder')
    P.PRETRAIN_CKPT_DIR = sys.argv[26] if len(sys.argv) >= 27 else ''
    P.PRETRN_MODE = sys.argv[27] if len(sys.argv) >= 28 else 'temporal'#'dual'#'temporal'#('dual' if P.IS_DUAL_PRETRN else 'geo')
    # dual pretraining trains two different encoders (temporal + geometric).
    # If their ckpt filenames are identical, the second one overwrites the first,
    # and later loading will fail with missing/unexpected keys.
    if P.PRETRN_MODE == 'dual' and P.GEO_ENCODER_NAME == P.TEMP_ENCODER_NAME:
        P.GEO_ENCODER_NAME = P.TEMP_ENCODER_NAME + '_geo'
    P.FUSION_MODE = sys.argv[28] if len(sys.argv) >= 29 else 'temporal_delta'
    P.GATE_SCALE = float(sys.argv[29]) if len(sys.argv) >= 30 else 1.0
    P.GATE_BIAS = float(sys.argv[30]) if len(sys.argv) >= 31 else 0.0
    P.GATE_HIDDEN = int(sys.argv[31]) if len(sys.argv) >= 32 else 64
    P.FUSION_NORM = bool(int(sys.argv[32])) if len(sys.argv) >= 33 else True
    P.GATE_REG = float(sys.argv[33]) if len(sys.argv) >= 34 else 0.0
    P.DELTA_REG = float(sys.argv[34]) if len(sys.argv) >= 35 else 0.0

device = torch.device('cuda:0') 
###########################################################
def main():
    script_start_time = datetime.now()
    reset_paper_timing()

    get_argv()

    # DATASET
    P.KEYWORD = 'pred_' + P.DATANAME + '_' + P.MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M") + '_' + str(os.getpid())
    P.PATH = '../save/' + P.KEYWORD
    if not P.PRETRAIN_CKPT_DIR:
        P.PRETRAIN_CKPT_DIR = P.PATH
    global data
    global data_ds
    global scaler
    n_dct_coeff = None
    if P.DATANAME == 'METRLA':
        print('P.DATANAME == METRLA')
        P.FLOWPATH = '../METRLA/metr-la.h5'
        P.n_dct_coeff = 3918
        P.ADJPATH = '../METRLA/adj_mx.pkl'
        P.N_NODE = 207
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMSBAY':
        print('P.DATANAME == PEMSBAY')
        P.FLOWPATH = '../PEMSBAY/pems-bay.h5'
        P.n_dct_coeff = 4107
        P.ADJPATH = '../PEMSBAY/adj_mx_bay.pkl'
        P.N_NODE = 325
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMSD7M':
        print('P.DATANAME == PEMSD7M')
        P.FLOWPATH = '../PEMSD7M/V_228.csv'
        P.n_dct_coeff = 860
        P.ADJPATH = '../PEMSD7M/W_228.csv'
        P.N_NODE = 228
        data = pd.read_csv(P.FLOWPATH,index_col=[0]).values
    elif P.DATANAME == 'PEMS11160':
        print('P.DATANAME == PEMS11160')
        P.BATCHSIZE = 16
        P.EPOCH = 20
        P.FLOWPATH = '../PEMS11160/pems12kSPEED2m.npy'
        P.n_dct_coeff = 2179
        P.ADJPATH = '../PEMS11160/adj_mat.pkl'
        P.N_NODE = 11160
        with open(P.FLOWPATH, 'rb') as f:
            data = np.load(f)
    else:
        print('NO DATA LOADED')

    if P.NETWORK_CALLS:
        network_calls()
        return

    # de-season
    if P.IS_DESEASONED:
        P.CHANNEL = 2
        data_ = dct(data, axis=0)
        data_[n_dct_coeff:, :] = 0
        data_ds = data - idct(data_, axis=0) # the seasonal data

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # de-season scaler
    if P.IS_DESEASONED:
        scaler = StandardScaler()
        data_ds = scaler.fit_transform(data_ds)

    print('data.shape', data.shape)

    pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
    train_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
    adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a = setups()

    P.AGCRN_NUM_NODES_MAX = int(max(
        train_iter.dataset.tensors[0].shape[2],
        val_u_iter.dataset.tensors[0].shape[2],
        val_a_iter.dataset.tensors[0].shape[2],
        tst_u_iter.dataset.tensors[0].shape[2],
        tst_a_iter.dataset.tensors[0].shape[2],
    ))
    print('AGCRN_NUM_NODES_MAX (pad/crop to match GWN-style variable subgraphs)', P.AGCRN_NUM_NODES_MAX)

    if P.IS_PRETRN:
        if P.SKIP_PRETRAIN:
            print(P.KEYWORD, 'skip pretraining stage; load existing checkpoints')
        else:
            print(P.KEYWORD, 'pretraining started', time.ctime())
            if P.PRETRN_MODE == 'geo':
                pretrainModel('encoder', 'pretrain', pretrn_iter, preval_iter)
            elif P.PRETRN_MODE == 'temporal':
                pretrainModel_temporal(P.TEMP_ENCODER_NAME, spatialSplit_unseen)
            elif P.PRETRN_MODE == 'dual':
                # In dual mode we train both temporal and geometric branches from scratch.
                pretrainModel_temporal(P.TEMP_ENCODER_NAME, spatialSplit_unseen)
                pretrainModel(P.GEO_ENCODER_NAME, 'pretrain', pretrn_iter, preval_iter)
            else:
                raise ValueError(f'Unknown PRETRN_MODE: {P.PRETRN_MODE}')
    else:
        print(P.KEYWORD, 'No pre-training')

    print(P.KEYWORD, 'training started', time.ctime())
    trainModel(P.MODELNAME, 'train',
        train_iter, val_u_iter, val_a_iter,
        adj_train, adj_val_u, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod)
    
    print(P.KEYWORD, 'testing started', time.ctime())
    testModel(P.MODELNAME, 'test_u', tst_u_iter, adj_tst_u, spatialSplit_unseen)
    testModel(P.MODELNAME, 'test_a', tst_a_iter, adj_tst_a, spatialSplit_allNod)
    print('SCRIPT DURATION', datetime.now()-script_start_time)
    emit_paper_timing_summary(script_start_time)

if __name__ == '__main__':
    main()

