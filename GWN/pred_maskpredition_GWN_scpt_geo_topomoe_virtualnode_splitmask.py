"""TopoMoE estimation with split virtual-node test masks.

Policy:
  - TRAIN/VAL: original point-wise random masks only.
  - TEST tst_u: unseen test nodes excluding fixed virtual nodes, random point masks.
  - TEST tst_v: fixed virtual test nodes only, 100% masked at all timesteps.
  - TEST tst_a: all-node test split, random point masks.
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch

import pred_maskpredition_GWN_scpt_geo as est
import unseen_nodes


def _parse_int_list(raw: str) -> list[int]:
    if not raw or not raw.strip():
        return []
    return [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]


def _read_policy_env() -> None:
    P = est.P
    P.FIXED_MASK_TEST_NODES = _parse_int_list(os.environ.get("FIXED_MASK_TEST_NODES", ""))
    P.FIXED_MASK_NODES = _parse_int_list(os.environ.get("FIXED_MASK_NODES", ""))
    P.VIRTUAL_NODES = _parse_int_list(os.environ.get("VIRTUAL_NODES", getattr(P, "VIRTUAL_NODES", "") or ""))
    P.FIXED_MASK_FRAC = float(os.environ.get("FIXED_MASK_FRAC", "0.05"))
    raw_n = os.environ.get("FIXED_MASK_N", "").strip()
    P.FIXED_MASK_N = int(raw_n) if raw_n else None
    P.MASK_POLICY = "train_random_test_split_virtual"


def _save_policy(path_dir: str, policy: dict) -> None:
    os.makedirs(path_dir, exist_ok=True)
    out = os.path.join(path_dir, "mask_policy.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)
    print(f"[splitmask] saved {out}")


def _init_virtual_test_nodes(spatial_split_unseen, n_nodes: int) -> dict:
    P = est.P
    _read_policy_env()
    rng = np.random.default_rng(int(P.seed_SS if P.seed_SS != -1 else P.seed))
    pool_trn = [int(i) for i in spatial_split_unseen.i_trn]
    pool_tst = [int(i) for i in spatial_split_unseen.i_tst]
    explicit = set(P.FIXED_MASK_TEST_NODES) | set(P.FIXED_MASK_NODES) | set(P.VIRTUAL_NODES)
    fixed_test = sorted([x for x in explicit if x in set(pool_tst)])
    if not fixed_test:
        n_sim = P.FIXED_MASK_N
        if n_sim is None:
            # Keep the same default scale as the previous fixedmask runs.
            n_sim = max(1, int(len(pool_trn) * float(P.FIXED_MASK_FRAC)))
        fixed_test = rng.choice(pool_tst, size=min(n_sim, len(pool_tst)), replace=False).astype(int).tolist()
    fixed_test = sorted({i for i in fixed_test if 0 <= i < n_nodes})
    test_u = [i for i in pool_tst if i not in set(fixed_test)]
    P.FIXED_MASK_NODES_TEST = fixed_test
    P.TST_U_NODES_NO_V = test_u
    P.TST_V_NODES = fixed_test
    policy = {
        "policy": P.MASK_POLICY,
        "seed": int(P.seed),
        "seed_SS": int(P.seed_SS),
        "n_nodes": int(n_nodes),
        "fixed_mask_frac": float(P.FIXED_MASK_FRAC),
        "fixed_mask_n": P.FIXED_MASK_N,
        "miss_ratio_train": float(getattr(P, "MISS_RATIO", 0.2)),
        "train_pool_size": len(pool_trn),
        "test_pool_size": len(pool_tst),
        "tst_u_nodes_no_v": test_u,
        "fixed_test_nodes": fixed_test,
        "note": "train random point masks only; tst_u excludes fixed_test_nodes; tst_v is fixed_test_nodes with all timesteps masked",
    }
    P.MASK_POLICY_JSON = policy
    return policy


def _random_point_mask(x: np.ndarray, m: np.ndarray, ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mask = rng.random(x.shape) < ratio
    x[mask] = 0.0
    m[mask] = 0.0
    return x, m


def _to_arrays(XS: list[np.ndarray], YS: list[np.ndarray], MS: list[np.ndarray]):
    XS_arr, YS_arr, MS_arr = np.array(XS), np.array(YS), np.array(MS)
    XS_arr = XS_arr[:, :, :, np.newaxis].transpose(0, 3, 2, 1)
    YS_arr = YS_arr[:, :, :, np.newaxis]
    MS_arr = MS_arr[:, :, :, np.newaxis].transpose(0, 3, 2, 1)
    return XS_arr, YS_arr, MS_arr


def getXSYS_estimation(data, mode, missing_ratio=0.2, missing_ratio_test=0.2, epoch: Optional[int] = None):
    P = est.P
    train_num = int(data.shape[0] * P.TRAINRATIO)
    XS, YS, MS = [], [], []
    if mode == "TRAIN":
        rng = np.random.default_rng(int(P.seed) + (int(epoch) if epoch is not None else 0))
        row_range = range(train_num)
        ratio = missing_ratio
    elif mode == "TEST":
        rng = np.random.default_rng(int(P.seed_SS if P.seed_SS != -1 else P.seed) + 99991)
        row_range = range(train_num, data.shape[0])
        ratio = missing_ratio_test
    elif mode == "TEST_VIRTUAL":
        rng = np.random.default_rng(0)
        row_range = range(train_num, data.shape[0])
        ratio = 0.0
    elif mode == "TEST_VFULL":
        # Full-graph tst_v: keep all nodes, but virtual test nodes are 100% masked at all timesteps.
        rng = np.random.default_rng(0)
        row_range = range(train_num, data.shape[0])
        ratio = 0.0
    else:
        raise ValueError(f"Unknown mode {mode}")

    fixed = set(getattr(P, "TST_V_NODES", []) or []) if mode in ("TEST_VIRTUAL", "TEST_VFULL") else set()
    for i in row_range:
        x = data[i : i + 1, :].copy()
        y = data[i : i + 1, :].copy()
        m = np.ones_like(x)
        if mode in ("TEST_VIRTUAL", "TEST_VFULL"):
            for node in fixed:
                if 0 <= node < x.shape[1]:
                    x[:, node] = 0.0
                    m[:, node] = 0.0
        else:
            x, m = _random_point_mask(x, m, ratio, rng)
        XS.append(x)
        YS.append(y)
        MS.append(m)
    return _to_arrays(XS, YS, MS)


def setups_estimation(missing_ratio=0.2):
    P = est.P
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
    P.MISS_RATIO = missing_ratio
    print(P.KEYWORD, "data splits (train random, split virtual-node tests)", time.ctime())

    data = est.data
    data_ds = est.data_ds
    spatialSplit_unseen = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=0.15, r_tst=0.15, seed=P.seed_SS)
    spatialSplit_allNod = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=min(1.0, P.R_TRN * 8 / 7), r_tst=1.0, seed=P.seed_SS)
    policy = _init_virtual_test_nodes(spatialSplit_unseen, data.shape[1])
    _save_policy(P.PATH, policy)
    print("[splitmask] fixed_test_nodes", P.TST_V_NODES)
    print("[splitmask] tst_u_nodes_no_v", P.TST_U_NODES_NO_V)

    trainXS, trainYS, trainMS = getXSYS_estimation(data, "TRAIN", missing_ratio=missing_ratio)
    testXS, testYS, testMS = getXSYS_estimation(data, "TEST", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
    testVXS, testVYS, testVMS = getXSYS_estimation(data, "TEST_VIRTUAL", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
    testVFullXS, testVFullYS, testVFullMS = getXSYS_estimation(data, "TEST_VFULL", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
    if P.IS_DESEASONED:
        trainXS_ds, _, trainMS_ds = getXSYS_estimation(data_ds, "TRAIN", missing_ratio=missing_ratio)
        testXS_ds, _, testMS_ds = getXSYS_estimation(data_ds, "TEST", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
        testVXS_ds, _, testVMS_ds = getXSYS_estimation(data_ds, "TEST_VIRTUAL", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
        testVFullXS_ds, _, testVFullMS_ds = getXSYS_estimation(data_ds, "TEST_VFULL", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
        trainXS = np.concatenate((trainXS, trainXS_ds), axis=1)
        testXS = np.concatenate((testXS, testXS_ds), axis=1)
        testVXS = np.concatenate((testVXS, testVXS_ds), axis=1)
        testVFullXS = np.concatenate((testVFullXS, testVFullXS_ds), axis=1)
        trainMS = np.concatenate((trainMS, trainMS_ds), axis=1)
        testMS = np.concatenate((testMS, testMS_ds), axis=1)
        testVMS = np.concatenate((testVMS, testVMS_ds), axis=1)
        testVFullMS = np.concatenate((testVFullMS, testVFullMS_ds), axis=1)

    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (1 - P.TRAINVALSPLIT))
    XS_torch_trn = trainXS[: P.train_size]
    YS_torch_trn = trainYS[: P.train_size]
    MS_torch_trn = trainMS[: P.train_size]
    XS_torch_val = trainXS[P.train_size :]
    YS_torch_val = trainYS[P.train_size :]
    MS_torch_val = trainMS[P.train_size :]

    tst_u_nodes = np.array(P.TST_U_NODES_NO_V, dtype=int)
    tst_v_nodes = np.array(P.TST_V_NODES, dtype=int)
    all_nodes = np.arange(data.shape[1], dtype=int)

    train_data = torch.utils.data.TensorDataset(
        torch.Tensor(XS_torch_trn[:, :, spatialSplit_unseen.i_trn, :]),
        torch.Tensor(YS_torch_trn[:, :, spatialSplit_unseen.i_trn, :]),
        torch.Tensor(MS_torch_trn[:, :, spatialSplit_unseen.i_trn, :]),
    )
    val_u_data = torch.utils.data.TensorDataset(
        torch.Tensor(XS_torch_val[:, :, spatialSplit_unseen.i_val, :]),
        torch.Tensor(YS_torch_val[:, :, spatialSplit_unseen.i_val, :]),
        torch.Tensor(MS_torch_val[:, :, spatialSplit_unseen.i_val, :]),
    )
    val_a_data = torch.utils.data.TensorDataset(
        torch.Tensor(XS_torch_val[:, :, spatialSplit_allNod.i_val, :]),
        torch.Tensor(YS_torch_val[:, :, spatialSplit_allNod.i_val, :]),
        torch.Tensor(MS_torch_val[:, :, spatialSplit_allNod.i_val, :]),
    )
    tst_u_data = torch.utils.data.TensorDataset(
        torch.Tensor(testXS[:, :, tst_u_nodes, :]),
        torch.Tensor(testYS[:, :, tst_u_nodes, :]),
        torch.Tensor(testMS[:, :, tst_u_nodes, :]),
    )
    tst_v_data = torch.utils.data.TensorDataset(
        torch.Tensor(testVXS[:, :, tst_v_nodes, :]),
        torch.Tensor(testVYS[:, :, tst_v_nodes, :]),
        torch.Tensor(testVMS[:, :, tst_v_nodes, :]),
    )
    # Full-graph tst_v: forward on all nodes, but only V nodes are marked missing (m=0) for metric.
    tst_v_full_data = torch.utils.data.TensorDataset(
        torch.Tensor(testVFullXS[:, :, all_nodes, :]),
        torch.Tensor(testVFullYS[:, :, all_nodes, :]),
        torch.Tensor(testVFullMS[:, :, all_nodes, :]),
    )
    tst_a_data = torch.utils.data.TensorDataset(
        torch.Tensor(testXS[:, :, spatialSplit_allNod.i_tst, :]),
        torch.Tensor(testYS[:, :, spatialSplit_allNod.i_tst, :]),
        torch.Tensor(testMS[:, :, spatialSplit_allNod.i_tst, :]),
    )

    pin_memory = True if est.device.type == "cuda" else False
    train_iter = torch.utils.data.DataLoader(train_data, P.BATCHSIZE, shuffle=True, num_workers=8, pin_memory=pin_memory)
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.BATCHSIZE, shuffle=False)
    val_a_iter = torch.utils.data.DataLoader(val_a_data, P.BATCHSIZE, shuffle=False)
    tst_u_iter = torch.utils.data.DataLoader(tst_u_data, P.BATCHSIZE, shuffle=False)
    tst_v_iter = torch.utils.data.DataLoader(tst_v_data, P.BATCHSIZE, shuffle=False)
    tst_v_full_iter = torch.utils.data.DataLoader(tst_v_full_data, P.BATCHSIZE, shuffle=False)
    tst_a_iter = torch.utils.data.DataLoader(tst_a_data, P.BATCHSIZE, shuffle=False)

    adj_mx = est.load_adj(P.ADJPATH, P.ADJTYPE, P.DATANAME)
    adj_train = [torch.tensor(i[spatialSplit_unseen.i_trn, :][:, spatialSplit_unseen.i_trn]).to(est.device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[spatialSplit_unseen.i_val, :][:, spatialSplit_unseen.i_val]).to(est.device) for i in adj_mx]
    adj_val_a = [torch.tensor(i[spatialSplit_allNod.i_val, :][:, spatialSplit_allNod.i_val]).to(est.device) for i in adj_mx]
    adj_tst_u = [torch.tensor(i[tst_u_nodes, :][:, tst_u_nodes]).to(est.device) for i in adj_mx]
    adj_tst_v = [torch.tensor(i[tst_v_nodes, :][:, tst_v_nodes]).to(est.device) for i in adj_mx]
    adj_tst_a = [torch.tensor(i[spatialSplit_allNod.i_tst, :][:, spatialSplit_allNod.i_tst]).to(est.device) for i in adj_mx]
    adj_tst_v_full = [torch.tensor(i[all_nodes, :][:, all_nodes]).to(est.device) for i in adj_mx]

    pretrn_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(XS_torch_trn[:, -1, spatialSplit_unseen.i_trn, 0]).T),
        P.BATCHSIZE,
        shuffle=True,
    )
    preval_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(trainYS[:, -1, spatialSplit_unseen.i_val, 0]).T.float()),
        P.BATCHSIZE,
        shuffle=False,
    )
    pretrn_iterg = random.sample(list(spatialSplit_unseen.i_trn), P.BATCHSIZE)
    preval_iterg = list(spatialSplit_unseen.i_val)

    P.TST_V_ITER = tst_v_iter
    P.TST_V_ADJ = adj_tst_v
    P.TST_V_MAPPING = {old: new for new, old in enumerate(tst_v_nodes)}
    P.TST_V_NODES = tst_v_nodes
    # For eval-only full-graph tst_v.
    P.TST_V_FULL_ITER = tst_v_full_iter
    P.TST_V_FULL_ADJ = adj_tst_v_full
    P.TST_V_FULL_NODES = all_nodes
    P.TST_V_FULL_MAPPING = {int(i): int(i) for i in all_nodes}
    P.TST_V_FULL_SPATIALSPLIT = spatialSplit_allNod
    spatialSplit_unseen.i_tst = tst_u_nodes
    mapping_tst_u = {old: new for new, old in enumerate(tst_u_nodes)}

    for k, v in vars(P).items():
        print(k, v)

    return (
        pretrn_iter,
        preval_iter,
        spatialSplit_unseen,
        spatialSplit_allNod,
        train_iter,
        val_u_iter,
        val_a_iter,
        tst_u_iter,
        tst_a_iter,
        adj_train,
        adj_val_u,
        adj_val_a,
        adj_tst_u,
        adj_tst_a,
        mapping_tst_u,
        pretrn_iterg,
        preval_iterg,
    )


import pred_maskpredition_GWN_scpt_geo_topomoe as topomoe  # noqa: E402


def testModel_estimation_with_split_virtual(name, mode, test_iter, node_indices, scaler, adj_tst_u, mapping, spatialsplit):
    topomoe.testModel_estimation_with_pretrain_topomoe(
        name, mode, test_iter, node_indices, scaler, adj_tst_u, mapping, spatialsplit
    )
    if mode == "tst_u":
        P = est.P
        topomoe.testModel_estimation_with_pretrain_topomoe(
            name,
            "tst_v",
            P.TST_V_ITER,
            P.TST_V_NODES,
            scaler,
            P.TST_V_ADJ,
            P.TST_V_MAPPING,
            spatialsplit,
        )


est.getXSYS_estimation = getXSYS_estimation
est.setups_estimation = setups_estimation
est.get_argv = topomoe.get_argv_topomoe_estimation
est.trainModel_estimation_with_pretrain = topomoe.trainModel_estimation_with_pretrain_topomoe
est.testModel_estimation_with_pretrain = testModel_estimation_with_split_virtual


if __name__ == "__main__":
    est.main()
