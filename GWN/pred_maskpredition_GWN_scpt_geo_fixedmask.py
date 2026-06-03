"""
Estimation with fixed full-node masks (virtual-node simulation) + optional point-wise random masks.

Compared to pred_maskpredition_GWN_scpt_geo.py:
  - TRAIN: each epoch re-randomizes point masks on non-fixed nodes; fixed nodes stay 100% masked.
  - VAL/TEST: masks fixed at setup time (reproducible).

Environment variables (optional):
  FIXED_MASK_NODES       comma-separated global node ids, fully masked everywhere
  FIXED_MASK_FRAC        fraction of train/tst pools to sample as virtual-like (default 0.05)
  FIXED_MASK_N           override count per pool (train and tst each)
  FIXED_MASK_TRAIN_NODES comma-separated, train pool only
  FIXED_MASK_TEST_NODES  comma-separated, test pool only
  MASK_RANDOM_ON_OTHER   1/0, point-wise random on non-fixed nodes (default 1)
  VIRTUAL_NODES          alias merged into fixed set (same as sensor2 virtual nodes)
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import Iterable, Optional, Set

import numpy as np
import torch

import pred_maskpredition_GWN_scpt_geo as geo
import unseen_nodes

# Re-export for topomoe / scripts
P = geo.P
device = geo.device
getModel = geo.getModel
masked_loss = geo.masked_loss
# ... main entry uses geo.main after patching


def _parse_int_list(raw: str) -> list[int]:
    if not raw or not str(raw).strip():
        return []
    out: list[int] = []
    for p in str(raw).replace(";", ",").split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _read_mask_config_from_env() -> None:
    P = geo.P
    P.FIXED_MASK_NODES = _parse_int_list(os.environ.get("FIXED_MASK_NODES", ""))
    P.FIXED_MASK_TRAIN_NODES = _parse_int_list(os.environ.get("FIXED_MASK_TRAIN_NODES", ""))
    P.FIXED_MASK_TEST_NODES = _parse_int_list(os.environ.get("FIXED_MASK_TEST_NODES", ""))
    P.VIRTUAL_NODES = _parse_int_list(
        os.environ.get("VIRTUAL_NODES", getattr(P, "VIRTUAL_NODES", "") or "")
    )
    P.FIXED_MASK_FRAC = float(os.environ.get("FIXED_MASK_FRAC", "0.05"))
    n_override = os.environ.get("FIXED_MASK_N", "").strip()
    P.FIXED_MASK_N = int(n_override) if n_override else None
    P.MASK_RANDOM_ON_OTHER = os.environ.get("MASK_RANDOM_ON_OTHER", "1").strip() != "0"
    P.MASK_POLICY = "fixed_virtual_sim"


def init_fixed_mask_policy(spatial_split_unseen, n_nodes: int) -> dict:
    """Pick fixed full-mask nodes once (reproducible)."""
    P = geo.P
    _read_mask_config_from_env()

    rng = np.random.default_rng(int(P.seed_SS if P.seed_SS != -1 else P.seed))
    pool_trn = [int(i) for i in spatial_split_unseen.i_trn]
    pool_tst = [int(i) for i in spatial_split_unseen.i_tst]

    n_sim = P.FIXED_MASK_N
    if n_sim is None:
        n_sim = max(1, int(len(pool_trn) * float(P.FIXED_MASK_FRAC)))

    explicit_all = set(P.FIXED_MASK_NODES) | set(P.VIRTUAL_NODES)
    fixed_train = sorted(set(P.FIXED_MASK_TRAIN_NODES) | (explicit_all & set(pool_trn)))
    fixed_test = sorted(set(P.FIXED_MASK_TEST_NODES) | (explicit_all & set(pool_tst)))

    if not fixed_train and not explicit_all:
        k = min(n_sim, len(pool_trn))
        if k > 0:
            fixed_train = rng.choice(pool_trn, size=k, replace=False).astype(int).tolist()
    if not fixed_test and not explicit_all:
        k = min(n_sim, len(pool_tst))
        if k > 0:
            fixed_test = rng.choice(pool_tst, size=k, replace=False).astype(int).tolist()

    # explicit global ids apply to whichever pool they belong to
    for nid in explicit_all:
        if nid in pool_trn and nid not in fixed_train:
            fixed_train.append(nid)
        if nid in pool_tst and nid not in fixed_test:
            fixed_test.append(nid)

    fixed_train = sorted({i for i in fixed_train if 0 <= i < n_nodes})
    fixed_test = sorted({i for i in fixed_test if 0 <= i < n_nodes})
    fixed_all = sorted(set(fixed_train) | set(fixed_test))

    P.FIXED_MASK_NODES_TRAIN = fixed_train
    P.FIXED_MASK_NODES_TEST = fixed_test
    P.FIXED_MASK_NODES_ALL = fixed_all

    policy = {
        "policy": P.MASK_POLICY,
        "seed": int(P.seed),
        "seed_SS": int(P.seed_SS),
        "n_nodes": int(n_nodes),
        "fixed_mask_frac": float(P.FIXED_MASK_FRAC),
        "fixed_mask_n": P.FIXED_MASK_N,
        "mask_random_on_other": bool(P.MASK_RANDOM_ON_OTHER),
        "miss_ratio_train": float(getattr(P, "MISS_RATIO", 0.2)),
        "fixed_train_nodes": fixed_train,
        "fixed_test_nodes": fixed_test,
        "fixed_all_nodes": fixed_all,
        "train_pool_size": len(pool_trn),
        "test_pool_size": len(pool_tst),
        "note": (
            "fixed nodes: all timesteps x=0,m=0 (simulate no history); "
            "other nodes: random point mask per TRAIN epoch"
        ),
    }
    P.MASK_POLICY_JSON = policy
    return policy


def save_mask_policy(path_dir: str) -> str:
    os.makedirs(path_dir, exist_ok=True)
    out = os.path.join(path_dir, "mask_policy.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(getattr(geo.P, "MASK_POLICY_JSON", {}), f, indent=2, ensure_ascii=False)
    print(f"[fixedmask] saved {out}")
    return out


def _fixed_set_for_mode(mode: str) -> Set[int]:
    P = geo.P
    s: Set[int] = set(getattr(P, "FIXED_MASK_NODES_ALL", []) or [])
    if mode == "TRAIN":
        s |= set(getattr(P, "FIXED_MASK_NODES_TRAIN", []) or [])
    elif mode == "TEST":
        s |= set(getattr(P, "FIXED_MASK_NODES_TEST", []) or [])
    return s


def _apply_masks(
    x: np.ndarray,
    m: np.ndarray,
    mode: str,
    missing_ratio: float,
    missing_ratio_test: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """x,m shape [1, N] before channel expand."""
    P = geo.P
    n_nodes = x.shape[1]
    fixed = _fixed_set_for_mode(mode)

    for node in fixed:
        if 0 <= node < n_nodes:
            x[:, node] = 0.0
            m[:, node] = 0.0

    if not getattr(P, "MASK_RANDOM_ON_OTHER", True):
        return x, m

    ratio = missing_ratio if mode == "TRAIN" else missing_ratio_test
    other = [i for i in range(n_nodes) if i not in fixed]
    if not other:
        return x, m
    sub = rng.random((x.shape[0], len(other))) < ratio
    for j, node in enumerate(other):
        x[:, node][sub[:, j]] = 0.0
        m[:, node][sub[:, j]] = 0.0
    return x, m


def getXSYS_estimation(
    data,
    mode,
    missing_ratio=0.2,
    missing_ratio_test=0.2,
    epoch: Optional[int] = None,
):
    """
    TRAIN: if epoch is set, RNG = seed+epoch (per-epoch random on non-fixed nodes).
           if epoch is None and mode is TRAIN, still uses seed (setup-time train tensors).
    TEST/VAL tensors: call with epoch=None and fixed seed at setup.
    """
    P = geo.P
    TRAIN_NUM = int(data.shape[0] * P.TRAINRATIO)
    XS, YS, MS = [], [], []

    if mode == "TRAIN":
        base_seed = int(P.seed) + (int(epoch) if epoch is not None else 0)
        rng = np.random.default_rng(base_seed)
        for i in range(TRAIN_NUM):
            x = data[i : i + 1, :].copy()
            y = data[i : i + 1, :].copy()
            m = np.ones_like(x)
            x, m = _apply_masks(x, m, "TRAIN", missing_ratio, missing_ratio_test, rng)
            XS.append(x)
            YS.append(y)
            MS.append(m)
    elif mode == "TEST":
        rng = np.random.default_rng(int(P.seed_SS if P.seed_SS != -1 else P.seed) + 99991)
        for i in range(TRAIN_NUM, data.shape[0]):
            x = data[i : i + 1, :].copy()
            y = data[i : i + 1, :].copy()
            m = np.ones_like(x)
            x, m = _apply_masks(x, m, "TEST", missing_ratio, missing_ratio_test, rng)
            XS.append(x)
            YS.append(y)
            MS.append(m)
    else:
        raise ValueError(f"Unknown mode {mode}")

    XS, YS, MS = np.array(XS), np.array(YS), np.array(MS)
    XS = XS[:, :, :, np.newaxis].transpose(0, 3, 2, 1)
    YS = YS[:, :, :, np.newaxis]
    MS = MS[:, :, :, np.newaxis].transpose(0, 3, 2, 1)
    return XS, YS, MS


def setups_estimation(missing_ratio=0.2):
    """Same as base setups, but spatial split before masking and fixed policy saved."""
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
    print(P.KEYWORD, "data splits (estimation + fixed virtual-like mask)", time.ctime())

    data = geo.data
    data_ds = geo.data_ds

    spatialSplit_unseen = unseen_nodes.SpatialSplit(
        data.shape[1], r_trn=P.R_TRN, r_val=0.15, r_tst=0.15, seed=P.seed_SS
    )
    spatialSplit_allNod = unseen_nodes.SpatialSplit(
        data.shape[1], r_trn=P.R_TRN, r_val=min(1.0, P.R_TRN * 8 / 7), r_tst=1.0, seed=P.seed_SS
    )
    init_fixed_mask_policy(spatialSplit_unseen, data.shape[1])
    save_mask_policy(P.PATH)
    print(
        "[fixedmask] train_fullmask_nodes",
        P.FIXED_MASK_NODES_TRAIN,
        "test_fullmask_nodes",
        P.FIXED_MASK_NODES_TEST,
    )

    trainXS, trainYS, trainMS = getXSYS_estimation(data, "TRAIN", missing_ratio=missing_ratio)
    testXS, testYS, testMS = getXSYS_estimation(
        data, "TEST", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio
    )

    if P.IS_DESEASONED:
        trainXS_ds, _, trainMS_ds = getXSYS_estimation(data_ds, "TRAIN", missing_ratio=missing_ratio)
        testXS_ds, _, testMS_ds = getXSYS_estimation(
            data_ds, "TEST", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio
        )
        trainXS = np.concatenate((trainXS, trainXS_ds), axis=1)
        testXS = np.concatenate((testXS, testXS_ds), axis=1)
        trainMS = np.concatenate((trainMS, trainMS_ds), axis=1)
        testMS = np.concatenate((testMS, testMS_ds), axis=1)

    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (1 - P.TRAINVALSPLIT))
    XS_torch_trn = trainXS[: P.train_size]
    YS_torch_trn = trainYS[: P.train_size]
    MS_torch_trn = trainMS[: P.train_size]
    XS_torch_val = trainXS[P.train_size :]
    YS_torch_val = trainYS[P.train_size :]
    MS_torch_val = trainMS[P.train_size :]

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

    train_data = torch.utils.data.TensorDataset(XS_torch_train, YS_torch_train, MS_torch_train)
    val_u_data = torch.utils.data.TensorDataset(XS_torch_val_u, YS_torch_val_u, MS_torch_val_u)
    val_a_data = torch.utils.data.TensorDataset(XS_torch_val_a, YS_torch_val_a, MS_torch_val_a)
    tst_u_data = torch.utils.data.TensorDataset(XS_torch_tst_u, YS_torch_tst_u, MS_torch_tst_u)
    tst_a_data = torch.utils.data.TensorDataset(XS_torch_tst_a, YS_torch_tst_a, MS_torch_tst_a)

    num_workers = 8
    pin_memory = True if device.type == "cuda" else False
    train_iter = torch.utils.data.DataLoader(
        train_data, P.BATCHSIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.BATCHSIZE, shuffle=False)
    val_a_iter = torch.utils.data.DataLoader(val_a_data, P.BATCHSIZE, shuffle=False)
    tst_u_iter = torch.utils.data.DataLoader(tst_u_data, P.BATCHSIZE, shuffle=False)
    tst_a_iter = torch.utils.data.DataLoader(tst_a_data, P.BATCHSIZE, shuffle=False)

    adj_mx = geo.load_adj(P.ADJPATH, P.ADJTYPE, P.DATANAME)
    print(adj_mx)
    adj_train = [
        torch.tensor(i[spatialSplit_unseen.i_trn, :][:, spatialSplit_unseen.i_trn]).to(device)
        for i in adj_mx
    ]
    adj_val_u = [
        torch.tensor(i[spatialSplit_unseen.i_val, :][:, spatialSplit_unseen.i_val]).to(device)
        for i in adj_mx
    ]
    adj_val_a = [
        torch.tensor(i[spatialSplit_allNod.i_val, :][:, spatialSplit_allNod.i_val]).to(device)
        for i in adj_mx
    ]
    adj_tst_u = [
        torch.tensor(i[spatialSplit_unseen.i_tst, :][:, spatialSplit_unseen.i_tst]).to(device)
        for i in adj_mx
    ]
    adj_tst_a = [
        torch.tensor(i[spatialSplit_allNod.i_tst, :][:, spatialSplit_allNod.i_tst]).to(device)
        for i in adj_mx
    ]

    pretrn_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(XS_torch_train[:, -1, :, 0].T),
        P.BATCHSIZE,
        shuffle=True,
    )
    preval_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(trainYS[:, -1, spatialSplit_unseen.i_val, 0]).T.float()
        ),
        P.BATCHSIZE,
        shuffle=False,
    )
    pretrn_iterg = random.sample(list(spatialSplit_unseen.i_trn), P.BATCHSIZE)
    preval_iterg = list(spatialSplit_unseen.i_val)

    for k, v in vars(P).items():
        print(k, v)
    mapping_tst_u = {old: new for new, old in enumerate(spatialSplit_unseen.i_tst)}

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


def trainModel_estimation_with_pretrain(
    name,
    train_iter,
    val_u_iter,
    val_a_iter,
    adj_train,
    adj_val_u,
    adj_val_a,
    spatialSplit_unseen,
    spatialSplit_allNod,
    pretrn_iterg,
    preval_iterg,
):
    """Copy of geo train loop: per-epoch mask uses getXSYS_estimation(..., epoch=epoch)."""
    data = geo.data
    data_ds = geo.data_ds
    print("trainModel (Estimation + Pretrain, fixedmask) Started ...", time.ctime())
    model = geo.getModel(name, device)
    criterion = geo.masked_loss
    s_time = datetime.now()
    gate_module = None
    temp_full_embed = geo_full_embed = None

    if P.IS_PRETRN:
        encoder = geo.Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
        encoder.load_state_dict(torch.load(P.PATH + "/" + "encoder" + ".pt", map_location=device))
        encoderg = geo.Geometric_Encoder(P.TEMPERATURE, P.FEATURES, P.GRAPH_NORM, P.HIDDEN).to(device)
        encoderg.load_state_dict(torch.load(P.PATH + "/encoderg.pt", map_location=device))
        temp_full_embed = geo._temporal_full_embed_est(encoder)
        geo_full_embed = geo._geometric_full_embed_est(encoderg)
        gate_module = geo.TemporalBaseDeltaFusion(
            embed_dim=temp_full_embed.shape[0],
            hidden_dim=getattr(P, "GATE_HIDDEN", 64),
            use_projection=True,
        ).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(gate_module.parameters()),
            lr=P.LEARN,
            weight_decay=P.weight_decay,
        )
        print("Using temporal_delta fusion, embed_dim=", temp_full_embed.shape[0])
        gate_module.eval()
        with torch.no_grad():
            fe0, _ = geo._fuse_embeddings_temporal_delta(
                temp_full_embed, geo_full_embed, gate_module, return_gate=True
            )
        train_embed = fe0[:, spatialSplit_unseen.i_trn]
        val_u_embed = fe0[:, spatialSplit_unseen.i_val]
        val_a_embed = fe0[:, spatialSplit_allNod.i_val]
    else:
        train_embed = torch.zeros(32, train_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_u_embed = torch.zeros(32, val_u_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_a_embed = torch.zeros(32, val_a_iter.dataset.tensors[0].shape[2]).to(device).detach()
        optimizer = torch.optim.Adam(model.parameters(), lr=P.LEARN, weight_decay=P.weight_decay)

    print("train_embed", train_embed.shape)
    print("val_u_embed", val_u_embed.shape)
    print("val_a_embed", val_a_embed.shape)
    min_val_loss = float("inf")

    for epoch in range(P.EPOCH):
        model.train()
        epoch_loss, n = 0.0, 0
        start_time = datetime.now()
        trainXS_ep, trainYS_ep, trainMS_ep = getXSYS_estimation(
            data, "TRAIN", missing_ratio=P.MISS_RATIO, epoch=epoch
        )
        if P.IS_DESEASONED:
            trainXS_ds_ep, _, trainMS_ds_ep = getXSYS_estimation(
                data_ds, "TRAIN", missing_ratio=P.MISS_RATIO, epoch=epoch
            )
            trainXS_ep = np.concatenate((trainXS_ep, trainXS_ds_ep), axis=1)
            trainMS_ep = np.concatenate((trainMS_ep, trainMS_ds_ep), axis=1)

        XS_ep = trainXS_ep[: P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        YS_ep = trainYS_ep[: P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        MS_ep = trainMS_ep[: P.train_size][:, :, spatialSplit_unseen.i_trn, :]

        train_iter_epoch = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(XS_ep), torch.Tensor(YS_ep), torch.Tensor(MS_ep)
            ),
            P.BATCHSIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True if device.type == "cuda" else False,
        )
        if gate_module is not None:
            gate_module.train()

        for x, y, mask in train_iter_epoch:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            y = y.squeeze(-1)
            if gate_module is not None:
                train_full, tr_delta = geo._fuse_embeddings_temporal_delta(
                    temp_full_embed, geo_full_embed, gate_module, return_gate=True
                )
                train_embed_ep = train_full[:, spatialSplit_unseen.i_trn]
                y_pred = model(x, adj_train, train_embed_ep)
                loss = criterion(y_pred, y, mask)
                dr = getattr(P, "DELTA_REG", 0.0)
                if dr > 0 and tr_delta is not None:
                    d = tr_delta[:, spatialSplit_unseen.i_trn]
                    loss = loss + dr * d.pow(2).mean()
            else:
                y_pred = model(x, adj_train, train_embed)
                loss = criterion(y_pred, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        train_loss = epoch_loss / max(n, 1)
        if gate_module is not None:
            gate_module.eval()
            with torch.no_grad():
                fe_ep, full_delta = geo._fuse_embeddings_temporal_delta(
                    temp_full_embed, geo_full_embed, gate_module, return_gate=True
                )
            train_embed = fe_ep[:, spatialSplit_unseen.i_trn]
            val_u_embed = fe_ep[:, spatialSplit_unseen.i_val]
            val_a_embed = fe_ep[:, spatialSplit_allNod.i_val]
        val_u_loss = geo.evaluateModel_estimation_with_pretrain(
            model, val_u_iter, val_u_embed, adj_val_u
        )
        val_a_loss = geo.evaluateModel_estimation_with_pretrain(
            model, val_a_iter, val_a_embed, adj_val_a
        )
        if val_u_loss < min_val_loss:
            min_val_loss = val_u_loss
            torch.save(model.state_dict(), f"{P.PATH}/{name}_best.pt")
        print(
            f"Epoch {epoch}, Time {(datetime.now() - start_time).seconds}s, "
            f"Train Loss: {train_loss:.6f}, Val_U: {val_u_loss:.6f}, Val_A: {val_a_loss:.6f}"
        )
        with open(f"{P.PATH}/{name}_log.txt", "a") as f:
            f.write(
                f"epoch,{epoch},train_loss,{train_loss:.10f},"
                f"val_u_loss,{val_u_loss:.10f},val_a_loss,{val_a_loss:.10f}\n"
            )

    print("TRAINING FINISHED. Best val_u:", min_val_loss, "duration:", datetime.now() - s_time)
    score = geo.evaluateModel_estimation_with_pretrain(model, train_iter, train_embed, adj_train)
    with open(f"{P.PATH}/{name}_prediction_scores.txt", "a") as f:
        f.write(f"{name}, estimation, MAE on train, {score:.10f}, {score:.10f}\n")


def patch_geo_module() -> None:
    geo.getXSYS_estimation = getXSYS_estimation
    geo.setups_estimation = setups_estimation
    geo.trainModel_estimation_with_pretrain = trainModel_estimation_with_pretrain


_orig_get_argv = geo.get_argv


def get_argv():
    _orig_get_argv()
    _read_mask_config_from_env()


def main():
    patch_geo_module()
    geo.get_argv = get_argv
    geo.main()


if __name__ == "__main__":
    main()
