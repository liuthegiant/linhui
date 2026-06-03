"""TopoMoE + fixed full-node masks (virtual-node simulation)."""
from __future__ import annotations

import time
from typing import List

import numpy as np
import torch

import pred_maskpredition_GWN_scpt_geo_fixedmask as fm

fm.patch_geo_module()

import pred_maskpredition_GWN_scpt_geo_topomoe as topomoe  # noqa: E402 — after patch

est = topomoe.est
P = est.P

_orig_train_topomoe = topomoe.trainModel_estimation_with_pretrain_topomoe
_orig_test_topomoe = topomoe.testModel_estimation_with_pretrain_topomoe


def trainModel_estimation_with_pretrain_topomoe_fixedmask(
    name, train_iter, val_u_iter, val_a_iter, adj_train, adj_val_u, adj_val_a,
    spatialSplit_unseen, spatialSplit_allNod, pretrn_iterg, preval_iterg,
):
    if not topomoe._enabled():
        return fm.trainModel_estimation_with_pretrain(
            name, train_iter, val_u_iter, val_a_iter, adj_train, adj_val_u, adj_val_a,
            spatialSplit_unseen, spatialSplit_allNod, pretrn_iterg, preval_iterg,
        )
    print("trainModel Estimation with TopoMoE + fixedmask Started ...", time.ctime())
    model = est.getModel(name, est.device)
    criterion = est.masked_loss
    experts, names, ctx, adj_dense = topomoe._inputs()
    fusion = topomoe._make(experts, names, ctx)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(fusion.parameters()), lr=P.LEARN, weight_decay=P.weight_decay
    )
    with torch.no_grad():
        fe0, aux0 = topomoe._run(fusion, experts, ctx, adj_dense, True)
    train_embed = fe0[:, spatialSplit_unseen.i_trn]
    print("experts", names, "initial alpha mean", aux0["alpha"].mean(0).detach().cpu().numpy())
    min_val = float("inf")
    s_time = __import__("datetime").datetime.now()
    for epoch in range(P.EPOCH):
        st = __import__("datetime").datetime.now()
        model.train()
        fusion.train()
        epoch_loss = 0.0
        n = 0
        batch_times: List[float] = []
        trainXS_ep, trainYS_ep, trainMS_ep = fm.getXSYS_estimation(
            est.data, "TRAIN", missing_ratio=P.MISS_RATIO, epoch=epoch
        )
        if P.IS_DESEASONED:
            trainXS_ds_ep, _, trainMS_ds_ep = fm.getXSYS_estimation(
                est.data_ds, "TRAIN", missing_ratio=P.MISS_RATIO, epoch=epoch
            )
            trainXS_ep = np.concatenate((trainXS_ep, trainXS_ds_ep), axis=1)
            trainMS_ep = np.concatenate((trainMS_ep, trainMS_ds_ep), axis=1)
        XS_ep = trainXS_ep[: P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        YS_ep = trainYS_ep[: P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        MS_ep = trainMS_ep[: P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        it = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(XS_ep), torch.Tensor(YS_ep), torch.Tensor(MS_ep)
            ),
            P.BATCHSIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True if est.device.type == "cuda" else False,
        )
        for x, y, mask in it:
            tb = time.perf_counter()
            optimizer.zero_grad()
            x = x.to(est.device, non_blocking=True)
            y = y.to(est.device, non_blocking=True).squeeze(-1)
            mask = mask.to(est.device, non_blocking=True)
            full, aux = topomoe._run(fusion, experts, ctx, adj_dense, True)
            train_embed_ep = full[:, spatialSplit_unseen.i_trn]
            pred = model(x, adj_train, train_embed_ep)
            task_loss = criterion(pred, y, mask)
            loss = topomoe.add_moe_regularization(task_loss, aux, P)
            loss.backward()
            optimizer.step()
            epoch_loss += task_loss.item()
            n += 1
            batch_times.append(time.perf_counter() - tb)
        train_loss = epoch_loss / max(n, 1)
        model.eval()
        fusion.eval()
        with torch.no_grad():
            full, aux_eval = topomoe._run(fusion, experts, ctx, adj_dense, True)
            val_u_embed = full[:, spatialSplit_unseen.i_val]
            val_a_embed = full[:, spatialSplit_allNod.i_val]
            val_u_loss = est.evaluateModel_estimation_with_pretrain(
                model, val_u_iter, val_u_embed, adj_val_u
            )
            val_a_loss = est.evaluateModel_estimation_with_pretrain(
                model, val_a_iter, val_a_embed, adj_val_a
            )
        if val_u_loss < min_val:
            min_val = val_u_loss
            torch.save(model.state_dict(), f"{P.PATH}/{name}_best.pt")
            torch.save(fusion.state_dict(), f"{P.PATH}/{name}_fusion_u.pt")
            topomoe.save_alpha_report(P.PATH, f"{name}_est_u", aux_eval["alpha"], names)
        am = aux_eval["alpha"].mean(0).detach().cpu().numpy()
        print(
            f"Epoch {epoch}, Time {( __import__('datetime').datetime.now()-st).seconds}s, "
            f"Train Loss: {train_loss:.6f}, Val_U: {val_u_loss:.6f}, Val_A: {val_a_loss:.6f}, "
            f"alpha_mean: {am}"
        )
    print("TRAINING FINISHED. Best val_u loss:", min_val)


est.get_argv = topomoe.get_argv_topomoe_estimation
est.trainModel_estimation_with_pretrain = trainModel_estimation_with_pretrain_topomoe_fixedmask
est.testModel_estimation_with_pretrain = _orig_test_topomoe

if __name__ == "__main__":
    fm.patch_geo_module()
    est.get_argv = topomoe.get_argv_topomoe_estimation
    est.trainModel_estimation_with_pretrain = trainModel_estimation_with_pretrain_topomoe_fixedmask
    est.testModel_estimation_with_pretrain = _orig_test_topomoe
    est.main()
