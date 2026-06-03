"""
STGFormer 替换 `pred_GWN_16_adpAdj_12_step.py` 中的下游 GWN。

- 复用原脚本的数据/预训练/训练流程（import + monkeypatch）。
- 将 (B, C, N, T) 转为 STGFormer 需要的 (B, T, N, input_dim)。
- **使用预训练空间编码 embed**：形状与 GWN 一致为 (D, N)，按节点维与 x 对齐、padding 后
  在通道维拼接进 STGFormer 输入（等价于把 SCPT 编码当作额外节点特征）。
- 训练时对 Adam 的 step 前做梯度裁剪，减轻 STGFormer 注意力在若干 epoch 后的 **NaN/爆炸**。
"""

from __future__ import annotations

import os
import sys


def _ensure_conda_runtime_libs() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    conda_lib = os.path.join(conda_prefix, "lib")
    if not os.path.isdir(conda_lib):
        return

    ld = os.environ.get("LD_LIBRARY_PATH") or ""
    parts = [p for p in ld.split(":") if p]
    if parts and parts[0] == conda_lib:
        return
    if conda_lib in parts:
        parts.remove(conda_lib)
    parts.insert(0, conda_lib)
    new_ld = ":".join(parts)

    if os.environ.get("_STGFORMER_REEXEC_DONE") == "1":
        return
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = new_ld
    new_env["_STGFORMER_REEXEC_DONE"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)


_ensure_conda_runtime_libs()

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_gwn_on_path() -> None:
    """使 `pred_GWN_16_adpAdj` 从 `9991/gith/GWN/` 加载（及其同目录依赖）。"""
    gwn_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GWN"))
    if gwn_dir not in sys.path:
        sys.path.insert(0, gwn_dir)


_ensure_gwn_on_path()

import pred_GWN_16_adpAdj as base


def _import_stgformer():
    stg_model_dir = os.path.join(
        os.path.dirname(__file__), "STGFormer", "STGformer-main", "model"
    )
    if stg_model_dir not in sys.path:
        sys.path.insert(0, stg_model_dir)
    from STGformer import STGformer  # type: ignore

    return STGformer


def _embed_dim_default() -> int:
    return int(os.environ.get("STGFORMER_EMBED_DIM", "32"))


class STGFormerGWNCallAdapter(nn.Module):
    """
    与 GWN 相同调用: forward(x, adj, embed)，其中
    - x: (B, C, N, T)
    - embed: (D, N) 预训练节点编码；无预训练时可为全零或与 x 同设备的 tensor
    """

    def __init__(
        self,
        stgformer: nn.Module,
        num_nodes_max: int,
        channel: int,
        embed_dim: int,
    ):
        super().__init__()
        self.stgformer = stgformer
        self.num_nodes_max = int(num_nodes_max)
        self.channel = int(channel)
        self.embed_dim = int(embed_dim)
        self.embed_ln = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor, adj, embed) -> torch.Tensor:
        del adj
        b, c, n, t = x.shape
        nmax = self.num_nodes_max
        if n > nmax:
            raise ValueError(f"batch nodes {n} > N_NODE {nmax}")
        if c != self.channel:
            raise ValueError(f"expected C={self.channel}, got {c}")

        if n < nmax:
            pad = x.new_zeros(b, c, nmax - n, t)
            x = torch.cat([x, pad], dim=2)

        x_bt = x.permute(0, 3, 2, 1).contiguous()

        d = self.embed_dim
        if embed is None:
            emb = x.new_zeros(d, nmax)
        else:
            emb = embed.to(device=x.device, dtype=x.dtype)
            if emb.dim() != 2:
                raise ValueError(f"embed must be (D, N), got {tuple(emb.shape)}")
            if emb.shape[0] != d:
                raise ValueError(f"embed dim {emb.shape[0]} != expected {d}")
            if emb.shape[1] != n:
                raise ValueError(f"embed N {emb.shape[1]} != x N {n}")
            if n < nmax:
                emb = F.pad(emb, (0, nmax - n), value=0.0)

        emb_nt = emb.transpose(0, 1)
        emb_bt = self.embed_ln(emb_nt).unsqueeze(0).unsqueeze(1).expand(b, t, nmax, d)
        x_in = torch.cat([x_bt, emb_bt], dim=-1)

        out = self.stgformer(x_in)
        if n < nmax:
            out = out[:, :, :n, :].contiguous()
        return out


def getModel(name: str, device: torch.device):  # noqa: ARG001
    STGformer = _import_stgformer()
    num_nodes = int(getattr(base.P, "N_NODE", 207))
    in_steps = int(base.P.TIMESTEP_IN)
    out_steps = int(base.P.TIMESTEP_OUT)
    channel = int(base.P.CHANNEL)
    embed_dim = _embed_dim_default()
    input_dim = channel + embed_dim

    model = STGformer(
        num_nodes=num_nodes,
        in_steps=in_steps,
        out_steps=out_steps,
        steps_per_day=288,
        input_dim=input_dim,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=0,
        dow_embedding_dim=0,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=12,
        num_heads=4,
        supports=[torch.eye(num_nodes, device=device)],
        num_layers=3,
        dropout=0.1,
        mlp_ratio=2,
        use_mixed_proj=True,
        dropout_a=0.3,
        kernel_size=[1],
    ).to(device)

    return STGFormerGWNCallAdapter(
        model,
        num_nodes_max=num_nodes,
        channel=channel,
        embed_dim=embed_dim,
    ).to(device)


_orig_adam = torch.optim.Adam


class AdamClipGradNorm(_orig_adam):
    """在 step 前对梯度做 global norm 裁剪，缓解 STGFormer 训练中后期 NaN。"""

    def __init__(self, *args, max_norm: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._stf_max_norm = float(max_norm)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            ps = [p for p in group["params"] if p.grad is not None]
            if ps:
                torch.nn.utils.clip_grad_norm_(ps, self._stf_max_norm)
        return super().step(closure=closure)


def main():
    if os.environ.get("STGFORMER_SMOKE") == "1":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = getModel("STGFormer", dev)
        n = int(getattr(base.P, "N_NODE", 207))
        d = _embed_dim_default()
        x = torch.zeros(2, int(base.P.CHANNEL), n, int(base.P.TIMESTEP_IN), device=dev)
        emb = torch.randn(d, n, device=dev) * 0.1
        y = m(x, None, emb)
        print("SMOKE y.shape", tuple(y.shape))
        return

    base.P.MODELNAME = "STGFormer"
    base.getModel = getModel  # type: ignore[assignment]

    max_norm = float(os.environ.get("STGFORMER_CLIP_NORM", "5.0"))
    torch.optim.Adam = lambda *a, **k: AdamClipGradNorm(*a, max_norm=max_norm, **k)
    try:
        base.main()
    finally:
        torch.optim.Adam = _orig_adam


if __name__ == "__main__":
    main()
