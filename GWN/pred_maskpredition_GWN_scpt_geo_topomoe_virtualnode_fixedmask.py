"""TopoMoE estimation with virtual-node fixed full-node masks.

This entrypoint is a named wrapper around the fixedmask implementation:
selected nodes are fully masked at every timestep to simulate virtual nodes
with no historical observations, while non-selected nodes keep the original
point-wise random masking behavior.
"""
from __future__ import annotations

import pred_maskpredition_GWN_scpt_geo_topomoe_fixedmask as fixedmask


if __name__ == "__main__":
    fixedmask.fm.patch_geo_module()
    fixedmask.est.get_argv = fixedmask.topomoe.get_argv_topomoe_estimation
    fixedmask.est.trainModel_estimation_with_pretrain = (
        fixedmask.trainModel_estimation_with_pretrain_topomoe_fixedmask
    )
    fixedmask.est.testModel_estimation_with_pretrain = fixedmask._orig_test_topomoe
    fixedmask.est.main()
