#!/usr/bin/env python3
"""Dump HuBERT intermediate activations from the torch port for C-side diffing."""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from wubu_hubert_torch import HuBERT

REF = r"C:\Users\eman5\WuBuMedia\outputs\rvc_ref"
pcm = np.load(os.path.join(REF, "pcm16k.npy"))
h = HuBERT(r"C:\Users\eman5\WuBuMedia\models\rvc\hubert_base.pt")

src = torch.from_numpy(pcm).unsqueeze(0)

# 1. conv front-end
with torch.no_grad():
    feats = h.feature_extractor(src)          # [1, 512, T']
    feats = feats.transpose(1, 2)             # [1, T', 512]  (non-contiguous!)
    np.save(os.path.join(REF, "dbg_conv.npy"), np.ascontiguousarray(feats[0]))
    feats = h.layer_norm(feats)
    np.save(os.path.join(REF, "dbg_conv_ln.npy"), np.ascontiguousarray(feats[0]))
    feats = h.post_extract_proj(feats)
    np.save(os.path.join(REF, "dbg_postproj.npy"), np.ascontiguousarray(feats[0]))

    # 2. pos_conv (extract_features applies pre-norm internally; don't
    # pre-apply layer_norm here or it double-norms)
    x_conv = h.encoder.pos_conv(feats.transpose(1, 2)).transpose(1, 2)
    np.save(os.path.join(REF, "dbg_posconv.npy"), np.ascontiguousarray(x_conv[0]))
    x_pre = h.encoder.layer_norm(feats + x_conv)  # for preln dump only
    np.save(os.path.join(REF, "dbg_preln.npy"), np.ascontiguousarray(x_pre[0]))

    # 3. full encoder via extract_features (correct: pos_conv + pre-norm +
    #    12 layers). Pass raw post_proj feats so the internal pre-norm is
    #    applied exactly once.
    full, _ = h.encoder.extract_features(feats)
    np.save(os.path.join(REF, "dbg_final.npy"), np.ascontiguousarray(full[0]))

    # 4. per-layer dumps using the same path as extract_features
    T = feats.size(1)
    pad = (2 - T % 2) % 2
    xp = torch.nn.functional.pad(feats, (0, 0, 0, pad))
    mask = torch.zeros((1, xp.size(1)), dtype=torch.bool)
    mask[:, -pad:] = True
    x0 = xp.transpose(0, 1)
    for L in range(12):
        out = h.encoder.layers[L](x0, self_attn_padding_mask=mask)
        x0 = out[0]
        np.save(os.path.join(REF, "dbg_layer%d.npy" % (L + 1)),
                np.ascontiguousarray(x0.transpose(0, 1)[0, :T]))
    print("dumped dbg_*.npy; shapes:", feats.shape, x_conv.shape, x_pre.shape)
