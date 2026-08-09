#!/usr/bin/env python3
"""wubu_hubert_torch.py — Real HuBERT content encoder in pure torch, no fairseq.

Faithful port of fairseq HubertModel.forward / extract_features for the
lj1995 hubert_base.pt checkpoint used by RVC:

  conv feature extractor (7 blocks: Conv1d no-bias + LayerNorm + GELU,
    strides [5,2,2,2,2,2,2], kernels [10,3,3,3,3,2,2])
  -> layer_norm(512)
  -> post_extract_proj (512 -> 768)
  -> pos_conv (Conv1d 768, k=128, groups=16, weight_norm dim=2) + SamePad + GELU
  -> encoder.layer_norm (pre-norm, layer_norm_first=False)
  -> 12 x TransformerSentenceEncoderLayer (post-LN):
       x = self_attn(x); x = LN(x + attn)
       x = fc1 -> GELU -> fc2; x = LN(x + ffn)
  -> output layer 9 (v1, then final_proj 768->256) or layer 12 (v2, 768 dim)

Usage:
  from wubu_hubert_torch import HuBERT
  h = HuBERT('models/rvc/hubert_base.pt')
  feats = h.extract(pcm16k, version='v2')   # [n_frames, 768] or [n_frames, 256]

License: WaefreBeorn-UMV3
"""
import os
import sys
import types
import importlib.abc
import importlib.machinery
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Dummy:
    def __init__(self, *a, **k):
        pass
    def __setattr__(self, n, v):
        pass


class _StubMod(types.ModuleType):
    def __getattr__(self, name):
        if name in ("__file__", "__cached__", "__spec__", "__loader__",
                    "__package__", "__path__"):
            return ""
        d = _Dummy
        d.__name__ = name
        return d


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = _StubMod(spec.name)
        mod.__path__ = []
        # Real path so inspect.getfile / bytecode-suffix checks don't reject
        # the stub as a "built-in module".
        mod.__file__ = os.path.abspath(__file__)
        sys.modules[spec.name] = mod
        return mod
    def exec_module(self, module):
        pass


class _FairseqBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fairseq" or fullname.startswith("fairseq."):
            return importlib.machinery.ModuleSpec(
                fullname, loader=_StubLoader(), is_package=True)
        return None


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        output = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class TransposeLast(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)


def fairseq_gelu(x):
    """fairseq/modules/gelu.py — tanh approximation with x^3.
    NOT torch.nn.GELU (erf). hubert_base.pt was trained/run with this."""
    return x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


class FairseqGELU(nn.Module):
    def forward(self, x):
        return fairseq_gelu(x)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, conv_layers, mode="default", conv_bias=False):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_d = 1
        for i, (dim, k, stride) in enumerate(conv_layers):
            def block():
                layers = [
                    nn.Conv1d(in_d, dim, k, stride=stride, bias=conv_bias),
                    nn.Dropout(p=0.0),
                ]
                # fairseq mode="default": only block 0 gets GroupNorm
                if i == 0:
                    layers.append(nn.GroupNorm(dim, dim, affine=True))
                layers.append(FairseqGELU())
                return nn.Sequential(*layers)
            self.conv_layers.append(block())
            in_d = dim

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class FairseqMHA(nn.Module):
    """Faithful fairseq MultiheadAttention (no fused fastpath):
    q_proj/k_proj/v_proj/out_proj Linear layers, scaled dot-product,
    key_padding_mask. Matches hubert checkpoint tensor names."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.qkv_same_embed_dim = True
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None):
        """query/key/value: [T, B, C]. Returns (attn_out, attn_weights)."""
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = self.head_dim
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query)  # [T, B, C]
        k = self.k_proj(key)
        v = self.v_proj(value)

        # [T, B, H, D] -> [B*H, T, D]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, head_dim).transpose(0, 1)

        q = q * scaling
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # [B*H, T, S]

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0)
        if key_padding_mask is not None:
            # [B, S] -> [B, 1, S] broadcast over heads and query positions
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
            attn_weights = attn_weights.masked_fill(
                mask.expand(-1, self.num_heads, tgt_len, -1).reshape(
                    bsz * self.num_heads, tgt_len, src_len), float("-inf"))

        attn_weights = torch.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)  # [B*H, T, D]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            aw = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn, aw
        return attn, None


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_embedding_dim, num_attention_heads,
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.0,
                 activation_fn="gelu", layer_norm_first=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.self_attn = FairseqMHA(
            self.embedding_dim, num_attention_heads, dropout=attention_dropout,
            bias=True)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.layer_norm_first = layer_norm_first
        self.activation_fn = FairseqGELU()

    def forward(self, x, self_attn_padding_mask=None, need_weights=False):
        residual = x
        out = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False)
        x = out[0] if isinstance(out, tuple) else out
        attn = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        x = self.dropout(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, layers,
                 conv_pos=128, conv_pos_groups=16, layer_norm_first=False,
                 required_seq_len_multiple=2):
        super().__init__()
        self.embedding_dim = embed_dim
        self.layer_norm_first = layer_norm_first
        self.required_seq_len_multiple = required_seq_len_multiple

        pos_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=conv_pos,
                             padding=conv_pos // 2, groups=conv_pos_groups)
        nn.init.normal_(pos_conv.weight, mean=0, std=0.01)
        nn.init.constant_(pos_conv.bias, 0)
        pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(pos_conv, SamePad(conv_pos), FairseqGELU())

        self.layers = nn.ModuleList([
            TransformerSentenceEncoderLayer(
                embedding_dim=embed_dim,
                ffn_embedding_dim=ffn_embed_dim,
                num_attention_heads=attention_heads,
                dropout=0.1, attention_dropout=0.1, activation_dropout=0.0,
                activation_fn="gelu", layer_norm_first=layer_norm_first)
            for _ in range(layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layerdrop = 0.0

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None,
                         min_layer=0):
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to multiple
        T = x.size(1)
        pad = (self.required_seq_len_multiple - T % self.required_seq_len_multiple) % self.required_seq_len_multiple
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            if padding_mask is not None:
                padding_mask = F.pad(padding_mask, (0, pad), value=True)
            else:
                # fairseq creates the mask when padding_mask is None
                padding_mask = torch.zeros((x.size(0), x.size(1)), dtype=torch.bool,
                                           device=x.device)
                padding_mask[:, -pad:] = True

        x = F.dropout(x, p=0.0, training=self.training)
        x = x.transpose(0, 1)  # T x B x C

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            out = layer(x, self_attn_padding_mask=padding_mask)
            x = out[0]
            z = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            lr = None
            if i >= min_layer:
                layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r
        x = x.transpose(0, 1)
        if pad:
            x = x[:, :-pad]
        return x, layer_results


class HuBERT(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.cfg = {
            "extractor_mode": "default",
            "encoder_layers": 12,
            "encoder_embed_dim": 768,
            "encoder_ffn_embed_dim": 3072,
            "encoder_attention_heads": 12,
            "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
            "conv_pos": 128,
            "conv_pos_groups": 16,
            "layer_norm_first": False,
            "final_dim": 0,
        }
        conv_layers = eval(self.cfg["conv_feature_layers"])
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers, mode=self.cfg["extractor_mode"], conv_bias=False)
        self.post_extract_proj = nn.Linear(512, 768)
        self.encoder = TransformerEncoder(
            embed_dim=768,
            ffn_embed_dim=3072,
            attention_heads=12,
            layers=12,
            conv_pos=128,
            conv_pos_groups=16,
            layer_norm_first=False,
            required_seq_len_multiple=2,
        )
        self.layer_norm = nn.LayerNorm(512)
        self.final_proj_linear = nn.Linear(768, 256)
        self.mask_emb = nn.Parameter(torch.zeros(768).uniform_())
        if checkpoint_path:
            self.load_weights(checkpoint_path)
        self.eval()

    def load_weights(self, checkpoint_path):
        if checkpoint_path not in sys.modules:
            sys.meta_path.insert(0, _FairseqBlocker())
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        own = self.state_dict()
        # checkpoint key is `final_proj.weight`; module attr is final_proj_linear
        keymap = {"final_proj.weight": "final_proj_linear.weight",
                  "final_proj.bias": "final_proj_linear.bias"}
        missing = []
        for k, v in sd.items():
            kk = keymap.get(k, k)
            if kk in own:
                if own[kk].shape == v.shape:
                    own[kk].copy_(v.detach().cpu())
                else:
                    print(f"[hubert] shape mismatch {k}: {own[kk].shape} vs {v.shape}")
            else:
                missing.append(k)
        if missing:
            print(f"[hubert] {len(missing)} unused keys (ok): {missing[:6]}...")

    @torch.no_grad()
    def extract(self, pcm, version="v2", output_layer=None):
        """pcm: float32 numpy [n_samples] at 16kHz. Returns [n_frames, dim]."""
        src = torch.from_numpy(np.asarray(pcm, dtype=np.float32))
        if src.dim() == 1:
            src = src.unsqueeze(0)  # [1, T]
        padding_mask = torch.zeros(src.shape, dtype=torch.bool)

        features = self.feature_extractor(src)          # [1, 512, T']
        features = features.transpose(1, 2)             # [1, T', 512]
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)     # [1, T', 768]
        features = F.dropout(features, p=0.0, training=False)

        if output_layer is None:
            output_layer = 9 if version == "v1" else 12
        x, _ = self.encoder(features, padding_mask=None, layer=output_layer)

        if version == "v1":
            x = self.final_proj_linear(x)
        return x[0].float().numpy()                     # [T', dim]

    @staticmethod
    def output_length(n_samples):
        """conv output length formula (fairseq _get_feat_extract_output_lengths)."""
        L = float(n_samples)
        for k, s in ((10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)):
            L = (L - k) / s + 1
        return int(L)

    # ── fairseq-compatible API (used by Mangio vc_infer_pipeline) ──
    @torch.no_grad()
    def extract_features(self, source, padding_mask=None, output_layer=None,
                         *a, **kw):
        """Fairseq-style extract_features(source, padding_mask, output_layer).
        source: [1, n_samples] float (16kHz mono).
        Returns [feats] where feats is [1, n_frames, 768] (v2 path)."""
        if source.dim() == 1:
            source = source.unsqueeze(0)
        if output_layer is None:
            output_layer = 12
        features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)
        features = F.dropout(features, p=0.0, training=False)
        x, _ = self.encoder(features, padding_mask=None, layer=output_layer)
        return [x]  # [1, T, 768]

    @torch.no_grad()
    def final_proj(self, x):
        """v1 path: project 768 -> 256 content."""
        return self.final_proj_linear(x)


if __name__ == "__main__":
    h = HuBERT(sys.argv[1] if len(sys.argv) > 1 else "models/rvc/hubert_base.pt")
    # quick synthetic test
    rng = np.random.RandomState(0)
    pcm = rng.randn(16000 * 2).astype(np.float32) * 0.1
    f2 = h.extract(pcm, "v2")
    f1 = h.extract(pcm, "v1")
    print("v2:", f2.shape, "mean", f2.mean().item(), "std", f2.std().item())
    print("v1:", f1.shape, "mean", f1.mean().item(), "std", f1.std().item())
