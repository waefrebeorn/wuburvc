#!/usr/bin/env python3
"""Detect RVC model architecture (v1 vs v2) from a .pth checkpoint.

Reads the config list embedded in the .pth and prints a structured summary
of the model's architecture parameters, including:
- Version (v1/v2)
- Sample rate
- Content dimension (256 for v1, 768 for v2)
- Hidden channels
- Upsample rates and kernel sizes
- Speaker embedding dimension
- Generator output channels (for PixelShuffle detection)

Usage: python3 detect_model_config.py model.pth
"""
import sys, json, os

def detect_config(src_pth):
    import torch
    ckpt = torch.load(src_pth, map_location='cpu', weights_only=False)

    config = ckpt.get('config', [])
    if isinstance(config, str):
        try:
            import ast
            config = ast.literal_eval(config)
        except:
            config = []

    version_str = ckpt.get('version', 'v2')
    if isinstance(version_str, str):
        version = 2 if 'v2' in version_str.lower() else 1
    else:
        version = 2

    result = {
        'version': version,
        'version_str': version_str,
        'config': list(config) if isinstance(config, (list, tuple)) else [],
    }

    # Parse config array
    # v1/v2 config layout:
    # [0] spec_channels, [1] segment_size, [2] inter_channels, [3] hidden_channels (192)
    # [4] content_dim (256 for v1, 768 for v2)
    # [5] ?, [6] ?, [7] ?, [8] ?
    # [9] version string ('1' or '2')
    # [10] resblock_kernels ([3,7,11])
    # [11] resblock_dilation_sizes
    # [12] upsample_rates
    # [13] upsample_initial_channel (512)
    # [14] upsample_kernel_sizes ([16,16,4,4])
    # [15] spk_embed_dim (109 or 128)
    # [16] hidden_channels (256)
    # [17] sample_rate (40000, 48000, 25600, 22050)
    if len(config) > 0:
        result['spec_channels'] = config[0] if config[0] else 513
    if len(config) > 4:
        result['content_dim'] = config[4]
        # Verify version from content_dim
        if config[4] == 768:
            result['version'] = 2
        elif config[4] == 256:
            result['version'] = 1
    if len(config) > 12:
        result['upsample_rates'] = list(config[12]) if isinstance(config[12], (list, tuple)) else config[12]
    if len(config) > 13:
        result['upsample_initial_channel'] = config[13]
    if len(config) > 14:
        result['upsample_kernel_sizes'] = list(config[14]) if isinstance(config[14], (list, tuple)) else config[14]
    if len(config) > 15:
        result['spk_embed_dim'] = config[15]
    if len(config) > 16:
        result['hidden_channels'] = config[16]
    if len(config) > 17:
        result['sample_rate'] = config[17]

    # Detect from tensor names if config is missing
    sd = ckpt.get('weight', ckpt.get('model', {})) if isinstance(ckpt, dict) else {}
    tensor_keys = list(sd.keys())

    # Detect v1 vs v2 from tensor names
    has_encoder_layers = any('encoder.attn_layers' in k for k in tensor_keys)
    has_ffn_layers = any('encoder.ffn_layers' in k for k in tensor_keys)
    has_final_proj = any('final_proj' in k for k in tensor_keys)
    has_spk_embed = any('emb_g' in k or 'speaker' in k.lower() for k in tensor_keys)

    # Detect content_dim from emb_phone weight shape
    for k, v in sd.items():
        if 'emb_phone.weight' in k:
            result['detected_content_dim'] = v.shape[1] if len(v.shape) >= 2 else 256
            break

    # Detect PixelShuffle (v2: conv_post outputs 32 channels, v1: 256 channels)
    for k, v in sd.items():
        if 'dec.conv_post.weight' in k:
            result['conv_post_channels'] = v.shape[1]  # out_channels
            break

    # Detect upsample rates from dec.ups weight kernel sizes
    upsample_kernels = []
    for k, v in sd.items():
        if 'dec.ups.' in k and 'weight_v' in k:
            idx = int(k.split('.')[2])
            k_size = v.shape[2] if len(v.shape) >= 3 else 1
            upsample_kernels.append((idx, k_size))
    upsample_kernels.sort()
    result['detected_upsample_kernels'] = [k for _, k in upsample_kernels]

    result['has_transformer_encoder'] = has_ffn_layers
    result['has_final_proj'] = has_final_proj
    result['has_spk_embed'] = has_spk_embed

    # Infer upsample rates from kernel sizes
    if 'detected_upsample_kernels' in result:
        kernel_to_stride = {16: 10, 24: 12, 20: 10, 4: 2}
        detected_rates = []
        for k_size in result['detected_upsample_kernels']:
            detected_rates.append(kernel_to_stride.get(k_size, k_size))
        result['detected_upsample_rates'] = detected_rates

    return result

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model.pth>")
        sys.exit(1)
    info = detect_config(sys.argv[1])
    print(json.dumps(info, indent=2, default=str))
