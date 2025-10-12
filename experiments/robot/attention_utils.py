import os
import torch
import matplotlib.pyplot as plt

def save_attention_heatmaps_per_token(
    attention_matrix: torch.Tensor,  # shape: [num_text_tokens, 256]
    save_dir: str,
    prefix: str = "text_token",
    yticklabels = None,
    figsize=(4, 4),
    cmap: str = "viridis"
):
    os.makedirs(save_dir, exist_ok=True)
    attention_matrix = attention_matrix.to(torch.float32)
    num_tokens = attention_matrix.shape[0]
    for i in range(num_tokens):
        attn = attention_matrix[i]  # shape: [256]
        attn_2d = attn.reshape(16, 16)  # reshape to 16x16
        plt.figure(figsize=figsize)
        plt.imshow(attn_2d.cpu().numpy(), cmap=cmap)
        plt.colorbar()
        title = f"{prefix}_{i}"
        if yticklabels is not None and i < len(yticklabels):
            title += f": {yticklabels[i]}"
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        filename = f"{prefix}_{i}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()
    print(f"Saved {num_tokens} heatmaps to {save_dir}")


def get_avg_patch_text_attention(attn, patch_len, text_mask, layer=-1):
    cls_offset = 1
    seq_offset_patch = cls_offset
    seq_offset_text = cls_offset + patch_len
    attn_layer = attn[layer][0]  # shape: (num_heads, seq_len, seq_len)
    avg_attn = attn_layer.mean(dim=0)  # shape: (seq_len, seq_len)
    text_len = text_mask.shape[0]
    patch2text = avg_attn[seq_offset_patch:seq_offset_text, seq_offset_text:seq_offset_text + text_len]
    patch2text = patch2text[:, text_mask[1:]]  # 筛选 text 列
    text2patch = avg_attn[seq_offset_text:seq_offset_text + text_len, seq_offset_patch:seq_offset_text]
    text2patch = text2patch[text_mask[1:], :]  # 筛选 text 行
    return patch2text, text2patch
