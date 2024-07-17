import torch
import logging
from datetime import datetime
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-normalize", action="store_true", help="normalize the calculation")
args = parser.parse_args()
NORMALIZE = args.normalize


@torch.jit.script
def box_tensor(a, b):
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).view(list(a.shape)[:-1] + [-1])


logging_folder = "logs"
logging_file_name = f"scaling_behavior_{datetime.now().strftime('%d.%m.%Y_%H:%M:%S')}.log"
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler(logging_folder + "/" + logging_file_name), logging.StreamHandler()],
)


Ns = [1, 2, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
ds = [4, 8, 16, 32, 64]
H = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_samples = 2**14
inp_tokens = 2**11
for d in ds:
    logging.info(f"Testing d={d}")
    for N in Ns:
        logging.info(f"Testing d={d}, N={N}")
        B = max(inp_tokens // N, 1)
        qq_norms = []
        kk_norms = []
        kkv_norms = []
        qkqkv_norms = []
        qkv_norms = []
        v_norms = []
        y_norms = []
        out_norms = []
        for i in tqdm(range(total_samples // B + 1), total=total_samples // B + 1):
            q, k, v = torch.randn(3, B, H, N, d, device=device).unbind(0)
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            v = torch.cat([torch.ones(*v.shape[:-1], 1, device=v.device, dtype=v.dtype), v], dim=-1)
            if NORMALIZE:
                v *= 1 / N
                q *= d**0.25
                k *= d**0.25
            qq_norms.append(box_tensor(q, q).norm(2, dim=-1).mean().item())
            kk_norms.append(box_tensor(k, k).norm(2, dim=-1).mean().item())
            kv_mod = box_tensor(k, k).transpose(-1, -2) @ v
            kkv_norms.append(kv_mod.norm(2, dim=[-1, -2]).mean().item())
            qkqkv_norms.append((box_tensor(q, q) @ kv_mod).norm(2, dim=-1).mean().item())
            qkv_norms.append((q @ (k.transpose(-1, -2) @ v)).norm(2, dim=-1).mean().item())
            v_norms.append(v.sum(-2).norm(2, dim=-1).mean().item())
            y = box_tensor(q, q) @ kv_mod + 0.5 * q @ (k.transpose(-1, -2) @ v) + 0.5 * v.sum(-2).view(B, H, 1, d + 1)
            y_norm, y = y[:, :, :, :1], y[:, :, :, 1:]
            y_norms.append(y_norm.norm(2, dim=-1).mean().item())
            y = y / y_norm
            out_norms.append(y.norm(2, dim=-1).mean().item())
            del q, k, v, kv_mod, y, y_norm
        # v = v/N
        n_iters = len(qq_norms)

        # logging.info(f"shapes: q,k: {q.shape}; v: {v.shape}")
        logging.info(f"\t Q^2: {sum(qq_norms) / n_iters}")
        logging.info(f"\t K^2: {sum(kk_norms) / n_iters}")
        logging.info(f"\tK^2V: {sum(kkv_norms) / n_iters}")
        logging.info(f"\t(QK^T)^2V: {sum(qkqkv_norms) / n_iters}")
        logging.info(f"\tQKV: {sum(qkv_norms) / n_iters}")
        logging.info(f"\tV: {sum(v_norms) / n_iters}")
        logging.info(f"\tY normalization: {sum(y_norms) / n_iters}")
        logging.info(f"\tOutput: {sum(out_norms) / n_iters}")
        logging.info(
            f"(d: {d}, N: {N}): {{ 'Q^2': {sum(qq_norms) / n_iters}, 'K^2': {sum(kk_norms) / n_iters}, 'K^2V': {sum(kkv_norms) / n_iters}, "
            f"'(QK^T)^2V': {sum(qkqkv_norms) / n_iters}, 'QKV': {sum(qkv_norms) / n_iters}, 'V': {sum(v_norms) / n_iters}, "
            f"'Y normalization': {sum(y_norms) / n_iters}, 'Output': {sum(out_norms) / n_iters} }}"
        )
