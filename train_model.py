import os
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


ASSET_DIR = "data/assets"
TARGET_SIZE = (32, 32)

ENTITY_FOLDERS = {
    "floor": "floor",
    "wall": "wall",
    "lava": "lava",
    "coin": "coin",
    "gem": "gem",
    "ghost": "ghost",
    "human": "human",
    "key": "key",
    "exit": "exit",
    "locked": "locked",
    "box": "box",
    "boots": "boots",
    "shield": "shield",
}

FEATURE_NAMES = [
    "mean", "std", "peak_ratio",
    "border_frac", "ring_border", "ring_mid", "ring_core",
    "q_tl", "q_tr", "q_bl", "q_br",
    "top_frac", "left_frac",
    "cx_norm", "cy_norm", "spread", "anisotropy",
    "row_peak", "col_peak", "row_peak_pos", "col_peak_pos",
    "sym_lr", "sym_ud", "sym_diag",
    "coverage", "strong_cov", "center_to_border",
] + [f"block_{i}" for i in range(16)]


def composite(bg, fg):
    result = bg[:, :, :3].astype(float).copy()
    alpha = fg[:, :, 3:4].astype(float) / 255.0
    result = result * (1 - alpha) + fg[:, :, :3].astype(float) * alpha
    return result.astype(np.uint8)


def shapify(gray):
    gx = np.zeros_like(gray)
    gx[1:-1, 1:-1] = (
        -gray[:-2, :-2] - 2 * gray[:-2, 1:-1] - gray[:-2, 2:]
        + gray[2:, :-2] + 2 * gray[2:, 1:-1] + gray[2:, 2:]
    )
    gy = np.zeros_like(gray)
    gy[1:-1, 1:-1] = (
        -gray[:-2, :-2] - 2 * gray[1:-1, :-2] - gray[2:, :-2]
        + gray[:-2, 2:] + 2 * gray[1:-1, 2:] + gray[2:, 2:]
    )
    return np.sqrt(gx ** 2 + gy ** 2)


def build_dataset(asset_dir=ASSET_DIR, target_size=TARGET_SIZE):
    floor_imgs = []
    floor_path = os.path.join(asset_dir, "floor")
    if os.path.isdir(floor_path):
        for f in sorted(os.listdir(floor_path))[:3]:
            if f.endswith(".png"):
                img = PILImage.open(os.path.join(floor_path, f)).convert("RGBA").resize(target_size, PILImage.LANCZOS)
                floor_imgs.append(np.array(img))

    templates = {}
    for etype, folder in ENTITY_FOLDERS.items():
        folder_path = os.path.join(asset_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        edge_imgs = []
        for f in sorted(os.listdir(folder_path)):
            if not f.endswith(".png"):
                continue

            img = PILImage.open(os.path.join(folder_path, f)).convert("RGBA").resize(target_size, PILImage.LANCZOS)
            fg = np.array(img)

            if etype in ("floor", "wall", "lava"):
                gray = np.mean(fg[:, :, :3], axis=2)
                edge_imgs.append(shapify(gray).astype(np.float32))
            else:
                if floor_imgs:
                    for fl in floor_imgs:
                        comp = composite(fl, fg)
                        gray = np.mean(comp[:, :, :3], axis=2)
                        edge_imgs.append(shapify(gray).astype(np.float32))
                else:
                    gray = np.mean(fg[:, :, :3], axis=2)
                    edge_imgs.append(shapify(gray).astype(np.float32))

        if edge_imgs:
            templates[etype] = edge_imgs

    return templates


def compute_features(edges: np.ndarray) -> np.ndarray:
    e = edges.astype(np.float64)
    h, w = e.shape
    total = e.sum() + 1e-8

    mean = e.mean()
    std = e.std()
    peak_ratio = e.max() / (mean + 1e-8)

    border_frac = (total - e[6:26, 6:26].sum()) / total
    ring_border_frac = (total - e[4:28, 4:28].sum()) / total
    ring_mid_frac = (e[4:28, 4:28].sum() - e[10:22, 10:22].sum()) / total
    ring_core_frac = e[10:22, 10:22].sum() / total

    tl = e[:16, :16].sum() / total
    tr = e[:16, 16:].sum() / total
    bl = e[16:, :16].sum() / total
    br = e[16:, 16:].sum() / total
    top = e[:16, :].sum() / total
    left = e[:, :16].sum() / total

    ys, xs = np.mgrid[0:h, 0:w]
    cx = (xs * e).sum() / total
    cy = (ys * e).sum() / total
    var_x = ((xs - cx) ** 2 * e).sum() / total
    var_y = ((ys - cy) ** 2 * e).sum() / total
    spread = np.sqrt(var_x + var_y)
    anisotropy = (var_y - var_x) / (var_y + var_x + 1e-8)

    row_sums = e.sum(1)
    col_sums = e.sum(0)
    row_peak = row_sums.max() / (row_sums.mean() + 1e-8)
    col_peak = col_sums.max() / (col_sums.mean() + 1e-8)
    row_peak_pos = row_sums.argmax() / h
    col_peak_pos = col_sums.argmax() / w

    sym_lr = 1.0 - (np.abs(e - e[:, ::-1]).sum() / (2 * total))
    sym_ud = 1.0 - (np.abs(e - e[::-1, :]).sum() / (2 * total))
    sym_diag = 1.0 - (np.abs(e - e.T).sum() / (2 * total))

    coverage = (e > mean).sum() / e.size
    strong_coverage = (e > mean * 2).sum() / e.size

    center_small = e[12:20, 12:20].mean()
    border_small = (
        e[:4, :].sum() + e[-4:, :].sum() + e[:, :4].sum() + e[:, -4:].sum()
    ) / (2 * 4 * h + 2 * 4 * (w - 8))
    center_to_border = center_small / (border_small + 1e-8)

    blocks = e.reshape(4, 8, 4, 8).mean(axis=(1, 3)).flatten()
    block_fracs = blocks / (blocks.sum() + 1e-8)

    scalar = np.array([
        mean, std, peak_ratio,
        border_frac, ring_border_frac, ring_mid_frac, ring_core_frac,
        tl, tr, bl, br,
        top, left,
        cx / w, cy / h, spread, anisotropy,
        row_peak, col_peak, row_peak_pos, col_peak_pos,
        sym_lr, sym_ud, sym_diag,
        coverage, strong_coverage, center_to_border,
    ])
    return np.concatenate([scalar, block_fracs])


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    eZ = np.exp(Z)
    return eZ / eZ.sum(axis=1, keepdims=True)


def train_logreg(X, y, n_classes, lr=0.3, n_iter=3000, l2=0.003):
    n, d = X.shape
    X1 = np.hstack([X, np.ones((n, 1))])
    W = np.zeros((d + 1, n_classes))
    Y = np.eye(n_classes)[y]
    for _ in range(n_iter):
        P = softmax(X1 @ W)
        grad = X1.T @ (P - Y) / n + l2 * W
        W -= lr * grad
    return W


def fmt1(xs, prec=4):
    return "[" + ", ".join(f"{v:+.{prec}f}" for v in xs) + "]"


def fmt2(M, prec=4):
    return "[\n" + ",\n".join("    " + fmt1(r, prec) for r in M) + "\n]"


def main():
    weights_out_path = "model_weights.py"

    T = build_dataset()
    classes = list(T.keys())

    X_all = np.array([compute_features(em) for c in classes for em in T[c]])
    y_all = np.array([ci for ci, c in enumerate(classes) for _ in T[c]])

    mu_all = X_all.mean(0)
    sd_all = X_all.std(0) + 1e-6
    Xn_all = (X_all - mu_all) / sd_all
    W_full = train_logreg(Xn_all, y_all, len(classes), l2=0.001, n_iter=3000)
    importance = np.abs(W_full[:-1]).max(axis=1)
    top25_idx = np.sort(np.argsort(-importance)[:25])

    Xk = X_all[:, top25_idx]
    mu_final = Xk.mean(0)
    sd_final = Xk.std(0) + 1e-6
    Xk_n = (Xk - mu_final) / sd_final
    W_final = train_logreg(Xk_n, y_all, len(classes), l2=0.003, n_iter=5000)

    weights_text = (
        "import numpy as np\n\n"
        f"_FEAT_MU = np.array({fmt1(mu_final)})\n"
        f"_FEAT_SD = np.array({fmt1(sd_final)})\n"
        f"_LOGREG_W = np.array({fmt2(W_final)})\n"
    )
    Path(weights_out_path).write_text(weights_text)


if __name__ == "__main__":
    main()