import numpy as np
from typing import List, Optional, Tuple

def _gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    if radius is None:
        radius = max(1, int(3.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return k

def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = _gaussian_kernel1d(sigma)
    tmp = np.pad(img, ((0, 0), (k.size // 2, k.size // 2)), mode="reflect")
    tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, tmp)
    tmp = np.pad(tmp, ((k.size // 2, k.size // 2), (0, 0)), mode="reflect")
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, tmp)
    return out

def make_grid(H: int, W: int, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)) -> np.ndarray:
    xs = np.linspace(x_range[0], x_range[1], W)
    ys = np.linspace(y_range[0], y_range[1], H)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    return coords

def images_to_prob_vectors(images: np.ndarray, invert=False, blur_sigma=0.0,
                           rescale_to: Optional[Tuple[int, int]] = None,
                           eps=1e-12) -> Tuple[List[np.ndarray], np.ndarray, Tuple[int, int]]:
    assert images.ndim == 3, "images must be (N, H, W)"
    N, H, W = images.shape
    imgs = images.astype(float)
    if imgs.max() > 1.0:
        imgs /= 255.0

    if rescale_to is not None:
        Ht, Wt = rescale_to
        out = np.zeros((N, Ht, Wt), dtype=float)
        if H % Ht == 0 and W % Wt == 0:
            fh, fw = H // Ht, W // Wt
            imgs_resh = imgs.reshape(N, Ht, fh, Wt, fw)
            out = imgs_resh.mean(axis=(2, 4))
        else:
            ys = (np.linspace(0, H - 1, Ht)).round().astype(int)
            xs = (np.linspace(0, W - 1, Wt)).round().astype(int)
            out = imgs[:, ys][:, :, xs]
        imgs = out
        H, W = Ht, Wt

    proc = np.empty_like(imgs)
    for i in range(N):
        im = imgs[i]
        if blur_sigma > 0:
            im = _gaussian_blur(im, blur_sigma)
        if invert:
            im = im.max() - im
        proc[i] = im

    measures: List[np.ndarray] = []
    for i in range(N):
        v = proc[i].clip(min=0)
        s = v.sum()
        if s < eps:
            v = np.zeros_like(v)
            v[H // 2, W // 2] = 1.0
            s = 1.0
        measures.append((v / s).reshape(-1))

    coords = make_grid(H, W, (-1.0, 1.0), (-1.0, 1.0))
    return measures, coords, (H, W)
