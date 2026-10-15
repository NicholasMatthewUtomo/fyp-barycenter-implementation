
# Convert a sparse barycenter (N x 3 array of [x, y, mass]) into a raster image on a target grid.
# Supports: nearest binning, bilinear splatting, and Gaussian splatting.
# Coordinates are assumed to lie in a rectangular domain, default [-1,1] x [-1,1].
# Output can be returned as float [0,1] or converted to MNIST-like uint8 (28x28 by default).

from typing import Literal, Optional, Tuple
import numpy as np

Mode = Literal["nearest", "bilinear", "gaussian"]

def _ensure_array(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Input must be an (N,3) array: columns are x, y, mass.")
    return a

def _grid_axes(H: int, W: int, x_range: Tuple[float,float], y_range: Tuple[float,float]) -> Tuple[np.ndarray,np.ndarray]:
    xs = np.linspace(x_range[0], x_range[1], W)
    ys = np.linspace(y_range[0], y_range[1], H)
    return xs, ys

def scatter_to_grid(
    xyw: np.ndarray,
    H: int,
    W: int,
    x_range: Tuple[float,float] = (-1.0, 1.0),
    y_range: Tuple[float,float] = (-1.0, 1.0),
    mode: Mode = "bilinear",
    gaussian_sigma: float = 0.0,   # in grid *pixels*; e.g., 0.8 ~ subpixel blur
    clip_outside: bool = True,
    renorm: bool = True,
    eps: float = 1e-18,
) -> np.ndarray:
    """
    Rasterize an (N,3) array [x, y, mass] to a (H,W) image aligned to a uniform grid over
    [x_range] x [y_range].

    Modes:
      - 'nearest': deposit full mass to nearest pixel center.
      - 'bilinear': split mass among the 4 neighboring pixels (triangle/hat kernel).
      - 'gaussian': spread mass with an isotropic Gaussian in pixel coordinates (uses sigma).

    If clip_outside=True, points outside the domain are ignored (or clipped for 'bilinear').
    If renorm=True, rescales so the sum of the image equals total mass and then scales to [0,1] by max.
    """
    P = _ensure_array(xyw)
    xs, ys = _grid_axes(H, W, x_range, y_range)
    # pixel size
    dx = (x_range[1] - x_range[0]) / max(W - 1, 1)
    dy = (y_range[1] - y_range[0]) / max(H - 1, 1)

    img = np.zeros((H, W), dtype=float)

    if mode == "nearest":
        # map to nearest indices
        ix = np.rint((P[:,0] - x_range[0]) / dx).astype(int)
        iy = np.rint((P[:,1] - y_range[0]) / dy).astype(int)
        if clip_outside:
            mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            ix, iy, m = ix[mask], iy[mask], P[mask,2]
        else:
            ix = np.clip(ix, 0, W-1); iy = np.clip(iy, 0, H-1); m = P[:,2]
        for xidx, yidx, w in zip(ix, iy, m):
            img[yidx, xidx] += w

    elif mode == "bilinear":
        # continuous pixel coords
        fx = (P[:,0] - x_range[0]) / dx
        fy = (P[:,1] - y_range[0]) / dy
        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        wx = fx - x0
        wy = fy - y0
        # neighbors: (x0,y0), (x0+1,y0), (x0,y0+1), (x0+1,y0+1)
        for i in range(P.shape[0]):
            m = P[i,2]
            for dx_i, wx_i in ((0, 1-wx[i]), (1, wx[i])):
                xi = x0[i] + dx_i
                if clip_outside:
                    if xi < 0 or xi >= W: 
                        continue
                else:
                    xi = min(max(xi, 0), W-1)
                for dy_i, wy_i in ((0, 1-wy[i]), (1, wy[i])):
                    yi = y0[i] + dy_i
                    if clip_outside:
                        if yi < 0 or yi >= H:
                            continue
                    else:
                        yi = min(max(yi, 0), H-1)
                    img[yi, xi] += m * (wx_i * wy_i)

    elif mode == "gaussian":
        if gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be > 0 for mode='gaussian'.")
        # convert xy to pixel space
        px = (P[:,0] - x_range[0]) / dx
        py = (P[:,1] - y_range[0]) / dy
        for i in range(P.shape[0]):
            cx, cy, w = px[i], py[i], P[i,2]
            # 3-sigma box around center in pixel coords
            r = int(np.ceil(3.0 * gaussian_sigma))
            x0 = int(np.floor(cx)) - r; x1 = int(np.floor(cx)) + r
            y0 = int(np.floor(cy)) - r; y1 = int(np.floor(cy)) + r
            if clip_outside:
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(W-1, x1); y1 = min(H-1, y1)
            else:
                x0 = max(-r, x0); y0 = max(-r, y0)
                x1 = min(W-1+r, x1); y1 = min(H-1+r, y1)
            if x0 > x1 or y0 > y1: 
                continue
            xs_idx = np.arange(x0, x1+1)
            ys_idx = np.arange(y0, y1+1)
            XX, YY = np.meshgrid(xs_idx, ys_idx)
            kern = np.exp(-((XX - cx)**2 + (YY - cy)**2) / (2*gaussian_sigma**2))
            s = kern.sum()
            if s > 0:
                kern = kern / s * w
                img[y0:y1+1, x0:x1+1] += kern
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize to keep total mass; then scale to [0,1] by max if renorm is True.
    total_in = P[:,2].sum()
    total_out = img.sum()
    if total_out > eps and total_in > eps:
        img *= (total_in / total_out)
    if renorm:
        m = img.max()
        if m > eps:
            img = img / m
    return img

def to_mnist_uint8(
    xyw: np.ndarray,
    target_hw: Tuple[int,int] = (28,28),
    x_range: Tuple[float,float] = (-1.0, 1.0),
    y_range: Tuple[float,float] = (-1.0, 1.0),
    mode: Mode = "bilinear",
    gaussian_sigma: float = 0.8,
    invert: bool = False,) -> np.ndarray:
    """
    \"\"\"Convenience wrapper: returns a uint8 image ready for MNIST-style viewing/saving.\"\"\"
    """
    img = scatter_to_grid(
        xyw, H=target_hw[0], W=target_hw[1],
        x_range=x_range, y_range=y_range,
        mode=mode, gaussian_sigma=gaussian_sigma,
        renorm=True
    )
    img = np.clip(img, 0, 1)
    if invert:
        img = 1.0 - img
    return (img * 255.0 + 0.5).astype(np.uint8)
