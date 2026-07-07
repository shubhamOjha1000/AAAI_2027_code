"""Grid geometry: gaze -> cell, fovea region, cell distances.

The global thumbnail is a GRID x GRID (8x8 = 64) token grid. Cell index is
row-major: g_idx = gy*GRID + gx, with gx = col, gy = row.
"""
import numpy as np

import config as C


def gaze_to_cell(x_norm, y_norm, grid=C.GRID):
    """Normalized gaze (x_norm,y_norm) in [0,1] -> (gx, gy, g_idx)."""
    gx = min(int(x_norm * grid), grid - 1)
    gy = min(int(y_norm * grid), grid - 1)
    return gx, gy, gy * grid + gx


def cell_rowcol(idx, grid=C.GRID):
    return idx // grid, idx % grid          # (row=gy, col=gx)


def fovea_region(g_idx, ring=C.FOVEA_RING, grid=C.GRID):
    """The gaze cell + its `ring`-neighborhood (Chebyshev). Excluded from FRM.

    ring=1 -> the 3x3 block around the gaze cell (fovea = what Stage 2a already
    keeps at high res). Returns a sorted list of cell indices.
    """
    gy, gx = g_idx // grid, g_idx % grid
    out = []
    for dy in range(-ring, ring + 1):
        for dx in range(-ring, ring + 1):
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < grid and 0 <= nx < grid:
                out.append(ny * grid + nx)
    return sorted(set(out))


def candidate_mask(g_idx, ring=C.FOVEA_RING, grid=C.GRID):
    """Boolean [K] mask: True where the cell is a FRM candidate (fovea excluded)."""
    m = np.ones(grid * grid, dtype=bool)
    m[fovea_region(g_idx, ring, grid)] = False
    return m


def cell_distance_to_gaze(g_idx, grid=C.GRID):
    """Euclidean distance (in grid cells) from every cell to the gaze cell. [K]."""
    gy, gx = g_idx // grid, g_idx % grid
    ys, xs = np.divmod(np.arange(grid * grid), grid)
    return np.sqrt((ys - gy) ** 2 + (xs - gx) ** 2)


def importance_centroid_distance(imp, g_idx, cand_mask=None, grid=C.GRID):
    """Distance from gaze cell to the importance-weighted centroid of context.

    Used to decide the FAR-context stratum: if the answer's importance mass sits
    far from gaze, this sample is where FRM must beat eccentricity.
    """
    imp = np.asarray(imp, dtype=float).copy()
    if cand_mask is not None:
        imp = imp * cand_mask
    if imp.sum() <= 0:
        return 0.0
    ys, xs = np.divmod(np.arange(grid * grid), grid)
    cy = (imp * ys).sum() / imp.sum()
    cx = (imp * xs).sum() / imp.sum()
    gy, gx = g_idx // grid, g_idx % grid
    return float(np.sqrt((cy - gy) ** 2 + (cx - gx) ** 2))
