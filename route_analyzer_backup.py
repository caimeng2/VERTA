"""
route_analysis.py — Standalone route-decision analysis on x–z trajectories

This module provides a small, dependency-light toolkit to analyze route choices
from movement trajectories on a Cartesian x–z plane. It is **generic**: no
experiment-specific names or parameters are baked in. You can:

- Load one or many "movement files" (CSV/TSV/parquet). Only coordinates are
  required; time is optional but enables timing metrics.
- Detect an initial movement direction after passing near a junction and
  cluster these directions into k branches.
- Assign branches to trajectories using learned/global centers.
- Compute simple timing metrics: time to travel a certain path length after a
  junction; time between entering two generic regions (e.g., reorientation).
- Summarize branch distribution and Shannon entropy.

The analysis is **solely based on the imported coordinates** (and optional time
column). No participant codes or metadata are required; trajectory IDs default
to the input filenames or provided keys.

Examples (CLI):

  # Discover 3 branches from CSVs in a folder and write results
  python route_analysis.py \
      --input ./data \
      --glob "*.csv" \
      --columns x=X,z=Z,t=time  \
      --junction 700 150 --radius 17.5 \
      --distance 100 \
      --k 3 \
      --out ./outputs

  # Assign branches to a second dataset using previously learned centers
  python route_analysis.py assign \
      --input ./new_data \
      --glob "*.csv" \
      --columns x=X,z=Z,t=time \
      --centers ./outputs/branch_centers.npy \
      --out ./outputs/new_assignments

"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# ------------------------------
# Data loading
# ------------------------------

@dataclass
class Trajectory:
    """Holds a single trajectory.

    Attributes
    ----------
    tid : str
        Trajectory ID (defaults to filename stem when loading from disk).
    x, z : np.ndarray
        Coordinates.
    t : Optional[np.ndarray]
        Monotonic timestamps (seconds). Optional; required only for timing metrics.
    """
    tid: str
    x: np.ndarray
    z: np.ndarray
    t: Optional[np.ndarray] = None


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    # Fallback: try CSV
    return pd.read_csv(path)

def load_folder(folder: str,
                pattern: str = "*.csv",
                columns: Dict[str, str] = None,
                require_time: bool = False,
                scale: float = 1.0,
                motion_threshold: float = 0.001
                ) -> List[Trajectory]:
    """Load movement files as Trajectory objects.

    Parameters
    ----------
    folder : str
        Folder to search.
    pattern : str
        Glob pattern (e.g., "*.csv").
    columns : dict
        Mapping specifying the column names in your files. Keys: 'x', 'z', optional 't'.
        Example: {'x': 'X', 'z': 'Z', 't': 'time'}
    require_time : bool
        If True, filters out files without a valid 't' column.
    """
    if columns is None:
        columns = {"x": "x", "z": "z", "t": "t"}

    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    out: List[Trajectory] = []

    for p in paths:
        try:
            df = _read_table(p)

            # mask rows with non-null coords (optionally including 'y' if provided)
            if columns.get("y"):
                coord_cols = [columns["x"], columns["y"], columns["z"]]
            else:
                coord_cols = [columns["x"], columns["z"]]
            mask = df[coord_cols].notnull().all(axis=1)

            # extract and scale coordinates
            x = df.loc[mask, columns["x"]].to_numpy(dtype=float) * scale
            z = df.loc[mask, columns["z"]].to_numpy(dtype=float) * scale

            # parse time if present
            t = None
            if columns.get("t") in df.columns:
                t = _to_seconds(df.loc[mask, columns["t"]])

            # if timing is required, ensure we actually have usable t
            if require_time:
                if t is None:
                    # no time column
                    continue
                if np.all(np.isnan(t)):
                    # time column exists but failed to parse
                    continue

            # trim initial static segment (first true motion), then zero time
            if len(x) > 1:
                dd = np.hypot(np.diff(x), np.diff(z))
                idx0 = int(np.argmax(dd > motion_threshold))  # 0 if none exceed
                x, z = x[idx0:], z[idx0:]
                if t is not None:
                    t = t[idx0:] - t[idx0]

            tid = os.path.splitext(os.path.basename(p))[0]
            out.append(Trajectory(tid=tid, x=x, z=z, t=t))

        except Exception as e:
            print(f"[load_folder] Skip {p}: {e}")

    return out

# ------------------------------
# Import helpers
# ------------------------------

def _to_seconds(series: pd.Series) -> np.ndarray:
    # Handles both numeric and hh:mm:ss.xxx strings
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float)
    try:
        td = pd.to_timedelta(series)  # converts '00:00:06.709' to timedelta
        return td.dt.total_seconds().to_numpy(dtype=float)
    except Exception:
        # fallback: NaN array if nothing works
        return np.full(len(series), np.nan)
    
# ------------------------------
# Geometry helpers
# ------------------------------

@dataclass
class Circle:
    cx: float
    cz: float
    r: float

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (x - self.cx) ** 2 + (z - self.cz) ** 2 <= self.r ** 2


@dataclass
class Rect:
    xmin: float
    xmax: float
    zmin: float
    zmax: float

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (x >= self.xmin) & (x <= self.xmax) & (z >= self.zmin) & (z <= self.zmax)


# ------------------------------
# Clustering
# ------------------------------

def kmeans_2d(vectors: np.ndarray, k: int = 3, max_iter: int = 100, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Tiny k-means for 2D vectors. Returns (labels, centers)."""
    if len(vectors) < k:
        raise ValueError("Not enough vectors for requested k")
    rng = np.random.default_rng(seed)
    centers = vectors[rng.choice(len(vectors), size=k, replace=False)].copy()
    labels = np.zeros(len(vectors), dtype=int)
    for _ in range(max_iter):
        # assign
        d = np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        # update
        new_centers = centers.copy()
        for j in range(k):
            pts = vectors[labels == j]
            if len(pts) > 0:
                new_centers[j] = pts.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers

import numpy as np

def best_k_by_silhouette(V: np.ndarray, k_min=2, k_max=6, seed=42):
    scores = {}
    for k in range(k_min, k_max + 1):
        if V.shape[0] <= k:
            continue
        labels, centers = kmeans_2d(V, k=k, seed=seed)
        S = _cosine_silhouette_score(V, labels)
        scores[k] = S
    if not scores:
        return min(3, max(1, V.shape[0])), {}
    best_k = max(scores, key=scores.get)
    return best_k, scores

def _cosine_silhouette_score(V: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette score using cosine distance 1 - dot(u,v) for unit vectors.
    Works without sklearn. Returns mean silhouette over all points with a valid cluster.
    """
    if V.size == 0 or labels.size != V.shape[0]:
        return 0.0
    # Precompute dot-similarity matrix (cosine on unit vectors = dot)
    D = V @ V.T  # in [-1,1]; similarity
    # Convert to distance
    dist = 1.0 - D
    n = len(V)
    s_vals = []
    for i in range(n):
        Li = labels[i]
        if Li < 0:  # noise/outlier: skip
            continue
        same = (labels == Li)
        other = (labels != Li) & (labels >= 0)

        # a(i): mean intra-cluster distance (excluding self)
        si = dist[i, same]
        if si.size <= 1:
            a = 0.0
        else:
            a = float((si.sum() - 0.0) / max(1, si.size - 1))

        # b(i): min mean distance to other clusters
        b = None
        for Lj in set(labels[other]):
            mask = (labels == Lj)
            if not mask.any():
                continue
            b_j = float(dist[i, mask].mean())
            b = b_j if b is None else min(b, b_j)
        if b is None:
            # no other clusters; silhouette undefined → 0
            s = 0.0
        else:
            s = 0.0 if (a == b == 0.0) else (b - a) / max(a, b)
        s_vals.append(s)
    return float(np.mean(s_vals)) if s_vals else 0.0

def merge_close_centers(centers: np.ndarray, labels: np.ndarray, min_sep_deg=12.0):
    if centers.shape[0] <= 1: 
        return centers, labels
    ang = np.arctan2(centers[:,1], centers[:,0])
    keep = np.ones(len(centers), dtype=bool)
    map_to = np.arange(len(centers))
    for i in range(len(centers)):
        if not keep[i]: 
            continue
        for j in range(i+1, len(centers)):
            if not keep[j]: 
                continue
            d = np.abs((ang[i]-ang[j]+np.pi)%(2*np.pi)-np.pi)
            if np.degrees(d) < min_sep_deg:
                # merge j -> i
                keep[j] = False
                map_to[map_to == j] = i
    new_ids = {old: idx for idx, old in enumerate(np.where(keep)[0])}
    new_centers = centers[keep]
    new_labels  = np.array([ new_ids[ map_to[l] ] for l in labels ], dtype=int)
    return new_centers, new_labels

import pandas as pd

def split_small_branches(assign_df: pd.DataFrame, min_frac=0.05):
    # assign_df: columns ["trajectory","branch"]
    counts = assign_df["branch"].value_counts().sort_index()
    n = int(counts.sum())
    small = set(counts[counts < max(1, int(np.ceil(min_frac*n)))].index)
    main  = assign_df[~assign_df["branch"].isin(small)].copy()
    minor = assign_df[ assign_df["branch"].isin(small)].copy()
    return main, minor, counts

def cluster_angles_dbscan(V: np.ndarray, eps_deg=15.0, min_samples=5):
    """
    Simple DBSCAN on the unit circle without sklearn.
    - Build neighbor graph by chord distance threshold derived from eps_deg.
    - Core points have >= min_samples neighbors (incl. self).
    - Expand clusters via BFS; others labeled -1.
    Returns (labels, centers[unit vectors]).
    """
    if V.size == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, 2), dtype=float)

    # Convert to angle embedding on unit circle (already unit vectors)
    X = V  # (n,2), assumed ~unit
    # chord distance threshold for angular eps:
    # chord = 2*sin(eps/2)
    eps = 2.0 * np.sin(np.deg2rad(eps_deg) / 2.0)

    # pairwise chord distances on unit circle between X[i], X[j]: ||X[i]-X[j]||
    # use (a-b)^2 = a^2 + b^2 - 2 a·b; here a^2=b^2=1 => ||a-b||^2 = 2 - 2(a·b)
    S = X @ X.T  # dot
    sq_chord = 2.0 - 2.0 * S
    sq_eps = eps * eps
    neigh = (sq_chord <= sq_eps)

    n = len(X)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    core = np.sum(neigh, axis=1) >= min_samples

    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if not core[i]:
            continue
        # start new cluster
        labels[i] = cid
        # expand via BFS over density-reachable points
        queue = [i]
        while queue:
            p = queue.pop()
            Np = np.where(neigh[p])[0]
            for q in Np:
                if not visited[q]:
                    visited[q] = True
                    if core[q]:
                        queue.append(q)
                if labels[q] == -1:
                    labels[q] = cid
        cid += 1

    # compute centers as normalized mean of unit vectors per cluster
    centers = []
    for c in range(cid):
        idx = (labels == c)
        m = X[idx].mean(axis=0)
        nrm = np.linalg.norm(m)
        centers.append(m / nrm if nrm > 0 else np.array([1.0, 0.0]))
    centers = np.array(centers) if centers else np.zeros((0, 2))
    return labels, centers


# ------------------------------
# Route-decision extraction
# ------------------------------

def first_unit_vector_after_distance(
    x: np.ndarray,
    z: np.ndarray,
    origin_region: Circle,
    path_length: float = 100.0,
    epsilon: float = 0.05,
    fallback_window: int = 10,
    linger_delta: float = 0.0
) -> Optional[np.ndarray]:
    """
    Returns a single unit direction vector capturing initial route choice.
    Fallback order:
      T1. First step >= epsilon after reaching `path_length`.
      T2. Largest single step anywhere >= epsilon.
      T3. Net displacement over last `fallback_window` steps (ignores epsilon).
    Returns None only if there is no motion at all.
    """
    if len(x) < 2 or len(z) < 2:
        return None

    min_radial = origin_region.r + max(0.0, linger_delta)

    # Start: first time inside the circle; else closest approach
    dist = np.hypot(x - origin_region.cx, z - origin_region.cz)
    inside = dist <= origin_region.r
    start = int(np.argmax(inside)) if inside.any() else int(np.argmin(dist))

    dx = np.diff(x[start:])
    dz = np.diff(z[start:])
    seg = np.hypot(dx, dz)
    if len(seg) == 0:
        return None

    cum = np.cumsum(seg)
    reach_idx = int(np.argmax(cum >= path_length)) if (cum >= path_length).any() else None

    # T1: first step >= epsilon after we reached the requested path length
    if reach_idx is not None:
        for j in range(reach_idx, len(dx)):
            if seg[j] >= epsilon:
                i = j  # step index after 'start'
                rad_now = float(np.hypot(x[start + i] - origin_region.cx,
                                        z[start + i] - origin_region.cz))
                if rad_now < min_radial:
                    continue  # NEW: keep scanning for a later, farther step
                v = np.array([dx[j], dz[j]]) / seg[j]
                return v

    # T2: largest single step anywhere
    jmax = int(np.argmax(seg))
    if seg[jmax] > 0:
        rad_at_jmax = float(np.hypot(x[start + jmax] - origin_region.cx,
                                    z[start + jmax] - origin_region.cz))
        if rad_at_jmax >= min_radial:  # NEW
            v = np.array([dx[jmax], dz[jmax]]) / seg[jmax]
            return v

    # T3: windowed net displacement (ignores epsilon threshold)
    w = min(fallback_window, len(dx))
    if w > 0:
        end_i = start + len(dx) - 1
        rad_now = float(np.hypot(x[end_i] - origin_region.cx, z[end_i] - origin_region.cz))
        if rad_now >= min_radial:  # NEW
            ddx = float(np.sum(dx[-w:]))
            ddz = float(np.sum(dz[-w:]))
            n = float(np.hypot(ddx, ddz))
            if n > 0:
                return np.array([ddx / n, ddz / n])

    return None

def first_unit_vector_after_radial_exit(
    x: np.ndarray,
    z: np.ndarray,
    junction: Circle,
    r_outer: float,
    epsilon: float = 0.05,
    window: int = 5,          # default smoothing window
) -> Optional[np.ndarray]:
    """
    Direction when the path *exits* an outer radius around the junction.
    Start at first time inside junction.r (else nearest approach).
    Trigger when r >= r_outer (with non-negative outward trend).
    Direction is the unit vector of the summed step vectors over a short window.
    """
    if len(x) < 2:
        return None

    rx = x - junction.cx
    rz = z - junction.cz
    r  = np.hypot(rx, rz)

    inside = r <= junction.r
    start = int(np.argmax(inside)) if inside.any() else int(np.argmin(r))

    # find the first index crossing r_outer with outward trend
    i_cross = None
    for i in range(start + 1, len(r)):
        if r[i] >= r_outer:
            j0 = max(start + 1, i - window)
            seg = r[j0:i+1]
            # Robust outward-trend test: if we don't have at least 2 samples,
            # accept (avoids "mean of empty slice" warning)
            if seg.size >= 2:
                outward = float(np.nanmean(np.diff(seg))) >= 0.0
            else:
                outward = True
            if outward:
                i_cross = i
                break
    if i_cross is None:
        return None

    # Smooth direction over the last `window` steps ending at i_cross
    j0 = max(start, i_cross - window)
    dx = np.diff(x[j0:i_cross+1])
    dz = np.diff(z[j0:i_cross+1])
    step = np.hypot(dx, dz)

    # Use meaningful steps if available; otherwise fall back to max step in window
    mask = step >= epsilon
    if np.any(mask):
        vx = dx[mask].sum()
        vz = dz[mask].sum()
    else:
        if step.size == 0 or float(np.nanmax(step)) <= 0:
            return None
        k = int(np.nanargmax(step))
        vx, vz = dx[k], dz[k]

    n = float(np.hypot(vx, vz))
    if n == 0:
        return None
    return np.array([vx / n, vz / n])

def _pick_vector_and_source(
    tr: Trajectory,
    junction: Circle,
    decision_mode: str,
    path_length: float,
    r_outer: Optional[float],
    epsilon: float,
    linger_delta: float = 0.0
) -> tuple[Optional[np.ndarray], str]:
    """Return (v, 'radial'|'pathlen') without changing existing APIs."""
    if decision_mode in ("radial", "hybrid"):
        rout = r_outer if (r_outer is not None and r_outer > junction.r) else (junction.r + 10.0)
        v_rad = first_unit_vector_after_radial_exit(tr.x, tr.z, junction, rout, epsilon=epsilon)
        if decision_mode == "radial":
            return v_rad, "radial"
        if v_rad is not None:
            return v_rad, "radial"
        # fallback
        v_pl = first_unit_vector_after_distance(tr.x, tr.z, junction, path_length=path_length, epsilon=epsilon, linger_delta=linger_delta)
        return v_pl, "pathlen"
    # pathlen only
    v_pl = first_unit_vector_after_distance(tr.x, tr.z, junction, path_length=path_length, epsilon=epsilon, linger_delta=linger_delta)
    return v_pl, "pathlen"

def discover_branches(trajectories: Sequence[Trajectory],
                      junction: Circle,
                      k: int = 3,
                      path_length: float = 100.0,
                      epsilon: float = 0.05,
                      seed: int = 0,
                      decision_mode="hybrid",
                      r_outer=None,
                      linger_delta: float = 0.0,
                      out_dir = None,
                      cluster_method: str = "kmeans",
                      k_min: int = 2,
                      k_max: int = 6,
                      min_sep_deg: float = 12.0,
                      angle_eps: float = 15.0,
                      min_samples: int = 5
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Compute initial movement vectors and cluster them into k branches.

    Returns
    -------
    assignments : DataFrame with columns ["trajectory", "branch"]
    summary     : DataFrame with columns ["branch", "count", "percent"]
    centers     : (k,2) ndarray of unit vectors (sorted by angle)
    """
    vecs, ids, diags, mode_log = [], [], [], []
    assign_all_rows = []  # will hold rows for −2 (no entry)

    for tr in trajectories:
        entered, _ = entered_junction_idx(tr.x, tr.z, junction)
        if not entered:
            # hard label −2, and DO NOT add to vecs/ids (excluded from clustering)
            assign_all_rows.append({"trajectory": tr.tid, "branch": -2})
            diags.append({"trajectory": tr.tid, "used": "n/a", "has_vector": False, "entered": False})
            continue

        # we did enter — extract vector as before
        v, src = _pick_vector_and_source(
            tr, junction, decision_mode, path_length, r_outer, epsilon, linger_delta=linger_delta
        )
        mode_log.append({"trajectory": tr.tid, "mode_used": src})
        diags.append({"trajectory": tr.tid, "used": src, "has_vector": v is not None, "entered": True})
        if v is not None:
            n = float(np.linalg.norm(v))
            if n > 0:
                vecs.append(v / n)
                ids.append(tr.tid)



    print(f"[discover] trajectories: {len(trajectories)}  extracted_vectors: {len(vecs)}")

    # Optional CSV diagnostics
    if out_dir is not None:
        try:
            pd.DataFrame(diags).to_csv(os.path.join(out_dir, "discover_diagnostics.csv"), index=False)
            print(f"[discover] diagnostics -> {os.path.join(out_dir, 'discover_diagnostics.csv')}")
            
            pd.DataFrame(mode_log).to_csv(os.path.join(out_dir, "decision_mode_used.csv"), index=False)
            print(f"[discover] decision_mode_used -> {os.path.join(out_dir, 'decision_mode_used.csv')}")
        except Exception as e:
            print(f"[discover] could not write diagnostics: {e}")

    if len(vecs) == 0:
        empty_assign = pd.DataFrame(columns=["trajectory", "branch"])
        empty_sum = pd.DataFrame(columns=["branch", "count", "percent"])
        return empty_assign, empty_sum, np.zeros((0, 2))

    V = np.vstack(vecs)

    if out_dir is not None and len(vecs):
        pd.DataFrame({"trajectory": ids, "vx": V[:, 0], "vz": V[:, 1]}).to_csv(os.path.join(out_dir, "vectors.csv"), index=False)

    # ---- CLUSTERING ----
    if cluster_method in ("kmeans", "auto"):
        if cluster_method == "auto":
            k_auto, sil = best_k_by_silhouette(V, k_min=k_min, k_max=k_max, seed=seed)
            print(f"[discover] auto-k silhouette -> {k_auto}  scores={sil}")
            k = k_auto
        if k > len(V):
            print(f"[discover] Requested k={k} but only {len(V)} vectors; capping.")
            k = len(V)

        labels, centers = kmeans_2d(V, k=k, seed=seed)
        # merge near-duplicate directions
        centers, labels = merge_close_centers(centers, labels, min_sep_deg=min_sep_deg)

        # angle-sort centers; remap labels to 0..C-1
        ang = np.arctan2(centers[:, 1], centers[:, 0])
        order = np.argsort(ang)
        mapping = {old: new for new, old in enumerate(order)}
        centers = centers[order]
        nrm = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.clip(nrm, 1e-12, None)
        labels = np.array([mapping[l] for l in labels], dtype=int)

    elif cluster_method == "dbscan":
        # density on angles; can yield outliers labeled -1
        lab, centers = cluster_angles_dbscan(V, eps_deg=angle_eps, min_samples=min_samples)
        labels = lab.copy()  # keep -1 for outliers

        # If no clusters found, all are outliers; centers is (0,2)
        if centers.size == 0:
            pass  # labels already -1; centers OK
        else:
            # Sort centers by angle for stable numbering
            ang = np.arctan2(centers[:, 1], centers[:, 0])
            order = np.argsort(ang)                  # order gives new IDs
            centers = centers[order]

            # Old DBSCAN cluster ids are 0..C-1 in the order they were built.
            # Build mapping: old_id -> new_id (angle-sorted)
            old_ids = np.arange(len(order))
            remap = {int(old_id): int(new_id) for new_id, old_id in enumerate(order)}

            # Remap labels >= 0; keep -1 as is
            for i, l in enumerate(labels):
                if l >= 0:
                    labels[i] = remap[int(l)]


    else:
        raise ValueError(f"Unknown cluster_method={cluster_method}")

    assignments = pd.DataFrame({"trajectory": ids, "branch": labels})
    assignments_all = pd.concat([assignments, pd.DataFrame(assign_all_rows)], ignore_index=True)

    # Summary (main branches only, >=0) stays the same using `assignments`
    mask_main = assignments["branch"] >= 0
    cnt = Counter(assignments.loc[mask_main, "branch"])
    total = int(mask_main.sum())
    summary = pd.DataFrame({
        "branch": sorted(cnt.keys()),
        "count": [cnt[b] for b in sorted(cnt.keys())],
        "percent": [cnt[b] / total * 100.0 if total else 0.0 for b in sorted(cnt.keys())],
    })

    # Write both CSVs if out_dir given and draw intercepts using the *all* table
    if out_dir is not None:
        assignments.to_csv(os.path.join(out_dir, "branch_assignments.csv"), index=False)
        assignments_all.to_csv(os.path.join(out_dir, "branch_assignments_all.csv"), index=False)
        mode_df = pd.DataFrame(mode_log)
        mode_df.to_csv(os.path.join(out_dir, "decision_mode_used.csv"), index=False)
        plot_decision_intercepts(
            trajectories, assignments_all, mode_df, centers,
            junction, r_outer, path_length, epsilon, linger_delta,
            out_path=os.path.join(out_dir, "Decision_Intercepts.png"),
            show_paths=False
        )

    return assignments, summary, centers


def assign_branches(trajectories,
                    centers: np.ndarray,
                    junction: Circle,
                    path_length: float = 100.0,
                    decision_mode="pathlen",
                    r_outer=None,
                    epsilon: float = 0.05,
                    linger_delta: float = 0.0,
                    assign_angle_eps: float = 15.0,
                    out_dir = None) -> pd.DataFrame:
    """Assign branches using fixed centers, consistent with discover."""
    ids, labs = [], []
    dbg_rows, mode_rows = [], []

    min_dot = float(math.cos(math.radians(assign_angle_eps)))

    for tr in trajectories:
        # hard −2 if we never enter the junction
        entered, _ = entered_junction_idx(tr.x, tr.z, junction)
        if not entered:
            ids.append(tr.tid); labs.append(-2)
            continue

        v, mode_used = _pick_vector_and_source(
            tr, junction, decision_mode, path_length, r_outer, epsilon, linger_delta=linger_delta
        )

        # entered but no usable vector → −1
        if v is None or centers.size == 0:
            ids.append(tr.tid); labs.append(-1)
            continue

        v = v / max(1e-12, np.linalg.norm(v))
        dots = centers @ v
        lab = int(np.argmax(dots))
        # too far from any center?  mark −1
        lab = lab if float(dots[lab]) >= min_dot else -1

        ids.append(tr.tid); labs.append(lab)
        dbg_rows.append({
            "trajectory": tr.tid, "vx": float(v[0]), "vz": float(v[1]),
            "assigned_branch": lab, "argmax_dot": float(dots[lab] if lab>=0 else np.max(dots)),
            "best_alt_branch": int(np.argsort(dots)[-2]) if len(dots) > 1 else -1,
            "best_alt_dot": float(np.sort(dots)[-2]) if len(dots) > 1 else float("nan"),
        })
        mode_rows.append({"trajectory": tr.tid, "mode_used": mode_used})

    df = pd.DataFrame({"trajectory": ids, "branch": labs})
    if out_dir is not None:
        pd.DataFrame(dbg_rows).to_csv(os.path.join(out_dir, "assign_vectors.csv"), index=False)
        pd.DataFrame(mode_rows).to_csv(os.path.join(out_dir, "assign_mode_used.csv"), index=False)
    return df




def suggest_knobs(trajectories):
    steps = []
    for tr in trajectories:
        if len(tr.x) > 1:
            s = np.hypot(np.diff(tr.x), np.diff(tr.z))
            if len(s): steps.append(np.median(s))
    if not steps: 
        print("[suggest] no steps found"); return
    med = float(np.median(steps)); p90 = float(np.percentile(steps, 90))
    print(f"[suggest] median_step≈{med:.4f}  p90_step≈{p90:.4f}  try --epsilon≈{max(med*0.5, p90*0.1):.5f}")

def intersects_junction(tr: Trajectory, junction: Circle) -> bool:
    r = np.hypot(tr.x - junction.cx, tr.z - junction.cz)
    return bool((r <= junction.r).any())

def entered_junction_idx(x: np.ndarray, z: np.ndarray, junction: Circle) -> tuple[bool, int]:
    """Return (entered, index). If entered==True: index is first inside sample.
       Otherwise: index of nearest approach (used only for plotting)."""
    r = np.hypot(x - junction.cx, z - junction.cz)
    inside = r <= junction.r
    if inside.any():
        return True, int(np.argmax(inside))
    return False, int(np.argmin(r))


# ------------------------------
# Metrics
# ------------------------------

def time_to_distance_after_junction(tr: Trajectory,
                                    junction: Circle,
                                    path_length: float) -> float:
    """Time (seconds) from first entering `junction` until cumulative traveled
    distance along the path reaches `path_length`. Returns NaN if no time data
    or cannot be computed.
    """
    if tr.t is None:
        return float("nan")
    dist = np.hypot(tr.x - junction.cx, tr.z - junction.cz)
    inside = dist <= junction.r
    if inside.any():
        start = int(np.argmax(inside))
    else:
        start = int(np.argmin(dist))
    dx = np.diff(tr.x[start:])
    dz = np.diff(tr.z[start:])
    seg = np.hypot(dx, dz)
    if len(seg) == 0:
        return float("nan")
    cum = np.cumsum(seg)
    mask = cum >= path_length
    if not mask.any():
        return float("nan")
    reach = int(np.argmax(mask))
    return float(tr.t[start + reach] - tr.t[start])

def time_from_junction_to_radial_exit(
    tr: Trajectory,
    junction: Circle,
    r_outer: float,
    window: int = 5,
    min_outward: float = 0.0,
) -> float:
    """
    Time (seconds) from first entering `junction` until the trajectory crosses r_outer
    with a non-negative (or >= min_outward) outward trend. NaN if no time or no crossing.
    """
    if tr.t is None:
        return float("nan")

    # reuse your existing index finder but with a configurable outward threshold
    r = np.hypot(tr.x - junction.cx, tr.z - junction.cz)
    inside = r <= junction.r
    start = int(np.argmax(inside)) if inside.any() else int(np.argmin(r))

    i_cross = None
    for i in range(start + 1, len(r)):
        if r[i] >= r_outer:
            j0 = max(start + 1, i - window)
            seg = r[j0:i+1]
            if seg.size < 2 or float(np.nanmean(np.diff(seg))) >= float(min_outward):
                i_cross = i
                break

    if i_cross is None:
        return float("nan")
    return float(tr.t[i_cross] - tr.t[start])

def time_between_regions(tr: Trajectory,
                         A: Rect | Circle,
                         B: Rect | Circle) -> Tuple[float, float, float]:
    """Return (t_A, t_B, dt) where t_A is the first timestamp when trajectory is in
    region A, t_B for region B, and dt = t_B - t_A. Returns (nan, nan, nan) if
    timestamps missing or sequence invalid.
    """
    if tr.t is None:
        return (float("nan"), float("nan"), float("nan"))
    inA = A.contains(tr.x, tr.z)
    inB = B.contains(tr.x, tr.z)
    iA = int(np.argmax(inA)) if inA.any() else None
    iB = int(np.argmax(inB)) if inB.any() else None
    if iA is not None and iB is not None and iB > iA:
        return float(tr.t[iA]), float(tr.t[iB]), float(tr.t[iB] - tr.t[iA])
    return (float("nan"), float("nan"), float("nan"))

def shannon_entropy(summary: pd.DataFrame) -> float:
    """Compute entropy (nats) from a branch summary with a 'percent' column."""
    if len(summary) == 0:
        return float("nan")
    p = summary["percent"].to_numpy(dtype=float) / 100.0
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))

def _timing_for_traj(
    tr: Trajectory,
    junction: Circle,
    decision_mode: str,
    distance: float,
    r_outer: float | None,
    trend_window: int,
    min_outward: float,
) -> tuple[float, str]:
    """
    Returns (time_value_seconds, mode_used).
    - pathlen: junction entry -> reach `distance` of walked path
    - radial:  junction entry -> first intercept of r_outer with outward trend
    - hybrid:  try radial first, fall back to pathlen
    """
    if decision_mode == "pathlen":
        return (
            time_to_distance_after_junction(tr, junction, path_length=float(distance)),
            "pathlen",
        )

    # radial (or hybrid attempt): need a sensible outer radius
    rout = None
    if r_outer is not None and float(r_outer) > float(junction.r):
        rout = float(r_outer)
    else:
        # soft default if user forgot to pass r_outer
        rout = float(junction.r) + 10.0

    t_rad = time_from_junction_to_radial_exit(
        tr,
        junction,
        r_outer=rout,
        window=int(trend_window),
        min_outward=float(min_outward),
    )
    if decision_mode == "radial":
        return (t_rad, "radial")

    # hybrid: radial if it worked, else pathlen
    if not (t_rad is None or (isinstance(t_rad, float) and (t_rad != t_rad))):  # not NaN
        return (t_rad, "radial")
    return (
        time_to_distance_after_junction(tr, junction, path_length=float(distance)),
        "pathlen",
    )

# ------------------------------
# Plotting
# ------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_branch_directions(centers: np.ndarray, junction_xy: tuple[float, float],
                           out_path: str, arrow_scale: float = 10.0) -> None:
    xj, zj = junction_xy
    plt.figure(figsize=(6,6))
    for i, v in enumerate(centers):
        plt.plot([xj, xj + v[0]*arrow_scale], [zj, zj + v[1]*arrow_scale])
        plt.arrow(xj, zj, v[0]*arrow_scale, v[1]*arrow_scale,
                  head_width=1.4, length_includes_head=True)
        plt.text(xj + v[0]*arrow_scale*1.02, zj + v[1]*arrow_scale*1.02,
                 f"Branch {i}", ha="left", va="center")
    plt.scatter([xj], [zj], label="Junction")
    plt.legend()
    plt.axis("equal")
    plt.title("Branch Directions")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

def plot_branch_counts(assignments: pd.DataFrame, out_path: str) -> None:
    counts = assignments["branch"].value_counts().sort_index()
    labels = [f"Branch {b}" for b in counts.index]
    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, counts.values)
    for rect, val in zip(bars, counts.values):
        plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{int(val)}",
                 ha="center", va="bottom")
    plt.ylabel("Assigned decisions")
    plt.title("Decisions per Branch")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

def _start_idx_in_junction(x: np.ndarray, z: np.ndarray, junction: Circle) -> int:
    r = np.hypot(x - junction.cx, z - junction.cz)
    inside = r <= junction.r
    return int(np.argmax(inside)) if inside.any() else int(np.argmin(r))

def _decision_index_radial(x: np.ndarray, z: np.ndarray, junction: Circle, r_outer: float,
                           window: int = 5) -> Optional[int]:
    r = np.hypot(x - junction.cx, z - junction.cz)
    start = _start_idx_in_junction(x, z, junction)
    for i in range(start + 1, len(r)):
        if r[i] >= r_outer:
            j0 = max(start + 1, i - window)
            seg = r[j0:i+1]
            if seg.size < 2 or float(np.nanmean(np.diff(seg))) >= 0.0:
                return i
    return None

def _decision_index_pathlen(x: np.ndarray, z: np.ndarray, junction: Circle, distance: float,
                            epsilon: float, linger_delta: float) -> Optional[int]:
    start = _start_idx_in_junction(x, z, junction)
    dx = np.diff(x[start:]); dz = np.diff(z[start:])
    step = np.hypot(dx, dz)
    if step.size == 0:
        return None
    cum = np.cumsum(step)
    min_rad = junction.r + max(0.0, float(linger_delta))

    # T1: first step >= epsilon after path length reached
    if (cum >= distance).any():
        k0 = int(np.argmax(cum >= distance))
        for k in range(k0, len(step)):
            if step[k] >= epsilon:
                i = start + k + 1
                rad_now = float(np.hypot(x[i] - junction.cx, z[i] - junction.cz))
                if rad_now >= min_rad:
                    return i
    # T2: fallback largest step, respecting linger guard
    kmax = int(np.argmax(step))
    if step[kmax] >= epsilon:
        i = start + kmax + 1
        rad_now = float(np.hypot(x[i] - junction.cx, z[i] - junction.cz))
        if rad_now >= min_rad:
            return i
    return None

def plot_decision_intercepts(trajectories: List[Trajectory],
                             assignments_df: pd.DataFrame,
                             mode_log_df: pd.DataFrame,
                             centers: np.ndarray,
                             junction: Circle,
                             r_outer: Optional[float],
                             path_length: float,
                             epsilon: float,
                             linger_delta: float,
                             out_path: str,
                             show_paths: bool = False,
                             plot_noenter_paths: bool = False) -> None:
    """Scatter plot of decision intercepts with branch coloring + legend by mode (stable & consistent).
       Also shows DBSCAN outliers (branch = -1) as grey squares with a legend entry."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # --- Maps: trajectory -> branch / mode ---
    tr_to_branch = {str(r["trajectory"]): int(r["branch"]) for _, r in assignments_df.iterrows()}
    tr_to_mode = {}
    if mode_log_df is not None and {"trajectory", "mode_used"} <= set(mode_log_df.columns):
        tr_to_mode = {str(r["trajectory"]): str(r["mode_used"]) for _, r in mode_log_df.iterrows()}

    # --- Stable branch colors (by branch index) ---
    branch_ids = list(range(int(centers.shape[0]) if centers is not None else 0))
    
    cmap = plt.get_cmap("tab10")
    branch_colors = {b: cmap(b % 10) for b in branch_ids}

    mode_marker = {"radial": "^", "pathlen": "o"}  # shapes by mode
    OUT_DBSCAN_COLOR = "0.5"   # -1
    OUT_NOENTER_COLOR = "black"  # -2
    MARK_DBSCAN = "s"           # square
    MARK_NOENTER = "x"          # X

    # --- Background rings ---
    theta = np.linspace(0, 2*np.pi, 512)
    cx = junction.cx + junction.r*np.cos(theta)
    cz = junction.cz + junction.r*np.sin(theta)

    plt.figure(figsize=(7.5, 7))
    ax = plt.gca()
    ax.plot(cx, cz, color="black", linewidth=1.0, label=f"junction r={junction.r:g}")
    if r_outer is not None and r_outer > junction.r:
        ox = junction.cx + float(r_outer)*np.cos(theta)
        oz = junction.cz + float(r_outer)*np.sin(theta)
        ax.plot(ox, oz, color="orange", linewidth=1.2, label=f"outer r={float(r_outer):g}")
    ax.scatter([junction.cx], [junction.cz], color="black", label="J", zorder=5)

    # --- Plot data ---

    from collections import defaultdict
    radial_by_branch = defaultdict(list)

    seen_pairs = set()     # (branch, mode) actually present
    seen_branches = set()  # main branches actually present
    saw_outliers = False

    for tr in trajectories:
        tid = str(tr.tid)
        if tid not in tr_to_branch:
            continue  # unassigned / filtered out

        br = tr_to_branch[tid]
        mode = tr_to_mode.get(tid, "pathlen")

        # br == -2  → never entered the junction
        if br == -2:
            if plot_noenter_paths and len(tr.x) > 1:
                # draw a very light grey path; no marker, no legend
                ax.plot(tr.x, tr.z, color="0.85", alpha=0.25, linewidth=1.0)
            # else: do NOTHING (no path, no legend)
            continue


        # Decide intercept index (compute BEFORE branch check so we can place outliers too)
        idx = None
        if br >= 0:
            if mode == "radial" and r_outer is not None:
                idx = _decision_index_radial(tr.x, tr.z, junction, float(r_outer))
            else:
                idx = _decision_index_pathlen(tr.x, tr.z, junction, float(path_length), float(epsilon), float(linger_delta))
        # for br == -1, keep idx as computed if available; otherwise fallback
        if idx is None:
            idx = int(np.argmin(np.hypot(tr.x - junction.cx, tr.z - junction.cz)))
            
        if br >= 0 and idx is not None and 0 <= idx < len(tr.x):
            ux = float(tr.x[idx] - junction.cx)
            uz = float(tr.z[idx] - junction.cz)
            n = math.hypot(ux, uz)
            if n > 0:
                radial_by_branch[br].append(np.array([ux/n, uz/n], dtype=float))


        # --- Outliers ---
        if br < 0:
            if show_paths and len(tr.x) > 1:
                ax.plot(tr.x, tr.z, color="0.6", alpha=0.18, linewidth=0.9)

            if br == -1:
                # entered, but off-center → gray square at the computed decision index (if any)
                if idx is not None and 0 <= idx < len(tr.x):
                    ax.plot([tr.x[idx]],[tr.z[idx]], marker="s", linestyle="None",
                            color="0.5", markersize=6, markeredgecolor="none")
                saw_outliers = True
            else:  # br == -2, no junction entry
                # mark nearest approach with black ×
                _, i_near = entered_junction_idx(tr.x, tr.z, junction)  # returned (False, i)
                if 0 <= i_near < len(tr.x):
                    ax.plot([tr.x[i_near]],[tr.z[i_near]], marker="x", linestyle="None",
                            color="k", markersize=6)
                saw_outliers = True
            continue


        # --- Main branches ---
        seen_branches.add(br)
        color = branch_colors.get(br, "0.5")

        # Optional full path (same branch color)
        if show_paths and len(tr.x) > 1:
            ax.plot(tr.x, tr.z, color=color, alpha=0.22, linewidth=1.0)

        # Intercept marker
        if idx is not None and 0 <= idx < len(tr.x):
            m = mode_marker.get(mode, "s")
            ax.plot([tr.x[idx]], [tr.z[idx]], marker=m, linestyle="None",
                    color=color, markersize=6, markeredgecolor="none")
            seen_pairs.add((br, mode))

    # --- Branch rays in matching colors (use mean radial direction of intercepts) ---
    ray_len = max(junction.r, float(r_outer) if r_outer is not None else junction.r)

    # compute mean radial direction per branch
    ray_dirs = {}
    for b, vecs in radial_by_branch.items():
        m = np.mean(np.vstack(vecs), axis=0)
        n = np.linalg.norm(m)
        if n > 0:
            ray_dirs[b] = m / n

    branches = sorted(ray_dirs.keys()) if ray_dirs else range(int(centers.shape[0]))
    for b in branches:
        v = ray_dirs.get(b, centers[b])  # fallback to centers if no intercepts collected
        col = branch_colors.get(b, "0.5")
        ax.plot([junction.cx, junction.cx + float(v[0])*ray_len],
                [junction.cz, junction.cz + float(v[1])*ray_len],
                linestyle="--", alpha=0.8, color=col)
        ax.text(junction.cx + float(v[0])*ray_len,
                junction.cz + float(v[1])*ray_len, f"B{b}", fontsize=9, color=col)


    # --- Legend (proxy handles; fixed order; exact match to what’s plotted) ---
    legend_handles, legend_labels = [], []
    for b in sorted(seen_branches):
        for m in ("radial","pathlen"):
            if (b, m) in seen_pairs:
                legend_handles.append(Line2D([0],[0], color=branch_colors[b], marker=mode_marker[m], linestyle="None"))
                legend_labels.append(f"branch {b} ({m})")

    if saw_outliers:
        legend_handles.append(Line2D([0],[0], color="0.5", marker="s", linestyle="None"))
        legend_labels.append("outlier: entered junction (−1)")

        if plot_noenter_paths:
            legend_handles.append(Line2D([0],[0], color="k", marker="x", linestyle="None"))
            legend_labels.append("outlier: no junction entry (−2)")


    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=9, loc="best")


    # --- Layout ---
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_title("Decision intercepts by branch (paths share branch color; pathlen=o, radial=^; outliers=■)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()



# ------------------------------
# I/O helpers
# ------------------------------

def save_centers(centers: np.ndarray, path: str) -> None:
    np.save(path, centers)

def save_centers_json(centers: np.ndarray, path: str) -> None:
    with open(path, "w") as f:
        json.dump(centers.tolist(), f)

def save_assignments(assignments: pd.DataFrame, path: str) -> None:
    assignments.to_csv(path, index=False)

def save_summary(summary: pd.DataFrame, path: str, with_entropy: bool = True) -> None:
    if with_entropy:
        ent = shannon_entropy(summary)
        summary = summary.copy()
        summary.loc[len(summary)] = {
            "branch": "entropy",
            "count": math.nan,
            "percent": ent,
        }
    summary.to_csv(path, index=False)

# ------------------------------
# CLI
# ------------------------------

def _parse_kv_columns(s: str) -> Dict[str, str]:
    # e.g., "x=X,z=Z,t=time"
    out = {}
    for kv in s.split(','):
        if not kv:
            continue
        k, v = kv.split('=')
        out[k.strip()] = v.strip()
    return out

def _validate_args(args, parser, *, strict: bool = False):
    # --- base checks ---
    if getattr(args, "radius", None) is not None:
        if not isinstance(args.radius, (int, float)) or args.radius <= 0:
            parser.error("--radius must be > 0")

    if getattr(args, "epsilon", None) is not None:
        if not isinstance(args.epsilon, (int, float)) or args.epsilon <= 0:
            parser.error("--epsilon must be > 0")

    if getattr(args, "k", None) is not None:
        if not isinstance(args.k, int) or args.k < 1:
            parser.error("--k must be >= 1")

    # --- r_outer for radial required
    if getattr(args, "decision_mode", None) == "radial":
        if getattr(args, "r_outer", None) is None or float(args.r_outer) <= float(getattr(args, "radius", 0.0)):
            parser.error("--decision_mode radial requires --r_outer > --radius")

    # --- r_outer checks ---
    if getattr(args, "r_outer", None) is not None:
        if args.r_outer <= 0:
            parser.error("--r_outer must be > 0")
        if args.r_outer <= getattr(args, "radius", 0):
            parser.error("--r_outer must be greater than --radius")

    # --- linger_delta vs r_outer ---
    has_ld = getattr(args, "linger_delta", None) is not None
    has_ro = getattr(args, "r_outer", None) is not None
    if has_ld and has_ro:
        min_radial = float(args.radius) + float(args.linger_delta)
        r_outer = float(args.r_outer)
        if min_radial >= r_outer:
            if strict:
                parser.error(
                    f"--linger_delta too large: radius + linger_delta = {min_radial:.3f} "
                    f"must be < r_outer = {r_outer:.3f}"
                )
            else:
                new_ld = max(0.0, r_outer - float(args.radius) - 1e-6)
                print(
                    f"[warn] radius + linger_delta ({min_radial:.3f}) >= r_outer ({r_outer:.3f}); "
                    f"setting linger_delta -> {new_ld:.6f}"
                )
                args.linger_delta = new_ld

    return args



def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Standalone route-decision analysis (x–z)")
    sub = parser.add_subparsers(dest="cmd")

    # Discover command
    p0 = sub.add_parser("discover", help="discover branches (default)")
    p0.add_argument("--input", required=True)
    p0.add_argument("--glob", default="*.csv")
    p0.add_argument("--columns", default="x=x,z=z,t=t")
    p0.add_argument("--scale", type=float, default=1.0, help="Multiply x,z by this factor (e.g., 0.2 if your old pipeline used /5)")
    p0.add_argument("--motion_threshold", type=float, default=0.001, help="Minimum step length (after scaling) to mark start of motion")
    p0.add_argument("--junction", nargs=2, type=float, required=True, metavar=("X", "Z"))
    p0.add_argument("--radius", type=float, required=True)
    p0.add_argument("--distance", type=float, default=100.0)
    p0.add_argument("--epsilon", type=float, default=0.015)
    p0.add_argument("--k", type=int, default=3)
    p0.add_argument("--decision_mode", choices=["pathlen","radial","hybrid"], default="hybrid")
    p0.add_argument("--r_outer", type=float, default=None)
    p0.add_argument("--trend_window", type=int, default=5, help="Steps used to smooth heading and check outward drift in radial mode.")
    p0.add_argument("--min_outward", type=float, default=0.0, help="Minimum average outward radial change required in the window (0 = non-negative).")
    p0.add_argument("--linger_delta", type=float, default=5.0, help="Extra radial distance beyond the junction radius required for a path-length decision. Set 0 to disable the guard.")
    p0.add_argument("--plot_intercepts", action="store_true", default=True, help="Save a scatter plot of decision intercepts colored by branch.")
    p0.add_argument("--show_paths", action="store_true", default=True, help="Overlay full trajectories in the intercept plot.")
    p0.add_argument("--outlier_frac", type=float, default=0.05, help="branches with < (outlier_frac * N) % trajectories will be marked as outlier (Default 5%).")
    p0.add_argument("--outlier_min", type=int, default=3, help="Minimum number of trajectories for a valid branch. Otherwise this will be flagged as an outlier.")
    p0.add_argument("--cluster_method", choices=["kmeans", "auto", "dbscan"], default="kmeans", help="kmeans: fixed k; auto: k picked by silhouette; dbscan: density on angles with outliers (-1).")
    p0.add_argument("--k_min", type=int, default=2, help="Auto-k: min k")
    p0.add_argument("--k_max", type=int, default=6, help="Auto-k: max k")
    p0.add_argument("--min_sep_deg", type=float, default=12.0, help="Merge centers closer than this (kmeans/auto).")
    p0.add_argument("--angle_eps", type=float, default=15.0, help="DBSCAN: neighborhood in degrees.")
    p0.add_argument("--min_samples", type=int, default=5, help="DBSCAN: minimum samples per cluster.")
    p0.add_argument("--seed", type=int, default=0)
    p0.add_argument("--plot_outliers", action="store_true", help="Include outlier (-1) trajectories in intercept plot.")
    p0.add_argument("--plot_noenter_paths", action="store_true", help="Plot -2 (no junction entry) trajectories as light grey paths; suppress × markers.")
    p0.add_argument("--include_noenter_in_assignments", action="store_true", help="Include -2 trajectories in branch_assignments.csv")
    p0.add_argument("--out", required=True)

    # Assign command
    p1 = sub.add_parser("assign", help="assign using precomputed centers")
    p1.add_argument("--input", required=True)
    p1.add_argument("--glob", default="*.csv")
    p1.add_argument("--columns", default="x=x,z=z,t=t")
    p1.add_argument("--scale", type=float, default=1.0, help="Multiply x,z by this factor (e.g., 0.2 if your old pipeline used /5)")
    p1.add_argument("--motion_threshold", type=float, default=0.001, help="Minimum step length (after scaling) to mark start of motion")
    p1.add_argument("--junction", nargs=2, type=float, required=True, metavar=("X", "Z"))
    p1.add_argument("--radius", type=float, required=True)
    p1.add_argument("--distance", type=float, default=100.0)
    p1.add_argument("--epsilon", type=float, default=0.015)
    p1.add_argument("--decision_mode", choices=["pathlen","radial","hybrid"], default="pathlen")
    p1.add_argument("--r_outer", type=float, default=None)
    p1.add_argument("--linger_delta", type=float, default=5.0, help="Extra radial distance beyond the junction radius required for a path-length decision. Set 0 to disable the guard.")
    p1.add_argument("--centers", required=True)
    p1.add_argument("--out", required=True)

    # Metrics command
    p2 = sub.add_parser("metrics", help="timing metrics")
    p2.add_argument("--input", required=True)
    p2.add_argument("--glob", default="*.csv")
    p2.add_argument("--columns", default="x=x,z=z,t=t")
    p2.add_argument("--scale", type=float, default=1.0, help="Multiply x,z by this factor (e.g., 0.2 if your old pipeline used /5)")
    p2.add_argument("--motion_threshold", type=float, default=0.001, help="Minimum step length (after scaling) to mark start of motion")
    p2.add_argument("--junction", nargs=2, type=float, required=True, metavar=("X", "Z"))
    p2.add_argument("--radius", type=float, required=True)
    p2.add_argument("--distance", type=float, default=100.0)
    p2.add_argument("--decision_mode", choices=["pathlen","radial","hybrid"], default="pathlen")
    p2.add_argument("--r_outer", type=float, default=None)
    p2.add_argument("--trend_window", type=int, default=5, help="Steps used to smooth heading and check outward drift in radial mode.")
    p2.add_argument("--min_outward", type=float, default=0.0, help="Minimum average outward radial change required in the window (0 = non-negative).")
    p2.add_argument("--linger_delta", type=float, default=5.0, help="Extra radial distance beyond the junction radius required for a path-length decision. Set 0 to disable the guard.")
    p2.add_argument("--epsilon", type=float, default=0.015)
    p2.add_argument("--regions", default=None, help="JSON with A and B regions, e.g. {\"A\":{\"rect\":[xmin,xmax,zmin,zmax]},\"B\":{\"rect\":[...]}} or circle")
    p2.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    _validate_args(args=args,parser=parser, strict=False)
    cmd = args.cmd or "discover"

    os.makedirs(args.out, exist_ok=True)
    cols = _parse_kv_columns(args.columns)
    trjs = load_folder(args.input, args.glob, columns=cols, require_time=(cmd != "discover"), scale=float(args.scale), motion_threshold=float(args.motion_threshold))

    junction = Circle(cx=float(args.junction[0]), cz=float(args.junction[1]), r=float(args.radius))

    if cmd == "discover":
        assignments, summary, centers = discover_branches(
            trjs, junction,
            k=int(args.k),
            path_length=float(args.distance),
            epsilon=float(args.epsilon),
            seed=int(args.seed),
            decision_mode=args.decision_mode,
            r_outer=args.r_outer,
            linger_delta=args.linger_delta,
            out_dir=args.out,
            cluster_method=args.cluster_method,
            k_min=int(args.k_min), k_max=int(args.k_max),
            min_sep_deg=float(args.min_sep_deg),
            angle_eps=float(args.angle_eps),
            min_samples=int(args.min_samples),
        )
        suggest_knobs(trjs)
                
        save_assignments(assignments, os.path.join(args.out, "branch_assignments_main.csv"))

        summary_all = (assignments["branch"]
                    .value_counts()
                    .sort_index()
                    .rename_axis("branch")
                    .to_frame("count"))
        summary_all["percent"] = summary_all["count"] / max(1, int(summary_all["count"].sum())) * 100.0
        save_summary(summary_all.reset_index(), os.path.join(args.out, "branch_summary_all.csv"), with_entropy=True)

        min_needed = max(int(np.ceil(float(args.outlier_frac) * len(assignments))), int(args.outlier_min))
        main_assign, minor_assign, counts = split_small_branches(assignments, min_frac=float(args.outlier_frac))

        if len(minor_assign):
            small_branches_abs = set(counts[counts < min_needed].index)
            if small_branches_abs:
                keep_mask = ~main_assign["branch"].isin(small_branches_abs)
                extra_minor = main_assign[~keep_mask]
                main_assign = main_assign[keep_mask]
                minor_assign = pd.concat([minor_assign, extra_minor], ignore_index=True)

        
        save_assignments(main_assign, os.path.join(args.out, "branch_assignments.csv"))

        if args.include_noenter_in_assignments:
            all_path = os.path.join(args.out, "branch_assignments_all.csv")
            if os.path.exists(all_path):
                df_all = pd.read_csv(all_path)
                noenter = df_all[df_all["branch"] == -2]
                combined = pd.concat([main_assign, noenter], ignore_index=True)
                save_assignments(combined, os.path.join(args.out, "branch_assignments.csv"))


        summary_main = (main_assign["branch"]
                        .value_counts()
                        .sort_index()
                        .rename_axis("branch")
                        .to_frame("count"))
        summary_main["percent"] = summary_main["count"] / max(1, int(summary_main["count"].sum())) * 100.0
        save_summary(summary_main.reset_index(), os.path.join(args.out, "branch_summary.csv"), with_entropy=True)

        
        if len(minor_assign):
            minor_assign.to_csv(os.path.join(args.out, "branch_assignments_outliers.csv"), index=False)
            print(f"[discover] flagged outlier branches: {len(minor_assign)} trajectories "
                f"(threshold = max({args.outlier_frac*100:.1f}% of N, {args.outlier_min}))")
        else:
            print("[discover] no outlier branches flagged")

        
        save_centers(centers, os.path.join(args.out, "branch_centers.npy"))
        save_centers_json(centers, os.path.join(args.out, "branch_centers.json"))

        
        plot_branch_directions(centers, (args.junction[0], args.junction[1]), os.path.join(args.out, "Branch_Directions.png"))
        plot_branch_counts(main_assign, os.path.join(args.out, "Branch_Counts.png"))  # main branches

        assign_all_path = os.path.join(args.out, "branch_assignments_all.csv")
        assign_for_plot = (pd.read_csv(assign_all_path) if args.plot_outliers else main_assign)

        if args.plot_intercepts:
            try:
                mode_log_path = os.path.join(args.out, "decision_mode_used.csv")
                mode_log_df = pd.read_csv(mode_log_path) if os.path.exists(mode_log_path) else None
                out_img = os.path.join(args.out, "Decision_Intercepts.png")
                plot_decision_intercepts(
                    trajectories=trjs,
                    assignments_df=assign_for_plot,
                    mode_log_df=mode_log_df,
                    centers=centers,
                    junction=junction,
                    r_outer=args.r_outer,
                    path_length=float(args.distance),
                    epsilon=float(args.epsilon),
                    linger_delta=float(args.linger_delta) if hasattr(args, "linger_delta") else 0.0,
                    out_path=out_img,
                    show_paths=args.show_paths,
                    plot_noenter_paths=args.plot_noenter_paths,
                )
                print(f"[discover] decision intercepts figure -> {out_img}")
            except Exception as e:
                print(f"[discover] intercept plot failed: {e}")

        print(f"Saved to {args.out}")

        with open(os.path.join(args.out, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        return

    if cmd == "assign":
        centers = np.load(args.centers)
        assignments = assign_branches(
            trjs, centers, junction, path_length=float(args.distance), epsilon=float(args.epsilon), decision_mode=args.decision_mode, r_outer=args.r_outer, linger_delta=args.linger_delta, out_dir=args.out
        )
        save_assignments(assignments, os.path.join(args.out, "branch_assignments.csv"))
        print(f"Saved to {args.out}")

        with open(os.path.join(args.out, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        return

    if cmd == "metrics":
        rows = []
        for tr in trjs:
            t_val, mode_used = _timing_for_traj(
                tr=tr,
                junction=junction,
                decision_mode=str(args.decision_mode),
                distance=float(args.distance),
                r_outer=args.r_outer,
                trend_window=int(args.trend_window),
                min_outward=float(args.min_outward),
            )

            row = {
                "trajectory": tr.tid,
                "time_value": t_val,
                "decision_mode_requested": str(args.decision_mode),
                "decision_mode_used": mode_used,   # "radial" or "pathlen"
                "distance": float(args.distance) if mode_used == "pathlen" else None,
                "r_outer": float(args.r_outer) if (mode_used == "radial" and args.r_outer is not None) else None,
                "trend_window": int(args.trend_window) if mode_used == "radial" else None,
                "min_outward": float(args.min_outward) if mode_used == "radial" else None,
            }

            # Optional A/B region timing (unchanged)
            if args.regions:
                spec = json.loads(args.regions)
                def parse_region(obj):
                    if "rect" in obj:  a,b,c,d = obj["rect"]; return Rect(a,b,c,d)
                    if "circle" in obj: a,b,r = obj["circle"]; return Circle(a,b,r)
                    raise ValueError("Unknown region spec")
                A = parse_region(spec["A"]) if "A" in spec else None
                B = parse_region(spec["B"]) if "B" in spec else None
                if A is not None and B is not None:
                    tA, tB, dt = time_between_regions(tr, A, B)
                    row.update({"t_A": tA, "t_B": tB, "dt_AB": dt})

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.out, "timing_metrics.csv"), index=False)
        print(f"Saved to {args.out}")

        with open(os.path.join(args.out, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        return




if __name__ == "__main__":
    main()
