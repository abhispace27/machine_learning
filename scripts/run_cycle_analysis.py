#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------
# GLOBAL SETTINGS
# ----------------------------
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix", "axes.unicode_minus": False})
micro_formatter = FuncFormatter(lambda x, pos: f'{x*1e6:g}')

# Repository-friendly path configuration.
# This preserves the original analysis logic, but makes paths independent
# of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "cycle_analysis"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze repeated LTP/LTD conductance cycles and reproduce CDF heatmaps."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing the input CSV file."
    )
    parser.add_argument(
        "--fname-part",
        default="S_13_P_D_C_final_50_cycle",
        help="Input filename without extension."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for side-by-side comparison with old results."
    )
    return parser.parse_args()


args = parse_args()

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1) Load and Process Data
# ----------------------------
data_dir = Path(args.data_dir)
fname_part = args.fname_part
input_path = data_dir / f"{fname_part}.csv"

if not input_path.exists():
    raise FileNotFoundError(
        f"Missing input file:\n  {input_path}\n\n"
        "Put the CSV file in data/raw/ or pass --data-dir explicitly."
    )

df = pd.read_csv(input_path)

# Create 2D arrays
num_repeats = df['Repeat'].nunique()
points_per_repeat = len(df) // num_repeats
voltage_2d = df['CH1 Voltage'].to_numpy().reshape((num_repeats, points_per_repeat))
current_2d = df['CH1 Current'].to_numpy().reshape((num_repeats, points_per_repeat))
# conductance_2d = current_2d / voltage_2d

# One average voltage per repeat (no eps, no masking)
V_avg_per_repeat = np.mean(voltage_2d, axis=1)   # shape (num_repeats,)

# Broadcast to all time points in that repeat
conductance_2d = current_2d / V_avg_per_repeat[:, None]

# ----------------------------
# 2) Define 2D Heatmap Logic
# ----------------------------
def cdf_heatmap_from_2d(G_2d, nbins_G=100, ngrid_dG=80):
    G_min, G_max = np.nanmin(G_2d), np.nanmax(G_2d)
    G_edges = np.linspace(G_min, G_max, nbins_G + 1)
    G_centers = 0.5 * (G_edges[:-1] + G_edges[1:])
    
    all_dG = np.diff(G_2d, axis=1).flatten()
    dG_grid = np.linspace(np.nanmin(all_dG), np.nanmax(all_dG), ngrid_dG)
    
    C_counts = np.zeros((ngrid_dG, nbins_G))
    total_in_bin = np.zeros(nbins_G)
    
    for row in G_2d:
        G0, dG = row[:-1], np.diff(row)
        for j in range(nbins_G):
            mask = (G0 >= G_edges[j]) & (G0 < G_edges[j+1])
            vals = dG[mask & np.isfinite(dG)]
            if vals.size > 0:
                for val in vals:
                    idx = np.searchsorted(dG_grid, val)
                    if idx < ngrid_dG:
                        C_counts[idx:, j] += 1
                    total_in_bin[j] += 1
    return G_centers, dG_grid, np.divide(C_counts, total_in_bin, where=total_in_bin != 0)

def plot_cdf_heatmap(G_centers, dG_grid, C, title, outpath, key, 
                     xlabel=r"Conductance ($\mu$S)", ylabel=r"$\Delta G$ ($\mu$S)"):
    plt.figure(figsize=(6.2, 4.8), dpi=1200)

    # 1. Setup mesh edges for pcolormesh
    G_edges = np.concatenate(([G_centers[0] - (G_centers[1]-G_centers[0])/2],
                              0.5*(G_centers[1:] + G_centers[:-1]),
                              [G_centers[-1] + (G_centers[-1]-G_centers[-2])/2]))
    dG_edges = np.concatenate(([dG_grid[0] - (dG_grid[1]-dG_grid[0])/2],
                               0.5*(dG_grid[1:] + dG_grid[:-1]),
                               [dG_grid[-1] + (dG_grid[-1]-dG_grid[-2])/2]))

    # 2. Plot heatmap
    im = plt.pcolormesh(G_edges, dG_edges, C, shading="auto", vmin=0, vmax=1, cmap="RdYlBu")
    
    # 3. Axis formatting
    ax = plt.gca()
    ax.xaxis.set_major_formatter(micro_formatter)
    ax.yaxis.set_major_formatter(micro_formatter)

    # 4. Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("CDF", fontsize=22, fontweight="bold")
    cbar.ax.tick_params(labelsize=22)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_family('serif')

    # 5. Labels and Ticks
    plt.xlabel(xlabel, fontsize=22, fontweight="bold")
    plt.ylabel(ylabel, fontsize=22, fontweight="bold")
    plt.xticks(fontsize=22, fontweight="bold")
    plt.yticks(fontsize=22, fontweight="bold")
    
    if key == "ltp":
        plt.ylim([0,np.max(dG_grid)])
        plt.yticks([0e-6, 10e-6, 20e-6, 30e-6], fontsize=22, fontweight="bold")
        plt.xticks([1e-4, 2e-4, 3e-4, 4e-4], fontsize=22, fontweight="bold")

    if key == "ltd":
        print(np.max(dG_grid))
        # plt.ylim([np.min(dG_grid), 0])
        plt.ylim([np.min(dG_grid), np.max(dG_grid)])
        # plt.yticks([0e-6, 10e-6, 20e-6, 30e-6], fontsize=22, fontweight="bold")
        plt.xticks([1e-4, 2e-4, 3e-4, 4e-4], fontsize=22, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    
    
def cdf_heatmap_from_2d_interp(
    G_2d,
    nbins_G=120,          # number of G grid points (bin centers)
    ngrid_dG=120,         # number of dG grid points (vertical axis)
    extrapolate=False,    # True: linear extrap; False: NaN outside data range
):
    """
    Implements the 'binning + linear interpolation' approach:
    1) For each repeat (row), compute (G0_k, dG_k).
    2) Interpolate dG(G0) onto a common G_grid.
    3) For each G_grid column, compute empirical CDF of dG across repeats.

    Returns:
        G_grid (centers), dG_grid, C (CDF heatmap)
    """

    # ---- 1) Build common G grid (centers) from all available G0 samples
    G0_all = G_2d[:, :-1].ravel()
    dG_all = np.diff(G_2d, axis=1).ravel()
    m_all = np.isfinite(G0_all) & np.isfinite(dG_all)
    if not np.any(m_all):
        raise ValueError("No finite (G0, dG) samples found.")

    G_min, G_max = np.min(G0_all[m_all]), np.max(G0_all[m_all])
    G_grid = np.linspace(G_min, G_max, nbins_G)

    # We'll also define dG_grid from all dG values (finite)
    dG_min, dG_max = np.min(dG_all[m_all]), np.max(dG_all[m_all])
    dG_grid = np.linspace(dG_min, dG_max, ngrid_dG)

    # ---- 2) Interpolate each row's dG onto G_grid
    nrep = G_2d.shape[0]
    dG_on_grid = np.full((nrep, nbins_G), np.nan, dtype=float)

    for r in range(nrep):
        row = G_2d[r, :]
        G0 = row[:-1]
        dG = np.diff(row)

        m = np.isfinite(G0) & np.isfinite(dG)
        G0 = G0[m]
        dG = dG[m]
        if G0.size < 2:
            continue

        # sort by G0 for interpolation
        order = np.argsort(G0)
        G0s = G0[order]
        dGs = dG[order]

        # drop duplicate G0s (np.interp needs increasing x)
        # keep last occurrence
        uniq_G0, uniq_idx = np.unique(G0s, return_index=True)
        # np.unique returns first index;
        # but first index is fine usually; for safety we can average duplicates instead
        if uniq_G0.size < G0s.size:
            # average duplicates
            # group by G0
            dG_avg = np.zeros_like(uniq_G0, dtype=float)
            counts = np.zeros_like(uniq_G0, dtype=int)
            # use a dict-ish approach via searchsorted
            inv = np.searchsorted(uniq_G0, G0s)
            for ii, v in zip(inv, dGs):
                dG_avg[ii] += v
                counts[ii] += 1
            dG_avg = dG_avg / np.maximum(counts, 1)
            G0s, dGs = uniq_G0, dG_avg
        else:
            G0s, dGs = uniq_G0, dGs[uniq_idx]

        if G0s.size < 2:
            continue

        # if extrapolate:
        #     # linear extrapolation on both ends
        #     # left extrap slope
        #     left_slope = (dGs[1] - dGs[0]) / (G0s[1] - G0s[0])
        #     right_slope = (dGs[-1] - dGs[-2]) / (G0s[-1] - G0s[-2])

        #     dGi = np.interp(G_grid, G0s, dGs)  # inside range
        #     left_mask = G_grid < G0s[0]
        #     right_mask = G_grid > G0s[-1]
        #     dGi[left_mask] = dGs[0] + left_slope * (G_grid[left_mask] - G0s[0])
        #     dGi[right_mask] = dGs[-1] + right_slope * (G_grid[right_mask] - G0s[-1])
        # else:
        #     # no extrapolation: NaN outside range
        #     dGi = np.interp(G_grid, G0s, dGs, left=np.nan, right=np.nan)

        if extrapolate:
            # Nearest/flat extrapolation:
            # outside each repeat's measured G-range, hold the end value constant
            dGi = np.interp(G_grid, G0s, dGs)   # normal linear interp inside range
            dGi[G_grid < G0s[0]]  = dGs[0]      # flat left
            dGi[G_grid > G0s[-1]] = dGs[-1]     # flat right
        else:
            # no extrapolation: NaN outside range (ignored in CDF)
            dGi = np.interp(G_grid, G0s, dGs, left=np.nan, right=np.nan)            
            
            

        dG_on_grid[r, :] = dGi

    # ---- 3) Compute CDF at each G_grid column over repeats
    C = np.full((ngrid_dG, nbins_G), np.nan, dtype=float)
    for j in range(nbins_G):
        vals = dG_on_grid[:, j]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        vals_sorted = np.sort(vals)
        # empirical CDF: C(y) = fraction of vals <= y
        # for each y in dG_grid: idx = searchsorted(vals_sorted, y, side='right')
        idxs = np.searchsorted(vals_sorted, dG_grid, side="right")
        C[:, j] = idxs / vals_sorted.size

    return G_grid, dG_grid, C

# ----------------------------
# 3) Generate Heatmaps
# ----------------------------
# Calculate the mean across repeats to find the representative peak
mean_conductance = np.mean(conductance_2d, axis=0)
imax = np.argmax(mean_conductance)

# 1. Create the specific slices
G_ltp_2d = conductance_2d[:, :imax+1]
G_ltd_2d = conductance_2d[:, imax:]

# 2. Use the slices in the heatmap functions
# Potentiation (Only LTP data)
# Gx_p, dGy_p, Cp = cdf_heatmap_from_2d(G_ltp_2d, nbins_G=100, ngrid_dG=100)
# plot_cdf_heatmap(Gx_p, dGy_p, Cp, "Potentiation", os.path.join(out_dir, f"heatmap_ltp_{fname_part}.png"))
# # Depression (Only LTD data)
# Gx_d, dGy_d, Cd = cdf_heatmap_from_2d(G_ltd_2d, nbins_G=100, ngrid_dG=100)
# plot_cdf_heatmap(Gx_d, dGy_d, Cd, "Depression", os.path.join(out_dir, f"heatmap_ltd_{fname_part}.png"))

Gx_p, dGy_p, Cp = cdf_heatmap_from_2d_interp(G_ltp_2d, nbins_G=120, ngrid_dG=120, extrapolate=True)
plot_cdf_heatmap(Gx_p, dGy_p, Cp, "Potentiation", os.path.join(out_dir, f"heatmap_ltp_{fname_part}.png"), "ltp")

Gx_d, dGy_d, Cd = cdf_heatmap_from_2d_interp(G_ltd_2d, nbins_G=120, ngrid_dG=120, extrapolate=True)
plot_cdf_heatmap(Gx_d, dGy_d, Cd, "Depression", os.path.join(out_dir, f"heatmap_ltd_{fname_part}.png"), "ltd")



#==============================================================================

from scipy.interpolate import RegularGridInterpolator

def cdf_heatmap_no_binning_linear_interp(G_2d, nG=120, ndG=120, extrapolate=False):
    """
    No histogram/bin counting.

    1) For each repeat r: compute pairs (G_k, dG_k), then build a function dG_r(G) by *linear interpolation*.
    2) Evaluate dG_r(G) on a common G_grid -> dG_on_grid[r, j]
    3) For each G_grid[j], compute empirical CDF of dG across repeats on dG_grid.

    Returns: G_grid, dG_grid, C  where C[i,j] = P(dG <= dG_grid[i] | G = G_grid[j])
    """

    # ---- collect global finite ranges for grids
    G0_all = G_2d[:, :-1].ravel()
    dG_all = np.diff(G_2d, axis=1).ravel()
    m_all = np.isfinite(G0_all) & np.isfinite(dG_all)
    if not np.any(m_all):
        raise ValueError("No finite (G0, dG) samples found.")

    G_min, G_max = np.min(G0_all[m_all]), np.max(G0_all[m_all])
    dG_min, dG_max = np.min(dG_all[m_all]), np.max(dG_all[m_all])

    G_grid  = np.linspace(G_min, G_max, nG)
    dG_grid = np.linspace(dG_min, dG_max, ndG)

    # ---- interpolate each repeat: dG_r(G) onto common G_grid
    nrep = G_2d.shape[0]
    dG_on_grid = np.full((nrep, nG), np.nan, dtype=float)

    for r in range(nrep):
        row = G_2d[r, :]
        G0 = row[:-1]
        dG = np.diff(row)

        m = np.isfinite(G0) & np.isfinite(dG)
        G0 = G0[m]
        dG = dG[m]
        if G0.size < 2:
            continue

        # sort by G0 for interpolation
        order = np.argsort(G0)
        G0s = G0[order]
        dGs = dG[order]

        # ensure strictly increasing x for np.interp:
        # average duplicates in G0
        uniq, inv = np.unique(G0s, return_inverse=True)
        if uniq.size < G0s.size:
            dG_avg = np.zeros_like(uniq, dtype=float)
            cnt = np.zeros_like(uniq, dtype=int)
            for ii, v in zip(inv, dGs):
                dG_avg[ii] += v
                cnt[ii] += 1
            dG_avg /= np.maximum(cnt, 1)
            G0s, dGs = uniq, dG_avg

        if G0s.size < 2:
            continue

        # if extrapolate:
        #     # linear extrapolation using end slopes
        #     left_slope  = (dGs[1]  - dGs[0])  / (G0s[1]  - G0s[0])
        #     right_slope = (dGs[-1] - dGs[-2]) / (G0s[-1] - G0s[-2])

        #     dGi = np.interp(G_grid, G0s, dGs)  # inside range
        #     left_mask  = G_grid < G0s[0]
        #     right_mask = G_grid > G0s[-1]
        #     dGi[left_mask]  = dGs[0]  + left_slope  * (G_grid[left_mask]  - G0s[0])
        #     dGi[right_mask] = dGs[-1] + right_slope * (G_grid[right_mask] - G0s[-1])
        # else:
        #     # outside measured G-range for that repeat -> NaN (ignored in stats)
        #     dGi = np.interp(G_grid, G0s, dGs, left=np.nan, right=np.nan)
                        
        if extrapolate:
            # Nearest/flat extrapolation:
            # outside each repeat's measured G-range, hold the end value constant
            dGi = np.interp(G_grid, G0s, dGs)   # normal linear interp inside range
            dGi[G_grid < G0s[0]]  = dGs[0]      # flat left
            dGi[G_grid > G0s[-1]] = dGs[-1]     # flat right
        else:
            # no extrapolation: NaN outside range (ignored in CDF)
            dGi = np.interp(G_grid, G0s, dGs, left=np.nan, right=np.nan)            
            

        dG_on_grid[r, :] = dGi

    # ---- empirical CDF across repeats, per G_grid
    C = np.full((ndG, nG), np.nan, dtype=float)
    for j in range(nG):
        vals = dG_on_grid[:, j]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vals_sorted = np.sort(vals)
        idxs = np.searchsorted(vals_sorted, dG_grid, side="right")
        C[:, j] = idxs / vals_sorted.size

    return G_grid, dG_grid, C


def upsample_C_linear(G_grid, dG_grid, C, upG=4, updG=4):
    """
    Pure linear interpolation on the already-computed C(G,dG) grid.
    This is for *plotting smoothness* only.
    """
    # new finer grids
    G_f  = np.linspace(G_grid[0],  G_grid[-1],  len(G_grid)*upG)
    dG_f = np.linspace(dG_grid[0], dG_grid[-1], len(dG_grid)*updG)

    interp = RegularGridInterpolator(
        (dG_grid, G_grid), C,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    DG, GG = np.meshgrid(dG_f, G_f, indexing="ij")
    pts = np.column_stack([DG.ravel(), GG.ravel()])
    C_f = interp(pts).reshape(len(dG_f), len(G_f))

    return G_f, dG_f, C_f


def plot_heatmap_imshow(G_grid, dG_grid, C, outpath, micro_formatter, key, 
                        xlabel=r"Conductance ($\mu$S)", ylabel=r"$\Delta G$ ($\mu$S)"):
    plt.figure(figsize=(6.2, 4.8), dpi=1200)
    ax = plt.gca()

    extent = [G_grid[0], G_grid[-1], dG_grid[0], dG_grid[-1]]
    im = ax.imshow(
        C, origin="lower", aspect="auto", extent=extent,
        vmin=0, vmax=1, cmap="RdYlBu",
        interpolation="none"  # important: WE already upsampled linearly
    )

    ax.xaxis.set_major_formatter(micro_formatter)
    ax.yaxis.set_major_formatter(micro_formatter)

    cbar = plt.colorbar(im)
    # Colorbar label
    cbar.set_label("CDF", fontsize=26, fontweight="bold")
    
    # Control tick fontsize (and optionally fontweight/family)
    cbar.ax.tick_params(labelsize=25)
    
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_family("serif")    
    
    # cbar.set_label("CDF", fontsize=22, fontweight="bold")

    plt.xlabel(xlabel, fontsize=26, fontweight="bold")
    plt.ylabel(ylabel, fontsize=26, fontweight="bold")
    plt.xticks(fontsize=26, fontweight="bold")
    plt.yticks(fontsize=26, fontweight="bold")
    
    # if key == "ltp":
    #     plt.ylim([0,8e-6])
    # if key == "ltd":
    #     plt.ylim([-10e-6,0])

    if key == "ltp":
        plt.ylim([0,np.max(dG_grid)])
        plt.yticks([0e-6, 10e-6, 20e-6, 30e-6], fontsize=26, fontweight="bold")
        plt.xticks([1e-4, 2e-4, 3e-4, 4e-4], fontsize=26, fontweight="bold")

    if key == "ltd":
        print(np.min(dG_grid))
        # plt.ylim([np.min(dG_grid), 0])
        plt.ylim([np.min(dG_grid), np.max(dG_grid)])
        plt.yticks([-5e-5, -3e-5, -1e-5], fontsize=26, fontweight="bold")
        plt.xticks([1e-4, 2e-4, 3e-4, 4e-4], fontsize=26, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    
    
Gg, dGg, C = cdf_heatmap_no_binning_linear_interp(G_ltp_2d, nG=140, ndG=140, extrapolate=True)
Gf, dGf, Cf = upsample_C_linear(Gg, dGg, C, upG=5, updG=5)
plot_heatmap_imshow(Gf, dGf, Cf, os.path.join(out_dir, f"heatmap_interpolated_ltp_{fname_part}.png"), micro_formatter, "ltp")

Gg, dGg, C = cdf_heatmap_no_binning_linear_interp(G_ltd_2d, nG=140, ndG=140, extrapolate=True)
Gf, dGf, Cf = upsample_C_linear(Gg, dGg, C, upG=5, updG=5)
plot_heatmap_imshow(Gf, dGf, Cf, os.path.join(out_dir, f"heatmap_interpolated_ltd_{fname_part}.png"), micro_formatter, "ltd")
