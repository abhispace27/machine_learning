#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 17:21:43 2026

This code reads in LTP and LTD conductance
it normalizes them and fits a curve.
Using the curve, it does machine learning on MNIST data
using ReLU and adam optimizer.

this is sk roy style work for CsPbI3 perovskite data


@author: abhinav
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import FuncFormatter


# ----------------------------
# GLOBAL FONT SETTINGS: Times New Roman
# ----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix", # Makes math symbols look like Times
    "axes.unicode_minus": False # Fixes potential minus sign issues with serif fonts
})

# Formatter to multiply axis values by 10^6
micro_formatter = FuncFormatter(lambda x, pos: f'{x*1e6:g}')

#=====================================================
# Repository-friendly path configuration.
# This preserves the original analysis logic, but makes paths independent
# of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "device_fit_mnist"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit LTP/LTD conductance curves and run the device-inspired MNIST experiment."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing the input Excel file."
    )
    parser.add_argument(
        "--fname-part",
        default="LTP_LTD_CsPbI3_perovskite",
        help="Input filename without extension."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for side-by-side comparison with old results."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the noisy conductance heatmaps and ML initialization."
    )
    return parser.parse_args()


args = parse_args()
np.random.seed(args.seed)

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1) Load 2-column Excel data
# ----------------------------
data_dir = Path(args.data_dir)
fname_part = args.fname_part
input_path = data_dir / f"{fname_part}.xlsx"

if not input_path.exists():
    raise FileNotFoundError(
        f"Missing input file:\n  {input_path}\n\n"
        "Put the Excel file in data/raw/ or pass --data-dir explicitly."
    )

df = pd.read_excel(input_path)

x = df.iloc[:, 0].to_numpy(dtype=float)  # pulse index
G = df.iloc[:, 1].to_numpy(dtype=float)  # conductance

order = np.argsort(x)
x = x[order]
G = G[order]

# ----------------------------
# 2) Split into LTP (rise) and LTD (fall) using max
# ----------------------------
imax = np.argmax(G)
# here ltp and ltd share one point
# x_ltp, G_ltp = x[:imax+1], G[:imax+1]
# x_ltd, G_ltd = x[imax:],  G[imax:]      # include peak in LTD

# NEW: peak belongs to LTP only
x_ltp, G_ltp = x[:imax+1], G[:imax+1]    # 1..35
x_ltd, G_ltd = x[imax+1:], G[imax+1:]    # 36..70

# Global extrema 
Gmin = np.min(G)
Gmax = np.max(G)



# ----------------------------
# 2b) FORCE LTD length to match LTP length by truncating extra LTD points
# ----------------------------
n_target = len(G_ltp)  # 33 in this case

if len(G_ltd) > n_target:
    x_ltd = x_ltd[:n_target]
    G_ltd = G_ltd[:n_target]
elif len(G_ltd) < n_target:
    # This is not the current case, but making it explicit
    raise ValueError(f"LTD shorter than LTP: len(G_ltd)={len(G_ltd)} < len(G_ltp)={n_target}")
    
    
# ============================================================
# FIG 6b/6c-style plots: CDF heatmaps of ΔG vs Conductance
#   - Potentiation: use LTP branch (ΔG > 0 typically)
#   - Depression  : use LTD branch (ΔG < 0 typically)
# Color encodes CDF value in [0,1]
# ============================================================

def cdf_heatmap_from_sequence(G_seq, nbins_G=40, ngrid_dG=120, dG_limits=None):
    """
    Build a CDF heatmap:
      x-axis: conductance bin (current state G)
      y-axis: ΔG grid
      value : CDF(ΔG | G_bin) = P(ΔG' <= ΔG)

    Parameters
    ----------
    G_seq : 1D array
        Conductance sequence along one programming direction.
    nbins_G : int
        Number of bins along conductance axis.
    ngrid_dG : int
        Number of points along ΔG axis.
    dG_limits : (float, float) or None
        Force ΔG range. If None, uses min/max from data with small padding.
    """
    G_seq = np.asarray(G_seq, dtype=float)
    G0 = G_seq[:-1]
    dG = G_seq[1:] - G_seq[:-1]

    # Bin by starting conductance
    G_edges = np.linspace(np.nanmin(G0), np.nanmax(G0), nbins_G + 1)
    G_centers = 0.5 * (G_edges[:-1] + G_edges[1:])

    # ΔG grid
    if dG_limits is None:
        dG_min = np.nanmin(dG)
        dG_max = np.nanmax(dG)
        pad = 0.03 * (dG_max - dG_min + 1e-12)
        dG_min -= pad
        dG_max += pad
    else:
        dG_min, dG_max = dG_limits

    dG_grid = np.linspace(dG_min, dG_max, ngrid_dG)

    # CDF matrix: shape (ngrid_dG, nbins_G)
    C = np.full((ngrid_dG, nbins_G), np.nan, dtype=float)

    for j in range(nbins_G):
        mask = (G0 >= G_edges[j]) & (G0 < G_edges[j+1])
        vals = dG[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue
        vals = np.sort(vals)
        # empirical CDF evaluated at dG_grid
        # CDF(x) = fraction of vals <= x
        C[:, j] = np.searchsorted(vals, dG_grid, side="right") / vals.size

    return G_centers, dG_grid, C


def plot_cdf_heatmap(G_centers, dG_grid, C, title, outpath,
                     xlabel=r"Conductance ($\mu$S)", ylabel=r"$\Delta G$ ($\mu$S)"):
    plt.figure(figsize=(6.2, 4.8), dpi=1200)

    # ... (existing edges calculation) ...
    G_edges = np.concatenate(([G_centers[0] - (G_centers[1]-G_centers[0])/2],
                              0.5*(G_centers[1:] + G_centers[:-1]),
                              [G_centers[-1] + (G_centers[-1]-G_centers[-2])/2]))
    dG_edges = np.concatenate(([dG_grid[0] - (dG_grid[1]-dG_grid[0])/2],
                               0.5*(dG_grid[1:] + dG_grid[:-1]),
                               [dG_grid[-1] + (dG_grid[-1]-dG_grid[-2])/2]))

    im = plt.pcolormesh(G_edges, dG_edges, C, shading="auto", vmin=0, vmax=1, cmap="RdYlBu")
    
    # # --- FORCE EQUAL ASPECT RATIO ---
    # plt.gca().set_aspect('equal', adjustable='box')
    
    # 1. Scale Main Plot Ticks by 10^6
    plt.gca().xaxis.set_major_formatter(micro_formatter)
    plt.gca().yaxis.set_major_formatter(micro_formatter)

    # 2. Setup Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("CDF", fontsize=22, fontweight="bold")
    
    # 3. MAKE COLORBAR TICKS BOLD
    cbar.ax.tick_params(labelsize=22)
    # This iterates through the labels on the colorbar axis and sets them to bold
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_family('serif') # Ensures Times New Roman is applied here too

    # 4. Standard Formatting
    # plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel(xlabel, fontsize=22, fontweight="bold")
    plt.ylabel(ylabel, fontsize=22, fontweight="bold")
    
    plt.xticks([1e-4, 2e-4, 3e-4, 4e-4], fontsize=22, fontweight="bold")
    plt.yticks(fontsize=22, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ----------------------------
# Using measured branches
# ----------------------------
# IMPORTANT: These should be the *raw* conductance values (not normalized).
# If Excel G is already in micro-S
G_ltp_raw = np.array(G_ltp, dtype=float)
G_ltd_raw = np.array(G_ltd, dtype=float)

# Potentiation heatmap (Fig 6b-style)
Gx_p, dGy_p, Cp = cdf_heatmap_from_sequence(G_ltp_raw, nbins_G=100, ngrid_dG=80)
plot_cdf_heatmap(
    Gx_p, dGy_p, Cp,
    title="Potentiation: CDF of conductance update",
    outpath=os.path.join(out_dir, f"fig6b_cdf_heatmap_potentiation_{fname_part}.png")
)

# Depression heatmap (Fig 6c-style)
Gx_d, dGy_d, Cd = cdf_heatmap_from_sequence(G_ltd_raw, nbins_G=100, ngrid_dG=80)
plot_cdf_heatmap(
    Gx_d, dGy_d, Cd,
    title="Depression: CDF of conductance update",
    outpath=os.path.join(out_dir, f"fig6c_cdf_heatmap_depression_{fname_part}.png")
)

print("Saved Fig6b/6c-style heatmaps to:", out_dir)


# ----------------------------
# 3) IMPORTANT FIX: use local pulse count per branch
#    (Pmax = max pulses used in that branch)
# ----------------------------
P_ltp_local = x_ltp - x_ltp[0]     # starts at 0
P_ltd_local = x_ltd - x_ltd[0]     # starts at 0 (peak is first point)

Pmax_ltp = P_ltp_local.max()
Pmax_ltd = P_ltd_local.max()

# ----------------------------
# 4) Saha Eq.(4)-(5) models (fit alpha), using branch-local P and Pmax
#   Eq.(4): Gp = Gmin + G0*(1 - exp(-alpha * P/Pmax))
#   Eq.(5): Gd = Gmax - G0*(1 - exp(alpha*(P/Pmax - 1)))
#   G0     = (Gmax - Gmin) / (1 - exp(-alpha))
#
# NOTE: The LTD data decreases from Gmax -> Gmin.
#       Eq.(5) increases from Gmin -> Gmax with P increasing.
#       So we fit LTD using P_rev = Pmax - P_local (reversed pulse coordinate).
# ----------------------------
eps = 1e-12

def G0_of_alpha(alpha):
    return (Gmax - Gmin) / (1.0 - np.exp(-alpha) + eps)

def GLTP_model_Saha(P_local, alpha):
    # Eq.(4)
    G0 = G0_of_alpha(alpha)
    return Gmin + G0 * (1.0 - np.exp(-alpha * (P_local / (Pmax_ltp + eps))))

def GLTD_model_Saha_decreasing(P_local, alpha):
    # Eq.(5) but evaluated on reversed coordinate so it DECREASES with P_local
    # P_rev = Pmax - P_local
    P_rev = (Pmax_ltd - P_local)
    G0 = G0_of_alpha(alpha)
    return Gmax - G0 * (1.0 - np.exp(alpha * ((P_rev / (Pmax_ltd + eps)) - 1.0)))

# Fit alpha for each branch
alpha0 = 1.0  # reasonable initial guess

(alpha_ltp_fit,), _ = curve_fit(GLTP_model_Saha, P_ltp_local, G_ltp, p0=[alpha0], bounds=(1e-9, np.inf))
(alpha_ltd_fit,), _ = curve_fit(GLTD_model_Saha_decreasing, P_ltd_local, G_ltd, p0=[alpha0], bounds=(1e-9, np.inf))

print("alpha_ltp_fit (Saha Eq4) =", float(alpha_ltp_fit))
print("alpha_ltd_fit (Saha Eq5, decreasing via P_rev) =", float(alpha_ltd_fit))


# ------------------------------------------------------------
# 1. Generate 1000-point dense pulse sequences
# ------------------------------------------------------------
n_dense = 1000
P_ltp_dense = np.linspace(0, Pmax_ltp, n_dense)
P_ltd_dense = np.linspace(0, Pmax_ltd, n_dense)

# 2. Get Y-axis (Conductance) values from the fitted models
G_ltp_smooth = GLTP_model_Saha(P_ltp_dense, alpha_ltp_fit)
G_ltd_smooth = GLTD_model_Saha_decreasing(P_ltd_dense, alpha_ltd_fit)

# 3. Create Heatmaps using the smooth model output
# By using 1000 points of the *fitted model*, the CDF will look like 
# a perfect, continuous line rather than a staircase.

# Potentiation (Smooth)
Gx_ps, dGy_ps, Cp_s = cdf_heatmap_from_sequence(G_ltp_smooth, nbins_G=20, ngrid_dG=150)
plot_cdf_heatmap(
    Gx_ps, dGy_ps, Cp_s,
    title="Potentiation (Smooth Model)",
    outpath=os.path.join(out_dir, f"fig6b_smooth_heatmap_potentiation.png")
)

# Depression (Smooth)
Gx_ds, dGy_ds, Cd_s = cdf_heatmap_from_sequence(G_ltd_smooth, nbins_G=20, ngrid_dG=150)
plot_cdf_heatmap(
    Gx_ds, dGy_ds, Cd_s,
    title="Depression (Smooth Model)",
    outpath=os.path.join(out_dir, f"fig6c_smooth_heatmap_depression.png")
)

# ------------------------------------------------------------
# ADD GAUSSIAN NOISE & PRODUCE NOISY PLOTS
# ------------------------------------------------------------
# Define noise level as 2% of the total conductance range
noise_factor = 0.02
range_G = Gmax - Gmin

# Use the smooth arrays generated earlier:
noise_ltp = np.random.normal(0, noise_factor * range_G, n_dense)
noise_ltd = np.random.normal(0, noise_factor * range_G, n_dense)

# Apply noise to the smooth (fitted) model outputs
G_ltp_noisy = G_ltp_smooth + noise_ltp
G_ltd_noisy = G_ltd_smooth + noise_ltd

# Potentiation (Noisy)
Gx_pn, dGy_pn, Cp_n = cdf_heatmap_from_sequence(G_ltp_noisy, nbins_G=20, ngrid_dG=150)
plot_cdf_heatmap(
    Gx_pn, dGy_pn, Cp_n,
    title="Potentiation (Noisy)",
    outpath=os.path.join(out_dir, f"fig6b_noisy_heatmap_potentiation_{fname_part}.png")
)

# Depression (Noisy)
Gx_dn, dGy_dn, Cd_n = cdf_heatmap_from_sequence(G_ltd_noisy, nbins_G=20, ngrid_dG=150)
plot_cdf_heatmap(
    Gx_dn, dGy_dn, Cd_n,
    title="Depression (Noisy)",
    outpath=os.path.join(out_dir, f"fig6c_noisy_heatmap_depression_{fname_part}.png")
)

# ----------------------------
# 5) Smooth fitted curves for overlay
# ----------------------------
Pgrid_ltp_local = np.linspace(P_ltp_local.min(), P_ltp_local.max(), 300)
Pgrid_ltd_local = np.linspace(P_ltd_local.min(), P_ltd_local.max(), 300)

Gfit_ltp = GLTP_model_Saha(Pgrid_ltp_local, alpha_ltp_fit)
Gfit_ltd = GLTD_model_Saha_decreasing(Pgrid_ltd_local, alpha_ltd_fit)

# ----------------------------
# 6) Fig 3c normalization (plotting convention)
#   y: global (G - Gmin)/(Gmax - Gmin)
#   x: LTP forward 0->1, LTD reversed 0->1
# ----------------------------
G_ltp_n = (G_ltp - Gmin) / (Gmax - Gmin)
G_ltd_n = (G_ltd - Gmin) / (Gmax - Gmin)

Gfit_ltp_n = (Gfit_ltp - Gmin) / (Gmax - Gmin)
Gfit_ltd_n = (Gfit_ltd - Gmin) / (Gmax - Gmin)

#-- following is for inversion for writing output------------------------------
# --- LTD inverted + endpoint-normalized (for Fig 3c-style display) ---
G_ltd_inv = 1.0 - G_ltd_n
Gfit_ltd_inv = 1.0 - Gfit_ltd_n

# force endpoints: start 0, end 1
G_ltd_inv_01 = (G_ltd_inv - G_ltd_inv[0]) / (G_ltd_inv[-1] - G_ltd_inv[0])
Gfit_ltd_inv_01 = (Gfit_ltd_inv - Gfit_ltd_inv[0]) / (Gfit_ltd_inv[-1] - Gfit_ltd_inv[0])
#-- above is for inversion ----------------------------------------------------


# x-axis normalization for plotting (unchanged idea)
p_ltp_n     = (x_ltp - x_ltp[0]) / (x_ltp[-1] - x_ltp[0])
pgrid_ltp_n = (Pgrid_ltp_local - P_ltp_local.min()) / (P_ltp_local.max() - P_ltp_local.min())

# LTD reversed for Fig 3c style
p_ltd_n     = (x_ltd[-1] - x_ltd) / (x_ltd[-1] - x_ltd[0])
pgrid_ltd_n = 1.0 - (Pgrid_ltd_local - P_ltd_local.min()) / (P_ltd_local.max() - P_ltd_local.min())


# ----------------------------
# 7) Plot like Fig 3c
# ----------------------------
plt.figure(figsize=(6, 4), dpi = 1200)

plt.scatter(p_ltp_n, G_ltp_n, s = 90, color='tab:blue', zorder=1, label="Exp LTP")
plt.scatter(p_ltd_n, G_ltd_n, s = 90, color='tab:orange', zorder=1, label="Exp LTD")

plt.plot(pgrid_ltp_n, Gfit_ltp_n, lw = 3, color='tab:pink', zorder=2, label="Fit LTP")
plt.plot(pgrid_ltd_n, Gfit_ltd_n, lw = 3, color='tab:green', zorder=2, label="Fit LTD")

# plt.scatter(p_ltp_n, G_ltp_n, s=100, label="exp LTP")
# plt.scatter(p_ltd_n, G_ltd_inv_01, s=100, label="exp LTD (inverted)")

# plt.plot(pgrid_ltp_n, Gfit_ltp_n, lw=3, label="fit LTP")
# plt.plot(pgrid_ltd_n, Gfit_ltd_inv_01, lw=3, label="fit LTD (inverted)")

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)

plt.xticks(fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')

plt.xlabel("Normalized pulse number", fontsize=15, fontweight='bold')
plt.ylabel("Normalized conductance", fontsize=15, fontweight='bold')
plt.legend(fontsize=15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(out_dir, "normalized_conductance_"+fname_part+".png"))
plt.clf()
plt.close()

# 3d plot
def scatter_3d(ax, x, y, color, s=90, label=None, zorder=3):
    # shadow
    ax.scatter(x, y, s=s*1.8, color='k', alpha=0.15, linewidth=0, zorder=zorder-1)
    # base sphere
    ax.scatter(x, y, s=s*1.2, color=color, edgecolor='k',
               linewidth=0.4, zorder=zorder, label=label)
    # highlight (offset slightly up-left)
    ax.scatter(x - 0.008, y + 0.008, s=s*0.25,
               color='white', alpha=0.8, linewidth=0, zorder=zorder+1)
    
fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

scatter_3d(ax, p_ltp_n, G_ltp_n, color='tab:blue',  s=90, zorder=1, label="Exp LTP")
scatter_3d(ax, p_ltd_n, G_ltd_n, color='tab:orange', s=90, zorder=1, label="Exp LTD")

ax.plot(pgrid_ltp_n, Gfit_ltp_n, lw=3, color='tab:pink', zorder=2, label="Fit LTP")
ax.plot(pgrid_ltd_n, Gfit_ltd_n, lw=3, color='tab:green', zorder=2, label="Fit LTD")

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

ax.set_xlabel("Normalized pulse number", fontsize=15, fontweight='bold')
ax.set_ylabel("Normalized conductance", fontsize=15, fontweight='bold')

ax.tick_params(labelsize=15, width=1.8)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight('bold')

ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "normalized_conductance_"+fname_part+"_3D.png"))
plt.close()
    
# save the raw data to file:
# ============================================================
# SAVE the EXACT plotted data from Fig 3c (raw arrays used in plot)
# ============================================================

xlsx_plotdata = os.path.join(out_dir, "relu_adam_fig3c_plotted_raw_data_"+fname_part+".xlsx")

def pad_to(arr, n):
    out = np.full(n, np.nan, dtype=float)
    out[:len(arr)] = np.asarray(arr, dtype=float)
    return out

# choose a common length so columns align
n = max(len(p_ltp_n), len(G_ltp_n), len(p_ltd_n), len(G_ltd_n),
        len(pgrid_ltp_n), len(Gfit_ltp_n), len(pgrid_ltd_n), len(Gfit_ltd_n))

df_plot = pd.DataFrame({
    # --- EXP (scatter) ---
    "p_ltp_n (x exp LTP)"     : pad_to(p_ltp_n, n),
    "G_ltp_n (y exp LTP)"     : pad_to(G_ltp_n, n),
    "p_ltd_n (x exp LTD)"     : pad_to(p_ltd_n, n),
    "G_ltd_n (y exp LTD)"     : pad_to(G_ltd_n, n),

    " " : np.nan,  # blank spacer column

    # --- FIT (lines) ---
    "pgrid_ltp_n (x fit LTP)" : pad_to(pgrid_ltp_n, n),
    "Gfit_ltp_n (y fit LTP)"  : pad_to(Gfit_ltp_n, n),
    "pgrid_ltd_n (x fit LTD)" : pad_to(pgrid_ltd_n, n),
    "Gfit_ltd_n (y fit LTD)"  : pad_to(Gfit_ltd_n, n),
})

df_plot.to_excel(xlsx_plotdata, index=False)
print("Saved plotted Fig 3c raw data to:", xlsx_plotdata)

    
# ============================================================
# SAVE normalized LTP / LTD data to Excel (Fig 3c)
# ============================================================

# ----------------------------
# Output file
# ----------------------------
os.makedirs(out_dir, exist_ok=True)
xlsx_out = os.path.join(out_dir, "relu_adam_ltp_ltd_simple_"+fname_part+".xlsx")

x_norm  = p_ltp_n
ltp_exp = G_ltp_n
ltd_exp = G_ltd_inv_01  # inverted LTD, 0 -> 1

# ----------------------------
# Fit values at same x
# ----------------------------

# LTP fit (grid already increasing)
fit_ltp = np.interp(x_norm, pgrid_ltp_n, Gfit_ltp_n)

# LTD fit: grid is reversed → sort, interpolate, then flip
idx = np.argsort(pgrid_ltd_n)
fit_ltd_tmp = np.interp(x_norm, pgrid_ltd_n[idx], Gfit_ltd_inv_01[idx])

# flip so LTD fit goes 0 → 1 along x_norm
fit_ltd = fit_ltd_tmp[::-1]

# enforce exact endpoints (numerical hygiene)
fit_ltd = (fit_ltd - fit_ltd[0]) / (fit_ltd[-1] - fit_ltd[0])

# ----------------------------
# Build table
# ----------------------------

df = pd.DataFrame({
    "Norm pulse number": x_norm,
    "Norm LTP Expt": ltp_exp,
    "Norm LTD inverted (Expt)": ltd_exp,
    " ": np.nan,
    "fit_y_ltp": fit_ltp,
    "invert of fit_y_ltd": fit_ltd
})

# ----------------------------
# Write to Excel
# ----------------------------
df.to_excel(xlsx_out, index=False)

print("Saved:", xlsx_out)


#==============================================================================
#==============================================================================

# ======================================================================
# 8) FIG 4b: MNIST ANN simulation using the fitted LTP/LTD nonlinearity
#     - Architecture: 784 -> 128 -> 10 (paper Fig 4a)
#     - Train with Adam on floating W, then clamp to device window (A2)
# ======================================================================

rng = np.random.default_rng(args.seed)
n_in, n_hid, n_out = 784, 128, 10

n_train = 60_000

# ----------------------------
# 8.1) Helper: load MNIST
# ----------------------------
def load_mnist():
    # Prefer keras if available
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except Exception:
        # Fallback: sklearn openml (slower first time)
        from sklearn.datasets import fetch_openml
        mn = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mn["data"].reshape(-1, 28, 28).astype(np.float32)
        y = mn["target"].astype(int)
        x_train, y_train = X[:n_train], y[:n_train]
        x_test,  y_test  = X[n_train:], y[n_train:]

    # normalize to [0,1] and flatten to 784
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, n_in)
    x_test  = x_test.reshape(-1, n_in)

    # labels
    y_train_oh = np.eye(n_out, dtype=np.float32)[y_train]
    y_test_oh  = np.eye(n_out, dtype=np.float32)[y_test]
    return x_train, y_train, y_train_oh, x_test, y_test, y_test_oh

x_train, y_train, y_train_oh, x_test, y_test, y_test_oh = load_mnist()

# ----------------------------
# 8.2) Working window
# ----------------------------
eps = 1e-12
Wh_min, Wh_max = 0.45, 0.55   # starting small; can widen to 0.4–0.6 later

def signed_linear_from_Wh(V, Wh, b=None):
    """
    Implements Eq.(4) from Das et al. (J. Mater. Chem. C 2023):
        WA = (2 WH - J) V
    using J = all-ones matrix so that (J V) = sum(V) broadcast to all outputs.

    V  : (B, n_in)
    Wh : (n_in, n_out)  in [0,1]
    b  : (n_out,) optional
    """
    Z = 2.0 * (V @ Wh) - V.sum(axis=1, keepdims=True) * np.ones((1, Wh.shape[1]), dtype=V.dtype)
    if b is not None:
        Z = Z + b
    return Z

# ----------------------------
# 8.3) ANN: 784 -> 128 -> 10
# ----------------------------
def init_weights():
    # algorithmic weights in [-1, 1]
    W1 = (0.02 * rng.standard_normal((n_in, n_hid))).astype(np.float32)
    W2 = (0.02 * rng.standard_normal((n_hid, n_out))).astype(np.float32)

    b1 = np.full((n_hid,), 0.01, dtype=np.float32)
    b2 = np.zeros((n_out,), dtype=np.float32)
    return W1, b1, W2, b2

def adam_update(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad * grad)
    mhat = m / (1.0 - beta1**t)
    vhat = v / (1.0 - beta2**t)
    param = param - lr * mhat / (np.sqrt(vhat) + eps)
    return param, m, v

def relu(x):
    return np.maximum(x, 0.0)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True) + eps)

def accuracy_from_logits(logits, y_true_int):
    pred = np.argmax(logits, axis=1)
    return (pred == y_true_int).mean()

def relu_grad(preact):
    return (preact > 0.0)

# ----------------------------
# 8.5) Train & reproduce Fig 4b curves (A2)
# ----------------------------
train_sizes = [2000, 4000, 6000, 8000, 10000]
epochs = 100

# Hyperparams 
batch_size = 256
adam_lr_w = 1e-3
adam_lr_b = 1e-3
beta1, beta2 = 0.9, 0.999
adam_eps = 1e-8

# Use fitted g values and Pmax from branches (kept for record)
alpha_ltp = float(alpha_ltp_fit)
alpha_ltd = float(alpha_ltd_fit)
Pmax_ltp_use = float(Pmax_ltp)
Pmax_ltd_use = float(Pmax_ltd)

plt.figure(figsize=(6,4), dpi=300)

# In-memory storage for Fig 4b
fig4b_epochs = np.arange(1, epochs + 1)
fig4b_accuracy = {}   # key = train size, value = accuracy array

cm_first_epoch = {}   # confusion matrix at epoch = 1
cm_last_epoch  = {}   # confusion matrix at final epoch

trained_models = {}  # store final weights per N

for N in train_sizes:
    W1, b1, W2, b2 = init_weights()

    # Adam state (reset per N)
    t_adam = 0
    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
    mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)

    acc_curve = np.zeros(epochs, dtype=float)

    for ep in range(epochs):
        # sample N training examples each epoch
        idx = rng.choice(x_train.shape[0], size=N, replace=False)
        Xep = x_train[idx]
        Yep = y_train_oh[idx]
        y_int = y_train[idx]

        # shuffle mini-batches
        perm = rng.permutation(N)
        Xep = Xep[perm]
        Yep = Yep[perm]
        y_int = y_int[perm]

        # minibatch loop
        for i0 in range(0, N, batch_size):
            Xb = Xep[i0:i0+batch_size]
            Yb = Yep[i0:i0+batch_size]

            # Map algorithmic W in [-1,1] -> hardware Wh in [0,1] and clamp to working window
            Wh1 = np.clip((W1 + 1.0) / 2.0, Wh_min, Wh_max).astype(np.float32)
            Wh2 = np.clip((W2 + 1.0) / 2.0, Wh_min, Wh_max).astype(np.float32)

            # forward
            Z1 = signed_linear_from_Wh(Xb, Wh1, b1)    # Eq.(4) here
            A1 = relu(Z1)
            Z2 = signed_linear_from_Wh(A1, Wh2, b2)    # Eq.(4) here
            P2 = softmax(Z2)

            B = Xb.shape[0]
            dZ2 = (P2 - Yb) / max(B, 1)

            # grads for Wh2 in Eq.(4): Z2 = 2*(A1@Wh2) - sum(A1) broadcast
            dWh2 = 2.0 * (A1.T @ dZ2)  # shape (n_hid, n_out)
            db2  = dZ2.sum(axis=0)

            # backprop to A1 through Eq.(4)
            dA1 = 2.0 * (dZ2 @ Wh2.T) - dZ2.sum(axis=1, keepdims=True) * np.ones((1, Wh2.shape[0]), dtype=dZ2.dtype)
            dZ1 = dA1 * relu_grad(Z1)

            # grads for Wh1
            dWh1 = 2.0 * (Xb.T @ dZ1)  # shape (n_in, n_hid)
            db1  = dZ1.sum(axis=0)

            # Adam step counter
            t_adam += 1

            # Adam update in W space using the computed gradients (A2)
            W2, mW2, vW2 = adam_update(W2, dWh2, mW2, vW2, t_adam, lr=adam_lr_w, beta1=beta1, beta2=beta2, eps=adam_eps)
            W1, mW1, vW1 = adam_update(W1, dWh1, mW1, vW1, t_adam, lr=adam_lr_w, beta1=beta1, beta2=beta2, eps=adam_eps)

            b2, mb2, vb2 = adam_update(b2, db2, mb2, vb2, t_adam, lr=adam_lr_b, beta1=beta1, beta2=beta2, eps=adam_eps)
            b1, mb1, vb1 = adam_update(b1, db1, mb1, vb1, t_adam, lr=adam_lr_b, beta1=beta1, beta2=beta2, eps=adam_eps)

            # project back to allowed hardware window
            Wh1 = np.clip((W1 + 1.0) / 2.0, Wh_min, Wh_max)
            Wh2 = np.clip((W2 + 1.0) / 2.0, Wh_min, Wh_max)
            W1 = (2.0 * Wh1 - 1.0).astype(np.float32)
            W2 = (2.0 * Wh2 - 1.0).astype(np.float32)

            b1 = b1.astype(np.float32)
            b2 = b2.astype(np.float32)

        # test
        Wh1t = np.clip((W1 + 1.0) / 2.0, Wh_min, Wh_max).astype(np.float32)
        Wh2t = np.clip((W2 + 1.0) / 2.0, Wh_min, Wh_max).astype(np.float32)

        Z1t = signed_linear_from_Wh(x_test, Wh1t, b1)
        A1t = relu(Z1t)
        Z2t = signed_linear_from_Wh(A1t, Wh2t, b2)

        acc = 100.0 * accuracy_from_logits(Z2t, y_test)
        acc_curve[ep] = acc
        print(f"N={N:5d} epoch={ep+1:2d}  acc={acc:5.2f}%")

        # store confusion matrices at FIRST and LAST epoch
        if ep == 0:
            y_pred = np.argmax(Z2t, axis=1)
            cm_first_epoch[N] = confusion_matrix(y_test, y_pred, labels=np.arange(10), normalize="true")

        if ep == epochs - 1:
            y_pred = np.argmax(Z2t, axis=1)
            cm_last_epoch[N] = confusion_matrix(y_test, y_pred, labels=np.arange(10), normalize="true")

    fig4b_accuracy[N] = acc_curve.copy()
    trained_models[N] = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

# Plot accuracy vs epoch
plt.figure(figsize=(6,4), dpi=1200)
markers = ['o', 's', '^', 'D', 'v']

for i, N in enumerate(train_sizes):
    plt.plot(fig4b_epochs, fig4b_accuracy[N], lw=2, marker=markers[i % len(markers)],
              markersize=10, markevery=2, label=str(N) )

plt.xlabel("Epoch", fontsize=20, fontweight = 'bold')
plt.ylabel("Accuracy %", fontsize=20, fontweight='bold')
plt.xticks(fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')
# plt.ylim(30, 100)
plt.ylim(50, 100)
plt.legend(loc='lower right', fontsize=13)
plt.tight_layout()
fname=os.path.join(out_dir, "relu_adam_accuracy_"+fname_part+".png")
plt.savefig(fname)
plt.close()
plt.clf()

# Save confusion matrices to txt
for N in train_sizes:
    fname_first = os.path.join(out_dir, f"relu_adam_confusion_matrix_epoch1_N{N}_"+fname_part+".txt")
    np.savetxt(fname_first, cm_first_epoch[N], fmt="%.6f")

    fname_last = os.path.join(out_dir, f"relu_adam_confusion_matrix_epoch{epochs}_N{N}_"+fname_part+".txt")
    np.savetxt(fname_last, cm_last_epoch[N], fmt="%.6f")

    print("Saved:", fname_first)
    print("Saved:", fname_last)

# Save confusion matrix FIGURES
def save_cm_figure(cm, fname):
    plt.figure(figsize=(6,5), dpi=1200)
    im = plt.imshow(cm, cmap="viridis", vmin=0, vmax=1)

    cbar = plt.colorbar(im)
    cbar.set_label("P(pred | true)", fontsize=15)
    cbar.ax.tick_params(labelsize=15)

    plt.xlabel("Inferred output digit", fontsize=15)
    plt.ylabel("Desired output digit", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.subplots_adjust(right=0.95, left=0.1, top=0.98, bottom=0.1)
    plt.savefig(fname)
    plt.close()

for N in train_sizes:
    save_cm_figure(cm_first_epoch[N], fname=os.path.join(out_dir, f"relu_adam_cm_epoch1_N{N}_"+fname_part+".png"))
    save_cm_figure(cm_last_epoch[N], fname=os.path.join(out_dir, f"relu_adam_cm_epoch{epochs}_N{N}_"+fname_part+".png"))

# Save accuracy vs epoch data to Excel
excel_fig4b = os.path.join(out_dir, "relu_adam_accuracy_vs_epoch_fig4b_"+fname_part+".xlsx")

with pd.ExcelWriter(excel_fig4b, engine="openpyxl") as writer:
    for N in train_sizes:
        df_acc = pd.DataFrame({
            "epoch": fig4b_epochs,
            "accuracy_percent": fig4b_accuracy[N]
        })
        sheet_name = f"N_{N}"
        df_acc.to_excel(writer, sheet_name=sheet_name, index=False)

print("Saved Fig 4b accuracy data to:", excel_fig4b)