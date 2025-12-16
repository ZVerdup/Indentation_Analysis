#!/usr/bin/env python
"""
Indentation test analysis for rock hardness / instrumented indentation.

- Reads one or more CSV files with columns like:
    'Time (sec)', 'Ch:Load (lbs)', 'Ch:Position (mm)', 'Ch:Stress (PSI)', ...

- Does:
    * Basic cleaning & unit conversions.
    * Segments test into loading / (optional) hold / unloading.
    * Builds P vs h curves using mm / SI units internally.
    * Fits Hertz-type P ~ h^(3/2) on loading branch to get a slope.
    * Estimates a "damage onset" point based on deviation from Hertz fit.
    * Fits an unloading stiffness on the unloading branch.
    * Estimates mean contact pressure using Hertz contact area approximation.
    * Writes a summary CSV across all tests.
    * Saves basic QC plots per test.

All *physics* calculations use N and mm/m; "standard" units (lbf, inch, psi)
are only used for reporting and convenience.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- Config & constants ----------------------

# Indenter geometry
BALL_DIAM_MM = 1.0  # mm
BALL_RADIUS_MM = BALL_DIAM_MM / 2.0
BALL_RADIUS_M = BALL_RADIUS_MM / 1000.0  # meters

# Thresholds
CONTACT_LOAD_LBF = 0.05  # above this we consider "contact" established
MIN_POINTS_FOR_FIT = 15  # minimum #points for any regression fit

# Loading Hertz fit region as fraction of peak load
HERTZ_MIN_FRAC = 0.10  # e.g. 10% of peak to avoid noise
HERTZ_MAX_FRAC = 0.40  # up to ~40% of peak to stay in elastic regime

# Damage onset deviation threshold (fractional)
DAMAGE_DEV_FRAC = 0.10  # 10% deviation from Hertz fit

# Unloading fit region as fraction of peak load
UNLOAD_MAX_FRAC = 0.98
UNLOAD_MIN_FRAC = 0.90

# Creep / dwell detection
CREEP_SLOPE_THRESH_LBF_PER_S = 0.1   # |dP/dt| below this -> "approximately constant load"
CREEP_MIN_DURATION_S = 19.9           # minimum dwell time in creep
UNLOAD_SLOPE_THRESH_LBF_PER_S = 0.05  # strongly negative dP/dt -> unloading

# Simple hold detection (for creep)
# HOLD_SLOPE_THRESH_LBF_PER_S = 0.02  # |dP/dt| below this = "approx constant load"
# MIN_HOLD_DURATION_S = 5.0           # require at least this much time in hold


# ---------------------- Helpers ----------------------

def read_indent_csv(path: Path) -> pd.DataFrame:
    """Read raw indentation CSV and standardize column names.

    Expected columns (from your frame):
        'Time (sec)', 'Ch:Load (lbs)', 'Ch:Position (mm)', 'Ch:Stress (PSI)',
        'Ch:Strain (mm/mm)', etc.

    Returns a DataFrame with new columns:
        time_s, load_lbf, disp_mm, disp_in, load_N, stress_psi, strain

    NOTE: Machine logs compression as negative. We flip signs so that:
        compression load  -> positive
        indentation depth -> positive
    """
    df = pd.read_csv(path)

    # Basic presence checks
    required = ['Time (sec)', 'Ch:Load (lbs)', 'Ch:Position (mm)']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in file: {path}")

    g = pd.DataFrame()
    g["time_s"] = pd.to_numeric(df["Time (sec)"], errors="coerce")

    # Raw machine values
    load_raw = pd.to_numeric(df["Ch:Load (lbs)"], errors="coerce")
    disp_raw = pd.to_numeric(df["Ch:Position (mm)"], errors="coerce")

    # Store raw for debugging
    g["load_lbf_raw"] = load_raw
    g["disp_mm_raw"] = disp_raw

    # Flip sign so compression is positive, indentation depth positive
    g["load_lbf"] = -load_raw
    g["disp_mm"] = -disp_raw

    # Optional / nice-to-have
    if "Ch:Stress (PSI)" in df.columns:
        g["stress_psi"] = pd.to_numeric(df["Ch:Stress (PSI)"], errors="coerce")
    else:
        g["stress_psi"] = np.nan

    if "Ch:Strain (mm/mm)" in df.columns:
        g["strain"] = pd.to_numeric(df["Ch:Strain (mm/mm)"], errors="coerce")
    else:
        g["strain"] = np.nan

    # Derived columns from *sign-corrected* displacement and load
    g["disp_in"] = g["disp_mm"] / 25.4
    g["load_N"] = g["load_lbf"] * 4.44822

    # Sort by time just in case
    g = g.sort_values("time_s").reset_index(drop=True)

    return g


def estimate_contact_zero(g: pd.DataFrame, contact_load_lbf: float) -> float:
    """Estimate 'zero' displacement (mm) as the mean displacement below contact_load_lbf.

    This sets the origin for indentation depth h.
    """
    pre_contact = g[g["load_lbf"].abs() < contact_load_lbf]
    if len(pre_contact) < 5:
        # Fallback: use minimum displacement as "zero"
        return float(g["disp_mm"].min())
    return float(pre_contact["disp_mm"].mean())


def segment_loading(g: pd.DataFrame, contact_load_lbf: float) -> pd.DataFrame:
    """Add 'segment' column: 'pre', 'loading', 'creep', 'unloading', 'post'.

    Logic:
      - pre: before first point above contact_load_lbf
      - loading: from contact to start of creep
      - creep: near-constant load after a small drop, for at least CREEP_MIN_DURATION_S
      - unloading: after creep, when load starts to decrease significantly
      - post: after load has fallen back below contact_load_lbf
    """
    g = g.copy()
    g["segment"] = "pre"

    load = g["load_lbf"].values
    time = g["time_s"].values

    # Contact index
    above_contact = np.where(np.abs(load) >= contact_load_lbf)[0]
    if len(above_contact) == 0:
        # No meaningful loading
        return g

    i_contact = int(above_contact[0])

    # Peak load index (usually the overshoot)
    i_peak = int(np.argmax(load))

    # Default segmentation if we fail to detect creep cleanly
    # (pre, loading, unloading, post)
    dPdt = np.gradient(load, time, edge_order=2)

    # Find where load finally drops below contact again (for "post")
    below_again = np.where(np.abs(load[i_peak:]) < contact_load_lbf)[0]
    if len(below_again) > 0:
        i_unload_end_default = i_peak + int(below_again[0])
    else:
        i_unload_end_default = len(g) - 1

    # ---- Detect creep start and unload start ----
    i_creep_start = None
    i_unload_start = None

    # Start looking just after peak
    for k in range(i_peak + 1, len(g)):
        # If we haven't found creep yet, look for near-constant load
        if i_creep_start is None:
            if abs(dPdt[k]) < CREEP_SLOPE_THRESH_LBF_PER_S:
                # Require that we're actually near the top (not late noise)
                i_creep_start = k
        else:
            # Already in creep zone; look for unloading onset
            t_in_creep = time[k] - time[i_creep_start]
            if (
                t_in_creep >= CREEP_MIN_DURATION_S
                and dPdt[k] < -UNLOAD_SLOPE_THRESH_LBF_PER_S
            ):
                i_unload_start = k
                break

    # If we never detect creep, fall back to simple loading/unloading segmentation
    if i_creep_start is None:
        g.loc[i_contact:i_peak, "segment"] = "loading"
        g.loc[i_peak:i_unload_end_default, "segment"] = "unloading"
        if i_unload_end_default < len(g) - 1:
            g.loc[i_unload_end_default + 1 :, "segment"] = "post"
        return g

    # If we detect creep but not unloading, treat everything after creep as creep
    if i_unload_start is None:
        i_unload_start = len(g)  # no unloading segment

    # Now assign stages
    g.loc[i_contact:i_creep_start - 1, "segment"] = "loading"
    g.loc[i_creep_start:i_unload_start - 1, "segment"] = "creep"

    # Unloading (if any) until we drop below contact again
    if i_unload_start < len(g):
        # Find where load drops below contact after i_unload_start
        below_after_unload = np.where(np.abs(load[i_unload_start:]) < contact_load_lbf)[0]
        if len(below_after_unload) > 0:
            i_unload_end = i_unload_start + int(below_after_unload[0])
        else:
            i_unload_end = len(g) - 1

        g.loc[i_unload_start:i_unload_end, "segment"] = "unloading"
        if i_unload_end < len(g) - 1:
            g.loc[i_unload_end + 1 :, "segment"] = "post"

    return g


def fit_linear(x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    """Simple linear fit y = a * x + b with numpy.polyfit.

    Returns dict with keys a, b, r2 or None if not enough points.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]

    if x_fit.size < MIN_POINTS_FOR_FIT:
        return None

    # polyfit returns [slope, intercept]
    a, b = np.polyfit(x_fit, y_fit, 1)
    y_pred = a * x_fit + b
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"a": float(a), "b": float(b), "r2": r2}


def hertz_loading_fit(g: pd.DataFrame, out_dict: Dict[str, Any]) -> None:
    """Fit P vs h^(3/2) on loading branch to approximate Hertz behavior.

    Populates out_dict with:
        hertz_slope_N_per_m32
        hertz_intercept_N
        hertz_r2
    """
    load_N = g["load_N"].values
    h_mm = g["h_mm"].values
    seg = g["segment"].values

    # Only loading branch
    m_load = (seg == "loading")
    if not np.any(m_load):
        return

    peak_load = np.nanmax(load_N[m_load])
    if not np.isfinite(peak_load) or peak_load <= 0:
        return

    # Base range by load fraction
    m_range = (
        m_load
        & (load_N >= HERTZ_MIN_FRAC * peak_load)
        & (load_N <= HERTZ_MAX_FRAC * peak_load)
    )

    # Require positive, finite h and load
    m_valid = (
        m_range
        & np.isfinite(h_mm)
        & np.isfinite(load_N)
        & (h_mm > 0.0)
    )

    if not np.any(m_valid):
        return

    h_m = h_mm[m_valid] / 1000.0  # mm -> m
    x = np.power(h_m, 1.5)        # h^(3/2) in m^(3/2)
    y = load_N[m_valid]

    fit = fit_linear(x, y)
    if fit is None:
        return

    out_dict["hertz_slope_N_per_m32"] = fit["a"]
    out_dict["hertz_intercept_N"] = fit["b"]
    out_dict["hertz_r2"] = fit["r2"]


def find_damage_onset(g: pd.DataFrame, summary: Dict[str, Any]) -> None:
    a = summary.get("hertz_slope_N_per_m32", None)
    b = summary.get("hertz_intercept_N", None)
    if a is None or b is None:
        return

    h_mm = g["h_mm"].values
    h_m = h_mm / 1000.0
    load_N = g["load_N"].values
    seg = g["segment"].values
    time_s = g["time_s"].values

    # Only consider loading branch
    m = (seg == "loading")
    if not np.any(m):
        return

    h_m_loading = h_m[m]
    load_N_loading = load_N[m]
    time_loading = time_s[m]

    # Only use positive, finite h for the power
    valid = np.isfinite(h_m_loading) & np.isfinite(load_N_loading) & (h_m_loading > 0.0)
    if not np.any(valid):
        return

    h_m_loading = h_m_loading[valid]
    load_N_loading = load_N_loading[valid]
    time_loading = time_loading[valid]

    x = np.power(h_m_loading, 1.5)
    P_fit = a * x + b
    mask = P_fit > 0

    if not np.any(mask):
        return

    dev = np.abs(load_N_loading[mask] - P_fit[mask]) / P_fit[mask]

    # Find first where deviation exceeds threshold
    idxs = np.where(dev > DAMAGE_DEV_FRAC)[0]
    if idxs.size == 0:
        return

    j = idxs[0]
    t_onset = time_loading[mask][j]
    P_onset_N = load_N_loading[mask][j]
    h_onset_m = h_m_loading[mask][j]
    h_onset_mm = h_onset_m * 1000.0

    # Hertz mean contact pressure p_m ≈ P / (π R h)
    p_mean_Pa = float(P_onset_N / (math.pi * BALL_RADIUS_M * h_onset_m)) if h_onset_m > 0 else float("nan")
    p_mean_MPa = p_mean_Pa / 1e6

    summary["damage_onset_time_s"] = float(t_onset)
    summary["damage_onset_load_lbf"] = float(P_onset_N / 4.44822)
    summary["damage_onset_h_mm"] = float(h_onset_mm)
    summary["damage_onset_mean_p_MPa"] = float(p_mean_MPa)


def unloading_stiffness(g: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """Fit linear unloading stiffness on unloading branch.

    Uses subset where load is between UNLOAD_MIN_FRAC and UNLOAD_MAX_FRAC of peak load.

    Populates summary with:
        unload_stiffness_N_per_mm
        unload_r2
    """
    seg = g["segment"].values
    load_N = g["load_N"].values
    h_mm = g["h_mm"].values

    m_unload = (seg == "unloading")
    if not np.any(m_unload):
        return

    peak_load = np.max(g["load_N"].values)
    if peak_load <= 0:
        return

    m_range = (
        m_unload
        & (load_N >= UNLOAD_MIN_FRAC * peak_load)
        & (load_N <= UNLOAD_MAX_FRAC * peak_load)
    )

    x = h_mm[m_range]
    y = load_N[m_range]

    fit = fit_linear(x, y)
    if fit is None:
        return

    summary["unload_stiffness_N_per_mm"] = fit["a"]
    summary["unload_r2"] = fit["r2"]


def analyze_hold_creep(g: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """Analyze displacement vs time during 'creep' segment (dwell at peak load).

    Populates:
        hold_duration_s
        hold_disp_change_um
        hold_creep_rate_um_per_s (simple linear over entire creep dwell)
    """
    m_hold = (g["segment"].values == "creep")
    if not np.any(m_hold):
        return

    h_mm = g.loc[m_hold, "h_mm"].values
    t_s = g.loc[m_hold, "time_s"].values

    if t_s.size < 5:
        return

    duration = float(t_s[-1] - t_s[0])
    if duration <= 0:
        return

    dh_mm = float(h_mm[-1] - h_mm[0])
    dh_um = dh_mm * 1000.0
    creep_rate_um_per_s = dh_um / duration

    summary["hold_duration_s"] = duration
    summary["hold_disp_change_um"] = dh_um
    summary["hold_creep_rate_um_per_s"] = creep_rate_um_per_s


def summary_for_test(g: pd.DataFrame, file_name: str) -> Dict[str, Any]:
    """Compute summary metrics for a single indentation test."""

    out: Dict[str, Any] = {
        "file": file_name,
    }

    # Peak load / displacement / stress
    idx_peak = int(g["load_lbf"].idxmax())
    out["peak_time_s"] = float(g.loc[idx_peak, "time_s"])
    out["peak_load_lbf"] = float(g.loc[idx_peak, "load_lbf"])
    out["peak_load_N"] = float(g.loc[idx_peak, "load_N"])
    out["peak_disp_mm"] = float(g.loc[idx_peak, "h_mm"])
    out["peak_disp_in"] = float(g.loc[idx_peak, "h_mm"] / 25.4)
    out["peak_stress_psi"] = float(g.loc[idx_peak, "stress_psi"]) if "stress_psi" in g else float("nan")

    # Mean contact pressure at peak (Hertz)
    h_peak_m = g.loc[idx_peak, "h_mm"] / 1000.0
    if h_peak_m > 0:
        p_peak_Pa = out["peak_load_N"] / (math.pi * BALL_RADIUS_M * h_peak_m)
        out["peak_mean_p_MPa"] = float(p_peak_Pa / 1e6)
    else:
        out["peak_mean_p_MPa"] = float("nan")

    # Hertz loading slope
    hertz_loading_fit(g, out)

    # Damage onset
    find_damage_onset(g, out)

    # Unloading stiffness
    unloading_stiffness(g, out)

    # Hold creep
    analyze_hold_creep(g, out)

    return out


def make_plots(g: pd.DataFrame, out_png_prefix: Path) -> None:
    """Generate basic QC plots for a single test.

    Creates:
      * load vs time
      * disp vs time
      * load vs disp (P–h curve)
      * load vs h^(3/2) (Hertz-style)
    Highlights the unloading-stiffness range on the load vs time and load vs h plots.
    """
    # Basic info
    seg_values = g["segment"].unique()
    peak_load_N = g["load_N"].max()

    # Mask for unloading stiffness region
    m_unload_stiff = (
        (g["segment"] == "unloading")
        & (g["load_N"] >= UNLOAD_MIN_FRAC * peak_load_N)
        & (g["load_N"] <= UNLOAD_MAX_FRAC * peak_load_N)
    )

    # 1) Load vs time
    plt.figure()
    for seg in seg_values:
        m = g["segment"] == seg
        plt.plot(g.loc[m, "time_s"], g.loc[m, "load_lbf"],
                 ".", label=seg, markersize=2)

    if m_unload_stiff.any():
        plt.plot(
            g.loc[m_unload_stiff, "time_s"],
            g.loc[m_unload_stiff, "load_lbf"],
            "-",
            linewidth=2,
            color="cyan",               # high-contrast stiffness range
            label="unload stiffness range",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Load (lbf)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_prefix.with_name(out_png_prefix.name + "_load_vs_time.png"), dpi=300)
    plt.close()

    # 2) Displacement vs time
    plt.figure()
    for seg in seg_values:
        m = g["segment"] == seg
        plt.plot(g.loc[m, "time_s"], g.loc[m, "h_mm"],
                 ".", label=seg, markersize=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Indentation depth h (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_prefix.with_name(out_png_prefix.name + "_disp_vs_time.png"), dpi=300)
    plt.close()

    # 3) Load vs depth (P–h)
    plt.figure()
    for seg in seg_values:
        m = g["segment"] == seg
        plt.plot(g.loc[m, "h_mm"], g.loc[m, "load_lbf"],
                 ".", label=seg, markersize=2)

    if m_unload_stiff.any():
        plt.plot(
            g.loc[m_unload_stiff, "h_mm"],
            g.loc[m_unload_stiff, "load_lbf"],
            "-",
            linewidth=2,
            color="cyan",               # high-contrast stiffness range
            label="unload stiffness range",
        )

    plt.xlabel("Indentation depth h (mm)")
    plt.ylabel("Load (lbf)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_prefix.with_name(out_png_prefix.name + "_load_vs_h.png"), dpi=300)
    plt.close()

    # 4) Load vs h^(3/2) (Hertz)
    # Clip h to >= 0 to avoid invalid power for tiny negative noise
    h_mm_all = g["h_mm"].values
    h_m = np.maximum(h_mm_all, 0.0) / 1000.0
    x = np.power(h_m, 1.5)

    plt.figure()
    for seg in seg_values:
        m = g["segment"] == seg
        plt.plot(x[m], g.loc[m, "load_N"],
                 ".", label=seg, markersize=2)

    if m_unload_stiff.any():
        plt.plot(
            x[m_unload_stiff],
            g.loc[m_unload_stiff, "load_N"],
            "-",
            linewidth=2,
            color="cyan",               # high-contrast stiffness range
            label="unload stiffness range",
        )

    plt.xlabel("h^(3/2) (m^(3/2))")
    plt.ylabel("Load (N)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_prefix.with_name(out_png_prefix.name + "_load_vs_h32.png"), dpi=300)
    plt.close()



def trim_pre_contact(g: pd.DataFrame, contact_load_lbf: float) -> pd.DataFrame:
    """Keep only the last ~20 points before contact and everything after.

    'Contact' is defined as the first index where |load_lbf| >= contact_load_lbf.
    All earlier rows are dropped, except for the last 20 before that index.
    """
    load = g["load_lbf"].values
    idxs = np.where(np.abs(load) >= contact_load_lbf)[0]
    if idxs.size == 0:
        # Never reaches contact; just return as-is
        return g

    i_contact = int(idxs[0])
    i_keep_start = max(0, i_contact - 20)

    g_trim = g.iloc[i_keep_start:].reset_index(drop=True)
    return g_trim


def process_file(path: Path, out_dir: Path) -> Dict[str, Any]:
    """Process a single CSV file and return summary dict."""
    df_raw = read_indent_csv(path)

    # Trim pre-contact data, keeping only the last 20 points before contact
    df_raw = trim_pre_contact(df_raw, CONTACT_LOAD_LBF)

    # First, segment based on load/time (no h_mm needed yet)
    df_seg = segment_loading(df_raw, CONTACT_LOAD_LBF)

    # # Compute h0 as the disp_mm at the last row of the "pre" segment
    # pre_mask = df_seg["segment"] == "pre"
    # if pre_mask.any():
    #     # last "pre" row
    #     h0 = float(df_seg.loc[pre_mask, "disp_mm"].iloc[-1])
    # else:
    #     # fallback: use first row's disp_mm if, for some reason, no "pre" exists
    #     h0 = float(df_seg["disp_mm"].iloc[0])

    # Compute h0 as the mean of the first 3 disp_mm of the "loading" segment
    load_mask = df_seg["segment"] == "loading"
    if load_mask.any():
        disp_loading = df_seg.loc[load_mask, "disp_mm"]
        # take the first 3 loading points (or fewer if there aren't 3)
        h0 = float(disp_loading.iloc[:3].mean())
    else:
        # fallback: use first row's disp_mm if, for some reason, no "loading" exists
        h0 = float(df_seg["disp_mm"].iloc[0])

    print(f"{path.name}: h0 = {h0:.6f} mm (last 'pre' disp)")

    # Now define indentation depth relative to that h0
    df_seg["h_mm"] = df_seg["disp_mm"] - h0

    # Summary metrics
    summary = summary_for_test(df_seg, file_name=path.name)

    # Add h0 to summary for sanity check
    summary["h0_mm"] = h0

    # Plots
    png_prefix = out_dir / path.stem
    make_plots(df_seg, png_prefix)

    processed_path = out_dir / f"{path.stem}_processed.csv"
    df_seg.to_csv(processed_path, index=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze spherical indentation tests on rock.")
    parser.add_argument("input", help="CSV file or directory containing CSV files")
    parser.add_argument("--out", default="out", help="Output directory for summary, plots, processed CSVs")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        files = sorted(p for p in in_path.glob("*.csv"))
    else:
        files = [in_path]

    summaries: List[Dict[str, Any]] = []

    for f in files:
        print(f"Processing {f} ...")
        try:
            summary = process_file(f, out_dir)
            summaries.append(summary)
        except Exception as e:
            print(f"  ERROR processing {f}: {e}")

    if summaries:
        df_summary = pd.DataFrame(summaries)
        df_summary.to_csv(out_dir / "indentation_summary.csv", index=False)
        print(f"Wrote summary to {out_dir / 'indentation_summary.csv'}")
    else:
        print("No files processed successfully.")

def debug_run():
    in_path = Path(r"J:\Data\Evan\Tools\Intentation_Analysis\data_raw")
    out_dir = Path(r"J:\Data\Evan\Tools\Intentation_Analysis\data_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input path:  {in_path}")
    print(f"Output path: {out_dir}")

    if in_path.is_dir():
        files = sorted(p for p in in_path.glob("*.csv"))
    else:
        files = [in_path]

    summaries = []
    for f in files:
        print(f"Processing {f} ...")
        try:
            summary = process_file(f, out_dir)
            summaries.append(summary)
        except Exception as e:
            print(f"  ERROR processing {f}: {e}")

    if summaries:
        df_summary = pd.DataFrame(summaries)
        df_summary.to_csv(out_dir / "indentation_summary.csv", index=False)
        print(f"Wrote summary to {out_dir / 'indentation_summary.csv'}")
    else:
        print("No files processed successfully.")

if __name__ == "__main__":
    # debug_run()   # use this in IDE
    main()      # use this when running from CLI with args
