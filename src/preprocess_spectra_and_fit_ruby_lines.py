from pathlib import Path
import numpy as np

from avantes_data_loader import load_folder  # <-- change if your loader filename differs

from scipy.optimize import curve_fit
from scipy.special import voigt_profile

import matplotlib.pyplot as plt


# ----------------------------
# USER TOGGLES (easy on/off)
# ----------------------------
ASK_MEASUREMENT_CONDITION = True   # <-- set False to skip prompt
EXPORT_PER_SPECTRUM_FILES = True   # <-- set False to disable per-spectrum exports
PLOT_FIRST_N = 0                   # <-- set 0 to disable diagnostic plots -- IMPORTANT: if N does not = 0, then you should close every illustration popup to allow the script to run. Otherise, it will be blocked after the first file in the folder. 

PRINT_FIRST_N = 3                  # sanity check prints (pre-fit detection)


# ----------------------------
# CONFIG
# ----------------------------
TAIL_START_IDX = 1403
TAIL_END_IDX = 1973  # inclusive

R2_OFFSET_FROM_R1_NM = 1.35
R2_WINDOW_HALF_WIDTH_NM = 0.40   # +/- 0.40 => 0.80 nm window
EXCLUDE_AROUND_R1_NM = 0.10

# Fit window: [R2 - 5 nm, R1 + 5 nm]
FIT_LEFT_PAD_NM = 5.0
FIT_RIGHT_PAD_NM = 5.0

# Bounds around initial centers (nm)
CENTER_WINDOW_NM = 0.40

# Width bounds (nm): sigma (std dev), gamma (HWHM)
SIGMA_MIN, SIGMA_MAX = 0.005, 0.50
GAMMA_MIN, GAMMA_MAX = 0.005, 1.00

# Baseline bounds on normalized spectrum
Y0_MIN, Y0_MAX = -0.2, 0.2

# Initial width guesses (nm)
SIGMA_R1_INIT, GAMMA_R1_INIT = 0.08, 0.06
SIGMA_R2_INIT, GAMMA_R2_INIT = 0.08, 0.06


# ----------------------------
# MATH HELPERS
# ----------------------------
def fwhm_voigt_approx(sigma: float, gamma: float) -> float:
    """
    Approximate Voigt FWHM from sigma (Gaussian std dev) and gamma (Lorentzian HWHM).
    """
    sigma = float(sigma)
    gamma = float(gamma)
    F_G = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    F_L = 2.0 * gamma
    return 0.5346 * F_L + np.sqrt(0.2166 * F_L * F_L + F_G * F_G)


def two_voigt_model(x, y0,
                    A_R1, R1_x, sigma_R1, gamma_R1,
                    A_R2, R2_x, sigma_R2, gamma_R2):
    return (
        y0
        + A_R1 * voigt_profile(x - R1_x, sigma_R1, gamma_R1)
        + A_R2 * voigt_profile(x - R2_x, sigma_R2, gamma_R2)
    )


# ----------------------------
# PREPROCESSING
# ----------------------------
def preprocess_single_spectrum(x_nm, y_raw):
    """
    Baseline correct + normalize + detect R1 and R2.

    Returns dict with:
      I_norm (full),
      tail_mean, tail_std,
      R1_idx, R1_x_nm,
      R2_idx, R2_x_nm, R2_I,
      R2_expected_x_nm, R2_window_min, R2_window_max
    """
    x_nm = np.asarray(x_nm, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    if x_nm.shape != y_raw.shape:
        raise ValueError(f"x and y must have same shape; got {x_nm.shape} vs {y_raw.shape}")

    n = len(y_raw)
    if not (0 <= TAIL_START_IDX < n) or not (0 <= TAIL_END_IDX < n) or (TAIL_END_IDX < TAIL_START_IDX):
        raise ValueError(
            f"Tail indices invalid for spectrum length {n}: {TAIL_START_IDX=} {TAIL_END_IDX=}"
        )

    tail = y_raw[TAIL_START_IDX:TAIL_END_IDX + 1]
    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail, ddof=1)) if tail.size > 1 else float("nan")

    # baseline correction
    y = y_raw - tail_mean

    # normalization (R1 becomes I=1)
    y_max = float(np.max(y))
    if y_max <= 0 or not np.isfinite(y_max):
        return None

    I_norm = y / y_max

    # R1: global max (longer wavelength)
    R1_idx = int(np.argmax(I_norm))
    R1_x_nm = float(x_nm[R1_idx])

    # R2: search in window around expected offset from R1
    R2_expected_x_nm = R1_x_nm - R2_OFFSET_FROM_R1_NM
    R2_window_min = R2_expected_x_nm - R2_WINDOW_HALF_WIDTH_NM
    R2_window_max = R2_expected_x_nm + R2_WINDOW_HALF_WIDTH_NM

    mask = (x_nm >= R2_window_min) & (x_nm <= R2_window_max)
    mask &= (np.abs(x_nm - R1_x_nm) > EXCLUDE_AROUND_R1_NM)

    if np.any(mask):
        idxs = np.where(mask)[0]
        R2_idx = int(idxs[np.argmax(I_norm[idxs])])
        R2_x_nm = float(x_nm[R2_idx])
        R2_I = float(I_norm[R2_idx])
    else:
        R2_idx = None
        R2_x_nm = None
        R2_I = None

    return {
        "I_norm": I_norm,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "R1_idx": R1_idx,
        "R1_x_nm": R1_x_nm,
        "R2_idx": R2_idx,
        "R2_x_nm": R2_x_nm,
        "R2_I": R2_I,
        "R2_expected_x_nm": R2_expected_x_nm,
        "R2_window_min": R2_window_min,
        "R2_window_max": R2_window_max,
    }


# ----------------------------
# FIT SETUP
# ----------------------------
def build_init_and_bounds(pre):
    """
    Build p0 and bounds for curve_fit.
    Separate widths for R1 and R2 (sigma/gamma per line).
    """
    R1_x0 = float(pre["R1_x_nm"])

    if pre["R2_x_nm"] is None:
        R2_x0 = R1_x0 - R2_OFFSET_FROM_R1_NM
        h_R2 = 0.3
    else:
        R2_x0 = float(pre["R2_x_nm"])
        h_R2 = float(pre["R2_I"]) if pre["R2_I"] is not None else 0.3

    y0_init = 0.0

    sigma_R1 = SIGMA_R1_INIT
    gamma_R1 = GAMMA_R1_INIT
    sigma_R2 = SIGMA_R2_INIT
    gamma_R2 = GAMMA_R2_INIT

    # height guesses in normalized units
    h_R1 = 1.0
    h_R2 = max(0.05, min(h_R2, 1.5))

    # convert height -> amplitude using voigt_profile(0)
    v0_R1 = float(voigt_profile(0.0, sigma_R1, gamma_R1))
    v0_R2 = float(voigt_profile(0.0, sigma_R2, gamma_R2))
    A_R1_init = h_R1 / v0_R1
    A_R2_init = h_R2 / v0_R2

    p0 = np.array([
        y0_init,
        A_R1_init, R1_x0, sigma_R1, gamma_R1,
        A_R2_init, R2_x0, sigma_R2, gamma_R2,
    ], dtype=float)

    lb = np.array([
        Y0_MIN,
        0.0, R1_x0 - CENTER_WINDOW_NM, SIGMA_MIN, GAMMA_MIN,
        0.0, R2_x0 - CENTER_WINDOW_NM, SIGMA_MIN, GAMMA_MIN,
    ], dtype=float)

    ub = np.array([
        Y0_MAX,
        1e6, R1_x0 + CENTER_WINDOW_NM, SIGMA_MAX, GAMMA_MAX,
        1e6, R2_x0 + CENTER_WINDOW_NM, SIGMA_MAX, GAMMA_MAX,
    ], dtype=float)

    return p0, (lb, ub), (R1_x0, R2_x0)


def select_fit_region(x, y_norm, R1_x0, R2_x0):
    x_min = float(R2_x0 - FIT_LEFT_PAD_NM)
    x_max = float(R1_x0 + FIT_RIGHT_PAD_NM)

    x_min = max(x_min, float(np.min(x)))
    x_max = min(x_max, float(np.max(x)))

    mask = (x >= x_min) & (x <= x_max)
    if np.count_nonzero(mask) < 20:
        raise ValueError(f"Fit window too small/out of range: [{x_min:.3f}, {x_max:.3f}] nm")

    return x[mask], y_norm[mask], x_min, x_max, mask


# ----------------------------
# EXPORTS
# ----------------------------
def export_per_spectrum_table(export_folder: Path, fname: str, x: np.ndarray, I_norm: np.ndarray,
                              mask_fit: np.ndarray, y_hat_win: np.ndarray):
    """
    Writes one file per spectrum with shared wavelength axis:
      wavelength_nm | I_norm | fit | residual

    I_norm: full range
    fit/residual: only on fit window, NaN elsewhere
    """
    base_name = Path(fname).stem
    out_path = export_folder / f"{base_name}_data_fit_resid_fullx.txt"

    fit_full = np.full_like(I_norm, np.nan, dtype=float)
    resid_full = np.full_like(I_norm, np.nan, dtype=float)

    fit_full[mask_fit] = y_hat_win
    resid_full[mask_fit] = I_norm[mask_fit] - y_hat_win

    mat = np.column_stack([x, I_norm, fit_full, resid_full])

    np.savetxt(
        out_path,
        mat,
        delimiter="\t",
        header="wavelength_nm\tI_norm\tfit\tresidual",
        comments="",
        fmt="%.8f"
    )


def write_results_tsv(path: Path, rows):
    preferred_order = [
        "file", "fit_ok", "error",
        "x_min_fit", "x_max_fit",
        "R1_center_nm", "R2_center_nm", "R1_minus_R2_nm",
        "FWHM_R1_nm", "FWHM_R2_nm",
        "sigma_R1_nm", "gamma_R1_nm", "sigma_R2_nm", "gamma_R2_nm",
        "A_R1", "A_R2",
        "y0",
        "rmse",
        "tail_mean_raw", "tail_std_raw",
    ]

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    headers = [k for k in preferred_order if k in all_keys]
    for k in sorted(all_keys):
        if k not in headers:
            headers.append(k)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# " + "\t".join(headers) + "\n") 
        for r in rows:
            f.write("\t".join(str(r.get(h, "")) for h in headers) + "\n")

def write_for_origin_txt(path: Path, rows, measurement_condition: str = ""):
    """
    Writes a single-row, Origin-ready summary file with:
      - mean / sample std dev (ddof=1) across spectra in this folder
      - using only rows where fit_ok == 1
    Other columns are left blank for later computation in Origin.
    """
    ok = [
        r for r in rows
        if r.get("fit_ok") == 1
        and np.isfinite(r.get("R1_center_nm", np.nan))
        and np.isfinite(r.get("R2_center_nm", np.nan))
        and np.isfinite(r.get("FWHM_R1_nm", np.nan))
    ]

    n_ok = len(ok)
    n_failed = len(rows) - n_ok

    def mean_sd(a: np.ndarray):
        if a.size == 0:
            return (float("nan"), float("nan"))
        mu = float(np.mean(a))
        sd = float(np.std(a, ddof=1)) if a.size > 1 else float("nan")
        return (mu, sd)

    def fmt(x: float) -> str:
        # Origin-friendly: blank if not finite
        return f"{x:.8f}" if np.isfinite(x) else ""

    r1 = np.array([r["R1_center_nm"] for r in ok], dtype=float)
    r2 = np.array([r["R2_center_nm"] for r in ok], dtype=float)
    fwhm_r1 = np.array([r["FWHM_R1_nm"] for r in ok], dtype=float)
    d12 = r1 - r2

    r1_mu, r1_sd = mean_sd(r1)
    r2_mu, r2_sd = mean_sd(r2)
    d_mu, d_sd = mean_sd(d12)
    f_mu, f_sd = mean_sd(fwhm_r1)

    headers = [

        # ---- Ground-truth pressure and temperature readouts
        "Measurement condition", # exported by Python
        "P (GPa) via R1", # computed in Origin
        "P unc. (GPa)", # computed in Origin; total uncertainty (random and systematic) ---> USE THIS as error in X when plotting something vs. P
        "P repeatability (GPa)", # computed in Origin; uncertainty in P readout introduced by random error (fluctuation of T and random error from R1 line fitting)
        "T (K)", # mean T, input in Origin
        "T unc. (K)", # computed in Origin; total uncertainty (random and instrumental)

        # ---- Spectral positions of the R1 and R2 lines
        "xc R1 mean", # computed by Python
        "xc R1 s.d.", # computed by Python
        "xc R2 mean", # computed by Python
        "xc R2 s.d.", # computed by Python

        # ---- Hydrostaticity check
        "xc R1 - xc R2, mean", # computed by Python
        "xc R1 - xc R2, s.d.", # computed by Python
        "fwhm R1 mean", # computed by Python
        "fwhm R1 s.d.", # computed by Python

        # ---- Added for uncertainty bookkeeping / Origin formulas ----
        "T s.d.", # random T uncertainty due to T fluctuation during acquisition; for fast acquisitions (ruby...), this can likely be estimated and will be constant for all entries.
        "T inst unc.", # systematic, instrumental T uncertainty; see Picolog thermocouple specifications. 

        "R1 xc ref (nm)", # Spectral position of the R1 line at ambient P and T
        "R1 xc ref unc. (nm)", # Treated as systematic error for simplicity. In reality, this is a random error. 

        "T ref (K)", # Ambient T, for computing delta T
        "T ref unc. (K)", # Respective uncertainty; Treated as systematic error for simplicity.

        "a (nm/K)", # Experimentally-derived T coefficient (slope of xc R1 vs T)
        "a unc. (nm/K)", # Respective error of linear fit to data

        "b (nm/GPa)", # P coefficient (slope of xc R1 vs P) taken from DOI: 10.1080/08957959.2021.1931168
        "b unc. (nm/GPa)", # Respective reported error of linear fit to data

        # ---- Auxiliary information
        "No. successful fits", # N (sample size) for convering standard deviation to SEM (standard error of the mean)
        "No. failed fits", # Useful for flagging "problematic" measurements, wherein poor fitting was observed
        "Time of acquisition", # Useful for correlating acquisition of Ruby lines to Picolog T data (for computing mean and s.d. of T during spectral acquisition)
    ]

    row = [

        # ---- Ground-truth pressure and temperature readouts
        measurement_condition,
        "",  # P (GPa) via R1
        "",  # P unc. (GPa)
        "",  # P repeatability (GPa)
        "",  # T (K)
        "",  # T unc. (K)

        # ---- Spectral positions
        fmt(r1_mu),
        fmt(r1_sd),
        fmt(r2_mu),
        fmt(r2_sd),

        # ---- Hydrostaticity check
        fmt(d_mu),
        fmt(d_sd),
        fmt(f_mu),
        fmt(f_sd),

        # ---- Uncertainty bookkeeping
        "",  # T s.d.
        "",  # T inst unc.

        "",  # R1 xc ref (nm)
        "",  # R1 xc ref unc. (nm)

        "",  # T ref (K)
        "",  # T ref unc. (K)

        "",  # a (nm/K)
        "",  # a unc. (nm/K)

        "",  # b (nm/GPa)
        "",  # b unc. (nm/GPa)

        # ---- Auxiliary information
        str(n_ok),
        str(n_failed),
        "",  # Time of acquisition
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        f.write("\t".join(row) + "\n")

# ----------------------------
# DIAGNOSTIC PLOT
# ----------------------------
def plot_diagnostic(fname, x, I_norm, pre, x_min, x_max, x_fit, y_hat_win):
    """
    Shows:
      - full normalized spectrum
      - R1/R2 detections
      - fit window shading
      - fitted curve (only in fit window)
    """
    plt.figure()
    plt.plot(x, I_norm, label="I_norm (full)")
    plt.axvline(pre["R1_x_nm"], linestyle="--", label=f"R1 detect: {pre['R1_x_nm']:.3f} nm")
    if pre["R2_x_nm"] is not None:
        plt.axvline(pre["R2_x_nm"], linestyle="--", label=f"R2 detect: {pre['R2_x_nm']:.3f} nm")

    plt.axvspan(x_min, x_max, alpha=0.2, label="fit window")
    plt.plot(x_fit, y_hat_win, label="two-Voigt fit (window)")

    plt.title(fname)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized intensity")
    plt.legend()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
def main():
    folder_input = input("Enter path to folder containing .txt files: ").strip().strip('"').strip("'")
    all_data = load_folder(folder_input, max_rows=None)

    export_input = input("Enter folder where fit results should be saved: ").strip().strip('"').strip("'")
    export_folder = Path(export_input)
    export_folder.mkdir(parents=True, exist_ok=True)
    
    measurement_condition = ""
    if ASK_MEASUREMENT_CONDITION:
        measurement_condition = input(
            "Measurement condition: "
        ).strip()

    results = []
    n_files = 0
    n_fit_ok = 0
    n_plotted = 0
    n_exported = 0

    for fname, d in all_data.items():
        n_files += 1

        x = np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)

        pre = preprocess_single_spectrum(x, y)
        if pre is None:
            results.append({"file": fname, "fit_ok": 0, "error": "preprocess_failed"})
            continue

        # --- Sanity print (first N) ---
        if n_files <= PRINT_FIRST_N:
            print("\n--- SANITY CHECK ---")
            print(f"File: {fname}")
            print(f"Tail mean/std (idx {TAIL_START_IDX}-{TAIL_END_IDX}): {pre['tail_mean']:.6g} / {pre['tail_std']:.6g}")
            print(f"Main peak (I=1): idx={pre['R1_idx']}, x={pre['R1_x_nm']:.4f} nm")
            print(
                f"Second-peak window: [{pre['R2_window_min']:.4f}, {pre['R2_window_max']:.4f}] nm "
                f"(expected center {pre['R2_expected_x_nm']:.4f} nm)"
            )
            if pre["R2_x_nm"] is not None:
                sep = pre["R1_x_nm"] - pre["R2_x_nm"]
                print(f"Second peak: idx={pre['R2_idx']}, x={pre['R2_x_nm']:.4f} nm, I={pre['R2_I']:.4f}")
                print(f"Separation: {sep:.4f} nm")
            else:
                print("Second peak: NOT FOUND (will fallback to R1 - 1.35 for init)")

        I_norm = pre["I_norm"]

        # init + bounds
        p0, bounds, (R1_x0, R2_x0) = build_init_and_bounds(pre)

        # fit window selection
        try:
            x_fit, y_fit, x_min, x_max, mask_fit = select_fit_region(x, I_norm, R1_x0, R2_x0)
        except Exception as e:
            results.append({"file": fname, "fit_ok": 0, "error": f"fit_window: {e}"})
            continue

        # fit
        try:
            popt, pcov = curve_fit(
                two_voigt_model,
                x_fit,
                y_fit,
                p0=p0,
                bounds=bounds,
                maxfev=30000,
            )
        except Exception as e:
            results.append({"file": fname, "fit_ok": 0, "error": f"curve_fit: {e}"})
            continue

        n_fit_ok += 1

        # compute fitted curve on window
        y_hat_win = two_voigt_model(x_fit, *popt)

        # optional export per spectrum
        if EXPORT_PER_SPECTRUM_FILES:
            export_per_spectrum_table(export_folder, fname, x, I_norm, mask_fit, y_hat_win)
            n_exported += 1

        # optional plotting
        if PLOT_FIRST_N > 0 and n_plotted < PLOT_FIRST_N:
            plot_diagnostic(fname, x, I_norm, pre, x_min, x_max, x_fit, y_hat_win)
            n_plotted += 1

        # unpack fitted params
        (y0,
         A_R1, R1_center, sigma_R1, gamma_R1,
         A_R2, R2_center, sigma_R2, gamma_R2) = popt

        # derived widths
        FWHM_R1 = fwhm_voigt_approx(sigma_R1, gamma_R1)
        FWHM_R2 = fwhm_voigt_approx(sigma_R2, gamma_R2)

        # RMSE on fit window
        rmse = float(np.sqrt(np.mean((y_fit - y_hat_win) ** 2)))

        valid_fit = (
            (rmse < 0.05)
            and (R1_center > R2_center)
            and (FWHM_R1 > 0)
            and (FWHM_R2 > 0)
        )

        results.append({
            "file": fname,
            "fit_ok": 1,
            "valid_fit": int(valid_fit),
            "error": "",

            "R1_center_nm": float(R1_center),
            "R2_center_nm": float(R2_center),

            "FWHM_R1_nm": float(FWHM_R1),
            "FWHM_R2_nm": float(FWHM_R2),

            "sigma_R1_nm": float(sigma_R1),
            "gamma_R1_nm": float(gamma_R1),
            "sigma_R2_nm": float(sigma_R2),
            "gamma_R2_nm": float(gamma_R2),

            "A_R1": float(A_R1),
            "A_R2": float(A_R2),

            "y0": float(y0),

            "tail_mean_raw": float(pre["tail_mean"]),
            "tail_std_raw": float(pre["tail_std"]),

            "rmse": float(rmse),
        })

    # Summary TSV across all spectra
    summary_path = export_folder / "voigt_fit_results.tsv"
    write_results_tsv(summary_path, results)

    summary_txt_path = export_folder / "voigt_fit_results.txt"
    write_results_tsv(summary_txt_path, results)

    # One-row summary for Origin
    for_origin_path = export_folder / "For_Origin.txt"
    write_for_origin_txt(for_origin_path, results, measurement_condition=measurement_condition)
    print("Saved For Origin:", for_origin_path)

    print("\nDone.")
    print("Files seen:", n_files)
    print("Fits successful:", n_fit_ok)
    print("Fits failed:", n_files - n_fit_ok)
    print("Saved summary:", summary_path)
    print("Saved summary:", summary_txt_path)
    print("Per-spectrum exports written:", n_exported, "(toggle EXPORT_PER_SPECTRUM_FILES)")
    print("Plots shown:", n_plotted, "(toggle PLOT_FIRST_N)")


if __name__ == "__main__":
    main()