"""
GPR-Based Earthquake Fault Slip Distribution Generator
========================================================

Given a scenario (moment magnitude + categorical style-of-faulting
features), this script uses a pre-trained Gaussian Process Regression
(GPR) model to predict fault geometry, asperity areas, and slip
statistics, then generates a "slipgen.in" file consumed by the
Fortran spatial-correlation slip generator (src/slipgen_von_karman.f90).

Usage
-----
    python src/run_pipeline.py

Or import and call run_pipeline(...) directly with custom scenario
parameters. See the __main__ block at the bottom for an example.

Requires the trained model artifacts in models/:
    gpr_model.pkl, gpr_mw_scaler.pkl, gpr_ohe.pkl,
    gpr_metadata.pkl, gpr_y_scaler.pkl
"""

import joblib
import numpy as np
import pandas as pd
import time
import os

# =====================================================================
# 1. LOAD THE SAVED GPR DATA
# =====================================================================
# This section loads all the pre-trained Gaussian Process Regression (GPR)
# artifacts that were saved to disk after model training:
#   - model_full    : the trained multi-output GPR model (one estimator per target)
#   - mw_scaler     : scaler used to normalize the moment magnitude (Mw) input
#   - ohe           : one-hot encoder for the categorical inputs (SVT, FM, TS, NM)
#   - metadata      : dictionary with column names, target names, and fixed values
#   - y_scaler_full : scaler used to inverse-transform predictions back to original units
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

print("Loading saved GPR models and metadata...")
try:
    model_full    = joblib.load(os.path.join(MODEL_DIR, "gpr_model.pkl"))
    mw_scaler     = joblib.load(os.path.join(MODEL_DIR, "gpr_mw_scaler.pkl"))
    ohe           = joblib.load(os.path.join(MODEL_DIR, "gpr_ohe.pkl"))
    metadata      = joblib.load(os.path.join(MODEL_DIR, "gpr_metadata.pkl"))
    y_scaler_full = joblib.load(os.path.join(MODEL_DIR, "gpr_y_scaler.pkl"))
    print("✓ All models and scalers loaded successfully!\n")
except FileNotFoundError as e:
    # If any of the .pkl files are missing, stop execution immediately since
    # nothing downstream can run without the trained model artifacts.
    print(f"\nERROR: Could not find one of the .pkl files in '{MODEL_DIR}'.\nDetails: {e}")
    exit()

# Pull out commonly used metadata fields so they don't need to be
# looked up from the dictionary every time they're used below.
target_cols = metadata['target_cols']   # names of the GPR output/target variables
svt_col     = metadata['svt_col']       # column name for "SVT" categorical feature
fm_col      = metadata['fm_col']        # column name for "FM" (fault mechanism) categorical feature
ts_col      = metadata['ts_col']        # column name for "TS" categorical feature
nm_col      = metadata['nm_col']        # column name for "NM" categorical feature
n_out       = len(target_cols)          # number of output targets the GPR predicts

# =====================================================================
# 2. GPR PREDICTION FUNCTION
# =====================================================================
def predict_new_gpr(Mw_new, SVT_new, FM_new, TS_new, NM_new):
    """
    Run the trained GPR model on a single new set of inputs and return
    predictions (plus uncertainty) for every target variable.

    Parameters
    ----------
    Mw_new  : float  - moment magnitude of the earthquake
    SVT_new : value  - categorical feature (e.g. site/style/tectonic type)
    FM_new  : value  - categorical feature (fault mechanism, e.g. 'ss', 'rv', 'nm')
    TS_new  : value  - categorical feature
    NM_new  : value  - categorical feature

    Returns
    -------
    pd.DataFrame with columns:
        Target            - name of the predicted variable
        Predicted_log10   - predicted value in log10 space (native GPR output space)
        Predicted_linear  - predicted value converted back to linear space (10**log10)
        Uncertainty_log10 - standard deviation of the prediction, in log10 space
    """
    # Scale the magnitude input using the same scaler fit during training.
    # NOTE: Mw is passed directly (no log transform) since it's already on
    # a natural scale that doesn't need log-compression.
    Mw_sc      = mw_scaler.transform([[Mw_new]])

    # One-hot encode the categorical inputs using the same encoder used in training.
    # Wrapped in a DataFrame so the encoder sees the expected column names.
    X_cat_new  = ohe.transform(pd.DataFrame(
                    [[SVT_new, FM_new, TS_new, NM_new]],
                    columns=[svt_col, fm_col, ts_col, nm_col]))

    # Combine the scaled numeric feature with the one-hot encoded categorical
    # features into a single feature vector for the model.
    X_new = np.hstack([Mw_sc, X_cat_new])

    # Pre-allocate arrays to hold the scaled prediction and its std dev
    # for each of the n_out target variables.
    pred_sc = np.zeros((1, n_out))
    std_sc  = np.zeros((1, n_out))

    # model_full.estimators_ is a list of individual GPR models, one per
    # target variable (this is a MultiOutputRegressor-style wrapper).
    # Loop through each one, predicting both the mean and std for that target.
    for i, est in enumerate(model_full.estimators_):
        p, s          = est.predict(X_new, return_std=True)
        pred_sc[0, i] = p[0]
        std_sc[0, i]  = s[0]

    # Convert the scaled predictions back to their original (log10) units.
    pred     = y_scaler_full.inverse_transform(pred_sc)[0]
    # Convert the scaled std dev back to original units by multiplying by
    # the scaler's per-feature scale factor (std dev scales linearly, unlike
    # the mean which also needs the offset/inverse_transform).
    std_pred = std_sc[0] * y_scaler_full.scale_

    # Package results into a tidy DataFrame: log10 value, linear value
    # (10^log10), and the uncertainty (still in log10 space).
    return pd.DataFrame({
        "Target"           : target_cols,
        "Predicted_log10"  : np.round(pred,                  6),
        "Predicted_linear" : np.round(10**pred,              4),
        "Uncertainty_log10": np.round(std_pred,              6),
    })

# =====================================================================
# 3. GENERATE slipgen.in (SQUARE ALA, EXACT-BLOCK AVLA, RANDOM ANCHORS)
# =====================================================================
def write_fault_in(fault_L, fault_W, A_ALA, A_AVLA, mean_slip, std_slip,
                   grid_N=512, n_blocks=4, ax=None, az=None, itype=1, H=0.7,
                   anchor="random", avla_anchor="random", output_path="slipgen.in"):
    """
    Build a block-based slip-distribution grid for a fault plane and write
    it out in the Fortran-readable "slipgen.in" format.

    The fault is divided into a coarse grid of rectangular blocks. Within
    that grid:
      - ALA  (Area of Large Asperity)      -> forced into a square region
      - AVLA (Area of Very Large Asperity) -> an exact-block-count region
                                               nested inside the ALA square
      - all remaining blocks               -> "background" slip

    Slip values for each region are auto-balanced so the area-weighted
    average slip across the whole fault matches `mean_slip`.

    Parameters
    ----------
    fault_L, fault_W : float   - fault length and width (km)
    A_ALA, A_AVLA     : float  - target areas (km²) for the large / very-large asperities
    mean_slip, std_slip : float - target mean slip (m) and standard deviation (m)
    grid_N            : int    - fine simulation grid resolution (points per axis)
    n_blocks          : int    - base number of coarse blocks along strike
    ax, az            : float  - fractal correlation lengths along strike/dip (km);
                                  defaults to fault_L/3 and fault_W/3 if not given
    itype             : int    - spectral model type (1=Von Karman, 2=Exponential, 3=Gaussian)
    H                 : float  - Hurst exponent for the fractal slip model
    anchor            : str    - placement rule for the ALA square within the block grid
    avla_anchor       : str    - placement rule for the AVLA region within the ALA square
    output_path       : str    - where to write the generated file

    Returns
    -------
    np.ndarray : the final (n_blocks_dip x n_blocks_strike) slip grid, in meters
    """

    # Seed the RNG using the current time so repeated runs get different
    # random asperity placements (when anchor/avla_anchor use randomness).
    used_seed = int(time.time())
    np.random.seed(used_seed)

    # --- 1. ASPECT RATIO SCALED GRID ---
    # The fine simulation grid (grid_N points) is used equally for strike and dip;
    # grid_N itself is not directly used to size the coarse block grid below,
    # it's carried through to the output file header.
    grid_dip    = grid_N
    grid_strike = grid_dip

    # Default fractal correlation lengths: 1/3 of the fault dimension along
    # each axis, unless the caller supplied explicit values (e.g. from GPR).
    if ax is None: ax = fault_L / 3.0
    if az is None: az = fault_W / 3.0

    # Number of coarse blocks along strike is capped at 5 (for stability/
    # realism per the caller's note). Number of blocks along dip is scaled
    # relative to strike blocks by the fault's aspect ratio (L/W), also capped at 5.
    n_blocks_strike    = min(n_blocks, 5)
    n_blocks_dip = min(int(round(((fault_L / fault_W) * n_blocks))), 5)

    # Physical size of each coarse block and the total block count.
    block_L      = fault_L / n_blocks_strike
    block_W      = fault_W / n_blocks_dip
    block_area   = block_L * block_W
    total_blocks = n_blocks_strike * n_blocks_dip

    # --- 2. CALCULATE SIZES FOR ALA (Square) AND AVLA (Flexible Inside Blocks) ---

    # FORCE ALA TO BE A PERFECT SQUARE:
    # Find the smallest integer k such that a (k x k) block region has an
    # area >= the target ALA area, then clip it to fit within the grid.
    k_ALA = int(np.ceil(np.sqrt(A_ALA / block_area)))
    max_k_ALA = min(n_blocks_dip, n_blocks_strike)  # can't exceed the smaller grid dimension
    if k_ALA > max_k_ALA: k_ALA = max_k_ALA
    if k_ALA < 1: k_ALA = 1

    # AVLA sizing: pick the block-count that best matches the target AVLA
    # area, then split it into a (dip x strike) rectangle that's as close
    # to square as possible, favoring a slightly wider "strike" dimension.
    target_AVLA = max(1, int(np.round(A_AVLA / block_area)))
    k_AVLA_dip  = max(1, int(np.floor(np.sqrt(target_AVLA))))
    k_AVLA_strike = max(1, int(np.ceil(target_AVLA / k_AVLA_dip)))

    # Ensure the AVLA rectangle can't be larger than the ALA square it must
    # be nested inside.
    k_AVLA_dip    = min(k_AVLA_dip, k_ALA)
    k_AVLA_strike = min(k_AVLA_strike, k_ALA)

    # Block counts for each region, used in the slip-balancing math below.
    N_AVLA     = k_AVLA_dip * k_AVLA_strike               # blocks inside the very-large asperity
    N_ALA_only = (k_ALA ** 2) - N_AVLA                     # ALA blocks that are NOT also AVLA
    N_bg       = total_blocks - N_AVLA - N_ALA_only        # remaining "background" blocks

    # --- 3. AUTO-BALANCING SLIP MATH (Mu and Sigma) ---
    # Assign higher slip to the asperities (AVLA gets +1 sigma, ALA gets
    # +0.5 sigma above the mean) and solve for the background slip that
    # keeps the overall area-weighted average equal to `mean_slip`.
    slip_AVLA = mean_slip + (1.0 * std_slip)
    slip_ALA  = mean_slip + (0.5 * std_slip)

    slip_bg = (mean_slip * total_blocks - N_AVLA * slip_AVLA - N_ALA_only * slip_ALA) / max(1, N_bg)

    # Safety check: if the background slip comes out negative (physically
    # impossible), cap std_slip to the largest value that keeps slip_bg >= 0,
    # then recompute all three slip values.
    if slip_bg < 0:
        max_sigma = (mean_slip * total_blocks) / (2.0 * N_AVLA + 1.0 * N_ALA_only)
        std_slip  = max_sigma * 0.95
        slip_AVLA = mean_slip + (1.0 * std_slip)
        slip_ALA  = mean_slip + (0.5 * std_slip)
        slip_bg   = (mean_slip * total_blocks - N_AVLA * slip_AVLA - N_ALA_only * slip_ALA) / max(1, N_bg)
        print(f"  [Notice] Standard Deviation capped to {std_slip:.3f}m to prevent negative background.")

    # --- 4. STAMP THE ASPERITIES ---
    # Start with a uniform grid of background slip, then overwrite the
    # ALA and AVLA sub-regions with their respective slip values.
    block_grid = np.full((n_blocks_dip, n_blocks_strike), slip_bg)

    def get_coords(k_d, k_s, anchor_type, max_d, max_s):
        """
        Compute the top-left (row, col) coordinate at which to place a
        (k_d x k_s) sub-region within a (max_d x max_s) grid, according
        to the requested anchor strategy.
        """
        if anchor_type == "top-left":      return 0, 0
        elif anchor_type == "top-right":   return 0, max_s - k_s
        elif anchor_type == "bottom-left": return max_d - k_d, 0
        elif anchor_type == "bottom-right":return max_d - k_d, max_s - k_s
        elif anchor_type == "middle":      return (max_d - k_d) // 2, (max_s - k_s) // 2
        elif anchor_type == "random-corner":
            # Pick one of the four corners at random, then reuse the
            # corner logic above.
            choices = ["top-left", "top-right", "bottom-left", "bottom-right"]
            return get_coords(k_d, k_s, np.random.choice(choices), max_d, max_s)
        else:
            # "random" (or any unrecognized value): place the region at a
            # uniformly random valid position anywhere within the bounds.
            r = np.random.randint(0, max_d - k_d + 1) if max_d > k_d else 0
            c = np.random.randint(0, max_s - k_s + 1) if max_s > k_s else 0
            return r, c

    # >> PLACE THE SQUARE ALA <<
    # Determine where the (k_ALA x k_ALA) large-asperity square goes within
    # the full block grid, then stamp it with the ALA slip value.
    r_ala, c_ala = get_coords(k_ALA, k_ALA, anchor, n_blocks_dip, n_blocks_strike)
    block_grid[r_ala:r_ala+k_ALA, c_ala:c_ala+k_ALA] = slip_ALA

    # >> PLACE THE AVLA (Inside the ALA) <<
    # Determine where the very-large-asperity rectangle sits *within* the
    # ALA square (local coordinates), then translate to absolute grid
    # coordinates by adding the ALA's own offset, and stamp its slip value.
    r_off, c_off = get_coords(k_AVLA_dip, k_AVLA_strike, avla_anchor, k_ALA, k_ALA)
    r_avla = r_ala + r_off
    c_avla = c_ala + c_off
    block_grid[r_avla:r_avla+k_AVLA_dip, c_avla:c_avla+k_AVLA_strike] = slip_AVLA

    # ══════════════════════════════════════════════════════════════════════
    # WRITE EXACT slipgen.in FORMAT
    # ══════════════════════════════════════════════════════════════════════
    # Write the fault geometry, grid resolution, fractal parameters, block
    # counts, and the final slip grid to a Fortran-style input file that
    # the downstream slip-generation program expects.
    try:
        with open(output_path, 'w') as fid:
            fid.write(f"{fault_L:.1f}\t{fault_W:.1f}\t\t! L (km), W (km)\n")
            fid.write(f"{grid_strike}\t{grid_dip}\t\t\t! M, N\n")
            fid.write(f"{ax:.2f}\t{az:.2f}\t\t! ax (km), az (km)\n")
            fid.write(f"{int(itype)}\t{H:.2f}\t\t\t! Model (1=VK, 2=Exp, 3=Gauss) & H\n")
            fid.write(f"{n_blocks_strike}\t{n_blocks_dip}\t\t\t! NX, NY\n")

            # Write the slip grid row by row (one row per dip-direction block),
            # tab-separated, formatted to 2 decimal places.
            for r in range(n_blocks_dip):
                row_str = "\t".join([f"{val:.2f}" for val in block_grid[r, :]])
                fid.write(f"{row_str}\n")

        # Summary printout confirming what was written and how the
        # asperities were sized/placed.
        print('\n================================================')
        print(f'✓ SUCCESS: Wrote Fortran input file to "{output_path}"')
        print(f'  -> Base Grid : {n_blocks_strike}x{n_blocks_dip} (Aspect Ratio Matched)')
        print(f'  -> ALA Size  : {k_ALA}x{k_ALA} (Square), Anchored: {anchor.upper()}')
        print(f'  -> AVLA Size : {k_AVLA_strike}x{k_AVLA_dip} (Exact Blocks), Anchored: {avla_anchor.upper()}')
        print('================================================\n')

    except IOError:
        raise IOError(f"Cannot open file for writing: {output_path}")

    return block_grid

# =====================================================================
# 4. RUN THE AUTOMATED PIPELINE
# =====================================================================
def run_pipeline(Mw, SVT, FM, TS, NM, grid_N=512, H=0.7,
                 model_choice="von_karman",
                 anchor="random", avla_anchor="random", n_blocks=5,
                 output_path="slipgen.in"):
    """
    End-to-end pipeline: given earthquake scenario inputs, use the trained
    GPR model to predict fault geometry / asperity / slip statistics, then
    generate the corresponding "slipgen.in" file.

    Parameters
    ----------
    Mw, SVT, FM, TS, NM : GPR model inputs (magnitude + categorical features)
    grid_N       : int   - fine simulation grid resolution passed to write_fault_in
    H            : float - Hurst exponent
    model_choice : str   - which fractal spectral model to use:
                            "von_karman", "exponential", or "gaussian"
    anchor       : str   - ALA placement strategy (see get_coords)
    avla_anchor  : str   - AVLA placement strategy (see get_coords)
    n_blocks     : int   - base coarse block count along strike
    output_path  : str   - where to write the generated slipgen.in file
    """

    print(f"--- Running Pipeline for Mw={Mw}, FM={FM} ---")

    # Get all GPR-predicted quantities (length, width, asperity areas,
    # slip statistics, fractal lengths) for this scenario in one call.
    gpr_preds = predict_new_gpr(Mw, SVT, FM, TS, NM)

    try:
        # EXTRACT DIMENSIONS
        # Use regex matching on the target names so this works regardless
        # of minor naming variations, as long as the key words are present.
        pred_L = gpr_preds.loc[gpr_preds['Target'].str.contains('Eff.*Length', regex=True), 'Predicted_linear'].values[0]
        pred_W = gpr_preds.loc[gpr_preds['Target'].str.contains('Eff.*Width', regex=True), 'Predicted_linear'].values[0]

        # EXTRACT ALA & AVLA
        pred_ALA = gpr_preds.loc[gpr_preds['Target'].str.contains('Area of large asperity', regex=True), 'Predicted_linear'].values[0]
        pred_AVLA = gpr_preds.loc[gpr_preds['Target'].str.contains('Area of.*very large asperity', regex=True), 'Predicted_linear'].values[0]

        # PHYSICS SAFEGUARD
        # The very-large asperity area can never exceed the large asperity
        # area. If the (independently-predicted) GPR outputs violate this,
        # clamp AVLA to 80% of ALA rather than trusting the raw ML output.
        if pred_AVLA > pred_ALA:
            print(f"  [Warning] ML Anomaly: AVLA ({pred_AVLA:.2f}) > ALA ({pred_ALA:.2f}). Applying physical constraints.")
            pred_AVLA = pred_ALA * 0.80

        # EXTRACT MEAN SLIP AND STD DEV (Convert cm to meters)
        pred_D_cm   = gpr_preds.loc[gpr_preds['Target'].str.contains('Eff Mean Slip', regex=True), 'Predicted_linear'].values[0]
        pred_Std_cm = gpr_preds.loc[gpr_preds['Target'].str.contains('Eff Standard Deviation', regex=True), 'Predicted_linear'].values[0]
        pred_D_m    = pred_D_cm / 100.0
        pred_Std_m  = pred_Std_cm / 100.0

        # DYNAMICALLY EXTRACT FRACTAL PARAMETERS BASED ON USER CHOICE
        # Select which set of predicted fractal correlation-length targets
        # (and the corresponding spectral model integer code) to use,
        # based on the requested model_choice string.
        choice_lower = model_choice.lower()
        if "exp" in choice_lower:
            ax_target, az_target, itype = 'ax_exp', 'az_exp', 2
        elif "gauss" in choice_lower:
            ax_target, az_target, itype = 'ax_gauss', 'az_gauss', 3
        else: # Default to Von Karman
            ax_target, az_target, itype = 'ax_von', 'az_von', 1

        pred_ax = gpr_preds.loc[gpr_preds['Target'].str.contains(ax_target, regex=True), 'Predicted_linear'].values[0]
        pred_az = gpr_preds.loc[gpr_preds['Target'].str.contains(az_target, regex=True), 'Predicted_linear'].values[0]

    except IndexError:
        # Raised if any of the .str.contains(...) filters above matched
        # zero rows, meaning an expected target name wasn't found in
        # gpr_preds. Print the available target names to help debugging.
        print("\nERROR: Could not find one of the required target names.")
        print(f"Available targets are: {gpr_preds['Target'].tolist()}")
        return

    # Log a concise summary of all the GPR-derived scenario parameters
    # before generating the fault input file.
    print(f"GPR Output -> L={pred_L:.2f}km, W={pred_W:.2f}km")
    print(f"              ALA={pred_ALA:.2f}km², AVLA={pred_AVLA:.2f}km²")
    print(f"              Mean Slip={pred_D_m:.3f}m, Sigma={pred_Std_m:.3f}m")
    print(f"              Model={model_choice.upper()}, ax={pred_ax:.2f}km, az={pred_az:.2f}km")

    # Feed all the GPR-predicted values into the file-writing function to
    # produce the final "slipgen.in" file.
    write_fault_in(
        fault_L=pred_L,
        fault_W=pred_W,
        A_ALA=pred_ALA,
        A_AVLA=pred_AVLA,
        mean_slip=pred_D_m,
        std_slip=pred_Std_m,
        ax=pred_ax,
        az=pred_az,
        itype=itype,             # Automatically passed based on model choice
        grid_N=grid_N,
        n_blocks=n_blocks,
        H=H,
        anchor=anchor,
        avla_anchor=avla_anchor,
        output_path=output_path
    )

# =====================================================================
# EXECUTE
# =====================================================================
# Example / default invocation: runs the full pipeline for a single
# scenario (Mw 6.5 strike-slip earthquake) and writes out slipgen.in.
if __name__ == "__main__":

    run_pipeline(
        Mw  = 6.5,
        SVT = metadata['svt_fixed'],
        FM  = 'ss',
        TS  = '0',
        NM  = '0',

        # --- USER INPUTS ---
        grid_N       = 256,
        H            = 0.7,
        n_blocks     = 5,                 # use either 3,4 or 5 - higher fractals will give unrealistic slips

        # --- NEW SETTINGS ---
        model_choice = "von_karman",      # Choose: "von_karman", "exponential", "gaussian"
    )
