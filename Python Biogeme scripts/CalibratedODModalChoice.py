#!/usr/bin/env python3
"""
Freight modal-choice framework for aggregate OD matrices.

This script estimates and validates a multinomial logit (MNL) model for freight
modal choice using aggregate origin-destination (OD) matrices. It was written for
a situation where the input data are not individual choices but annual tonnes by
mode, OD pair, and commodity group.

The three transport modes are assumed to use the following convention:

    1 = road, used as the reference mode
    2 = inland waterway transport (IWW)
    3 = rail

Each input row represents one OD pair and one commodity group. The observed
modal shares are computed from the tonnes:

    share_m = qty_m / (qty1 + qty2 + qty3)

The model is estimated with Biogeme using a share-weighted log-likelihood, and
the rows can be weighted by total tonnes.

Purpose of the script
---------------------
The goal is to provide a reusable framework for comparing several datasets that
may differ in geography, spatial resolution, and modal availability. In the
current project, for example, the same architecture is applied to:

    model 1: a broad European NUTS2 dataset,
    model 2: a Benelux-plus NUTS3 dataset,
    model 3: a Germany NUTS3 dataset.

The framework has two layers:

1. Behavioral layer
   This is the estimated modal-choice model. It captures systematic effects such
   as relative cost, relative moving time, commodity-specific constants, distance
   profiles, and a rail-market correction.

2. Base-year calibration layer
   This is an optional post-estimation correction applied to the IWW and rail
   utilities. It is designed to improve reproduction of the observed base-year
   modal OD matrices. These calibration constants are useful for matrix fitting,
   but they should not be interpreted as behavioral coefficients.

Why datasets are estimated sequentially
---------------------------------------
A single pooled Biogeme model could be written for all datasets at once. However,
in the intended use of this script, the behavioral coefficients are dataset-
specific. If all coefficients are dataset-specific, estimating the datasets
separately is equivalent to estimating one large pooled model with interactions
by dataset.

The sequential approach is preferable because it is much more robust in memory:

    - each dataset is loaded, estimated, validated, and released in turn;
    - large row-level validation tables are not kept in memory across datasets;
    - fine calibration levels such as mode_od or mode_od_group can otherwise
      create very large intermediate objects;
    - only compact summary tables are retained for the final cross-dataset
      comparison.

This is the reason for the garbage-collection calls and for not returning the
full long-form OD-cell dataframe from the validation functions.

Behavioral model
----------------
Road is normalized to zero. IWW and rail utilities are specified relative to
road:

    V_road = 0

    V_iww =
        ASC_iww[g]
      + B_COST_IWW * centered_log(cost_iww / cost_road)
      + B_MOVE_IWW * centered_log(vduration_iww / vduration_road)
      + distance_spline_iww

    V_rail =
        ASC_rail[g]
      + B_MOVE_RAIL * centered_log(vduration_rail / vduration_road)
      + distance_spline_rail
      + B_RAIL_IWW_AVAILABLE * I(IWW is available)

Optional term, activated with --include-rail-cost:

    + B_COST_RAIL * centered_log(cost_rail / cost_road)

where:
    - g is the commodity group, usually NST/R chapters 0 to 9;
    - vduration is the moving time variable in the database;
    - cost is total transport cost;
    - the distance spline is based on road distance, scaled in hundreds of km;
    - centered_log variables are log-ratios minus weighted means computed on
      rows where the corresponding mode is available.

Rail cost is excluded by default because previous diagnostics showed that it is
not robustly identified across all datasets. It can still be included for
sensitivity analysis.

Availability handling
---------------------
A mode is considered available in a row if its cost, distance, and moving time
are all present and strictly positive. Road must be available; rows with invalid
road data are dropped. IWW and rail may be unavailable. Unavailable alternatives
receive a very negative utility so that their predicted probability is
effectively zero.

Weighting
---------
The row weight is computed from total tonnes. Available choices are:

    raw    : weight = total_qty
    power  : weight = total_qty ** alpha
    sqrt   : weight = sqrt(total_qty)
    log1p  : weight = log(1 + total_qty)
    none   : weight = 1

For behavioral interpretation, power weights such as alpha = 0.8 or 0.9 may be
more balanced. For base-year OD-matrix reproduction, raw weights are often more
appropriate because large OD cells should dominate the fit.

Base-year calibration
---------------------
After estimating the behavioral model, the script can apply additive constants
to IWW and rail utilities:

    V_iww_calibrated  = V_iww  + CAL_IWW[cell]
    V_rail_calibrated = V_rail + CAL_RAIL[cell]

Road remains the reference mode. Calibration constants are estimated iteratively
using log-ratio updates comparing observed and predicted tonnes in each
calibration cell.

Available calibration levels are:

    none                         no calibration
    mode                         one constant per non-road mode
    mode_group                   mode x commodity group
    mode_distance                mode x distance band
    mode_group_distance          mode x commodity group x distance band
    mode_iww_availability        mode x IWW-availability segment
    mode_group_iww_availability  mode x group x IWW-availability segment
    mode_od                      mode x OD pair
    mode_od_group                mode x OD pair x commodity group

Practical interpretation:

    - mode_group_distance is a moderate segment calibration;
    - mode_od is a useful compromise for corridor-level calibration;
    - mode_od_group is the strongest base-year matrix reproduction option, but
      also the most likely to overfit.

The option --calibration-max-abs-constant caps the absolute value of calibration
constants. This is recommended for granular levels such as mode_od_group, because
many observed non-road cells are zero and the calibration may otherwise create
very large negative constants to suppress technically available but unused modes.

Validation outputs
------------------
The script reports two families of validation tables:

1. Aggregate share / aggregate tonnes validation
   These tables compare observed and predicted modal totals by mode, group,
   distance band, and IWW availability.

2. OD-matrix cell validation
   These tables compare observed and predicted tonnes cell by cell. They include
   MAE, RMSE, weighted MAE, weighted RMSE, MAPE on non-zero observed cells, WAPE,
   and bias as a percentage of observed tonnes.

MAPE is computed only for cells with positive observed tonnes, because percentage
error is undefined for zero observed cells. WAPE is usually more reliable for
sparse modal matrices.

Train/test holdout validation
-----------------------------
The script can also run a train/test holdout experiment. This is activated with:

    --holdout-fraction <fraction>
    --holdout-by <row|od|od_group|group>
    --holdout-seed <integer>

When --holdout-fraction is greater than zero, each dataset is split before
preprocessing. The training sample is used to estimate the behavioral
coefficients, the centering constants, the safe fallback values used for
unavailable modes, and the optional calibration constants. The test sample is
then transformed with the training centering constants and safe fallback values.
This avoids using information from the test sample when estimating or centering
the model.

The option --holdout-by controls the unit of the split:

    row       random individual OD-commodity rows
    od        complete OD pairs; all commodity groups for a selected OD pair
              are placed in the same sample
    od_group  complete OD x commodity-group cells
    group     complete commodity groups

For freight OD matrices, --holdout-by od is usually the most informative first
choice, because it tests whether the model can be applied to OD pairs that were
not used during estimation. A row-level split is easier but weaker, because rows
from the same OD pair may appear in both train and test samples.

Holdout validation is especially useful for separating two questions:

    1. Does the behavioral MNL skeleton generalize outside the estimation
       sample? This is assessed with holdout pre-calibration reports.

    2. Do coarser calibration levels, such as mode_group_distance or
       mode_group_iww_availability, provide transferable correction factors?
       This is assessed with holdout calibrated reports.

For very granular calibration levels such as mode_od or mode_od_group, a held-
out OD pair or OD x group cell has no estimated calibration constant. In that
case, the script applies the available training constants and treats missing
calibration keys as zero. This is intentional and is reported through a
calibration-key hit-rate diagnostic. Therefore, a poor holdout result for
mode_od_group should not be interpreted as a failure of base-year matrix
reproduction; rather, it shows that OD-specific calibration constants are not
transferable to unseen OD cells unless a coarser fallback strategy is used.

When holdout is enabled, the script writes the usual in-sample training
validation tables and additional files tagged as:

    validation_holdout_precalibration_*
    validation_holdout_calibrated_*
    validation_holdout_calibration_key_match.csv

The holdout reports use the same aggregate-share and OD-cell validation metrics
as the in-sample reports, including WAPE, MAE, RMSE, weighted MAE, and weighted
RMSE.

Typical command
---------------
Example for base-year OD-matrix reproduction with capped OD x group calibration:

    python CalibratedODModalChoice.py \
        --models 1,2,3 \
        --dataset-labels Europe_NUTS2,Benelux_NUTS3,Germany_NUTS3 \
        --groups 0,1,2,3,4,5,6,7,8,9 \
        --weight-type raw \
        --dist-thresholds-km 150,300 \
        --calibration-level mode_od_group \
        --calibration-max-abs-constant 8 \
        --include-rail-cost \
        --user nodus \
        --password nodus \
        --host 127.0.0.1 \
        --database nodus


Example for a train/test holdout test by OD pair with transferable segment
calibration:

    python CalibratedODModalChoice.py \
        --models 1,2,3 \
        --dataset-labels Europe_NUTS2,Benelux_NUTS3,Germany_NUTS3 \
        --groups 0,1,2,3,4,5,6,7,8,9 \
        --weight-type raw \
        --dist-thresholds-km 150,300 \
        --calibration-level mode_group_distance \
        --calibration-max-abs-constant 8 \
        --holdout-fraction 0.2 \
        --holdout-by od \
        --holdout-seed 12345 \
        --user nodus \
        --password nodus \
        --host 127.0.0.1 \
        --database nodus

Expected input tables
---------------------
The script reads MySQL tables named:

    biogeme_input_modelX

where X is the suffix supplied in --models. Each table must contain at least:

    org, dst, grp
    cost1, cost2, cost3
    length1, length2, length3
    vduration1, vduration2, vduration3
    qty1, qty2, qty3

Optional diagnostic variables such as fduration1, fduration2, and fduration3 are
used only for diagnostic summaries if present.

Main outputs and output naming
------------------------------
For each dataset, the script writes:
    - estimated parameter table;
    - centering constants;
    - pre-calibration validation tables;
    - post-calibration validation tables, if calibration is enabled;
    - calibration constants and calibration history, if calibration is enabled;
    - holdout/test validation tables, if --holdout-fraction is greater than zero.

After all datasets are processed, it writes combined comparison tables across
datasets.

Output names are intentionally compact. By default, no project/model prefix is
added to the generated filenames, and dataset labels are not included in the
filenames. If a prefix is useful to distinguish several runs in the same output
directory, use:

    --model-name-prefix <prefix>

The prefix is sanitized and prepended to dataset and comparison filenames. The
terminal output is also copied to a Markdown file in the output directory. Its
filename follows the same run-level naming pattern as the combined comparison
files, with the suffix _terminal_output.md, so runs with different options do
not overwrite each other. This file is useful because the console output
contains compact diagnostics and summaries that are not always convenient to
reconstruct from the CSV files.

Optional Nodus parameter-table export
-------------------------------------
The estimated coefficients, centering constants, distance-spline settings, and
base-year calibration constants can also be stored in a MySQL key-value table
that can be read later by a Nodus modal-choice plugin. Activate this with:

    --store-parameters-in-table

The table name can be controlled with:

    --parameter-table       explicit base table name
    --parameter-table-prefix generated prefix, default modal_choice_params

If several datasets are estimated in one run, the script appends _modelX to the
selected table name, for example modal_choice_params_model1. The corresponding
Nodus .costs file only needs a line such as:

    @paramTable=modal_choice_params_model1

The exported parameter table deliberately uses a simple structure:

    param_key    VARCHAR
    param_value  TEXT
    param_type   VARCHAR

Dataset metadata are stored as ordinary key-value rows, for example
@MC.Model, @MC.DatasetLabel, and @MC.ModelName. This keeps the table structure
identical to what the Java plugin needs to read.

Important caution
-----------------
The behavioral coefficients and the calibration constants have different
interpretations. Behavioral coefficients describe systematic sensitivity to
variables such as cost, time, and distance. Calibration constants are base-year
matrix correction factors. They improve reproduction of the observed modal OD
matrices but should be used carefully in forecasting.
"""


from __future__ import annotations

import argparse
import datetime as _datetime
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters

from BiogemeUtils import dropExistentOutput


# ---------------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------------

# Input tables are expected to be named biogeme_input_modelX, where X is the
# integer suffix passed through --models.
TABLE_PREFIX = "biogeme_input_model"

# Optional prefix used when naming output files. It is blank by default so that
# filenames stay compact. Use --model-name-prefix to set it at runtime. The
# dataset label is deliberately not used in output filenames.
MODEL_NAME_PREFIX = ""

# Distance is used in utility functions through piecewise-linear spline
# segments. Scaling by 100 means that one model unit represents 100 km.
DIST_SCALE = 100.0

# Default internal spline knots in km. With [150, 300], three segments are
# created: 0-150 km, 150-300 km, and 300+ km.
DEFAULT_DIST_THRESHOLDS_KM = [150.0, 300.0]

# Utility assigned to unavailable alternatives. It must be very negative, but
# not so extreme that numerical underflow becomes problematic.
UNAVAILABLE_UTILITY = -1.0e6

# All output files are written in this directory, relative to the current
# working directory from which the script is launched.
OUTPUT_DIR = Path("output")


# =============================================================================
# Parsing helpers
# =============================================================================

def parse_int_list(text_value: str) -> List[int]:
    values = [int(x.strip()) for x in text_value.split(",") if x.strip() != ""]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_groups(groups_text: Optional[str]) -> Optional[List[int]]:
    if groups_text is None or groups_text.strip() == "":
        return None
    return parse_int_list(groups_text)


def parse_labels(labels_text: Optional[str], models: List[int]) -> Dict[int, str]:
    if labels_text is None or labels_text.strip() == "":
        return {m: f"model{m}" for m in models}

    labels = [x.strip() for x in labels_text.split(",") if x.strip() != ""]
    if len(labels) != len(models):
        raise ValueError("--dataset-labels must contain the same number of labels as --models.")
    return dict(zip(models, labels))


def parse_thresholds(thresholds_text: Optional[str]) -> List[float]:
    if thresholds_text is None or thresholds_text.strip() == "":
        return list(DEFAULT_DIST_THRESHOLDS_KM)

    values = [float(x.strip()) for x in thresholds_text.split(",") if x.strip() != ""]
    if not values:
        raise ValueError("Distance thresholds list is empty.")
    if any(v <= 0 for v in values):
        raise ValueError("Distance thresholds must be strictly positive.")
    if values != sorted(values):
        raise ValueError("Distance thresholds must be in increasing order.")
    if len(set(values)) != len(values):
        raise ValueError("Distance thresholds must be unique.")
    return values


def log(message: str) -> None:
    print(message, flush=True)


def sanitize_output_token(value: Optional[str]) -> str:
    """Return a filesystem-safe token for generated filenames."""
    if value is None:
        return ""
    token = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).strip())
    token = token.strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def build_output_name(*parts: object) -> str:
    """Join non-empty filename tokens with underscores."""
    tokens = [sanitize_output_token(str(part)) for part in parts if part is not None and str(part).strip() != ""]
    return "_".join(token for token in tokens if token)


class TeeStream:
    """Write console output both to the terminal and to a run-log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def start_terminal_log(output_dir: Path, run_name: str) -> Tuple[Path, object, object, object]:
    """Tee stdout/stderr to a Markdown file in the output directory.

    The log filename uses the same run-level prefix as the combined comparison
    files, followed by _terminal_output.md. This prevents logs from different
    model settings from overwriting each other in the same output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_run_name = build_output_name(run_name) or "run"
    filename = build_output_name(safe_run_name, "terminal_output") + ".md"
    path = output_dir / filename
    fh = path.open("w", encoding="utf-8")
    fh.write(f"# Terminal output for {safe_run_name}\n\n")
    fh.write(f"Generated: {_datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
    fh.write("```text\n")
    fh.flush()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(old_stdout, fh)
    sys.stderr = TeeStream(old_stderr, fh)
    return path, fh, old_stdout, old_stderr


def stop_terminal_log(
    terminal_log_path: Path,
    terminal_log_file: object,
    old_stdout: object,
    old_stderr: object,
) -> None:
    """Restore stdout/stderr and close the Markdown run-log file."""
    try:
        log(f"\nTerminal output saved to: {terminal_log_path}")
        terminal_log_file.write("```\n")
        terminal_log_file.flush()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        terminal_log_file.close()


def print_run_parameters(
    args: argparse.Namespace,
    models: List[int],
    dataset_labels: Dict[int, str],
    groups: Optional[List[int]],
    dist_thresholds_km: List[float],
    weight_alpha: Optional[float],
) -> None:
    """
    Print a compact summary of the effective run configuration.

    This is mainly for reproducibility. The terminal output can be copied into a
    report, and it also makes it easier to check that the intended calibration
    level, weight type, and rail-cost option were actually used.
    """
    log("\n" + "=" * 80)
    log("Run parameters")
    log("=" * 80)

    params = {
        "models": models,
        "dataset_labels": dataset_labels,
        "groups": groups if groups is not None else "all",
        "weight_type": args.weight_type,
        "weight_alpha": weight_alpha,
        "dist_thresholds_km": dist_thresholds_km,
        "calibration_level": args.calibration_level,
        "include_rail_cost": args.include_rail_cost,
        "generate_reports": args.generate_reports,
        "save_row_predictions": args.save_row_predictions,
        "max_iterations": args.max_iterations,
        "calibration_max_iterations": getattr(args, "calibration_max_iterations", None),
        "calibration_damping": getattr(args, "calibration_damping", None),
        "calibration_max_step": getattr(args, "calibration_max_step", None),
        "calibration_max_abs_constant": getattr(args, "calibration_max_abs_constant", None),
        "user": args.user,
        "host": args.host,
        "database": args.database,
        "output_dir": args.output_dir,
        "model_name_prefix": getattr(args, "model_name_prefix", ""),
        "store_parameters_in_table": getattr(args, "store_parameters_in_table", False),
        "parameter_table": getattr(args, "parameter_table", None),
        "parameter_table_prefix": getattr(args, "parameter_table_prefix", None),
        "holdout_fraction": getattr(args, "holdout_fraction", 0.0),
        "holdout_by": getattr(args, "holdout_by", "od"),
        "holdout_seed": getattr(args, "holdout_seed", 12345),
    }

    for key, value in params.items():
        log(f"{key}: {value}")


# =============================================================================
# Loading data
# =============================================================================

def load_dataset_from_mysql(
    model: int,
    dataset_label: str,
    user: str,
    password: str,
    host: str,
    database: str,
    groups: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Load one model table from MySQL."""
    table = f"{TABLE_PREFIX}{model}"

    url = URL.create(
        drivername="mysql+mysqlconnector",
        username=user,
        password=password,
        host=host,
        database=database,
    )
    engine = create_engine(url)

    if groups is None:
        query = text(f"SELECT * FROM {table}")
        params = {}
    else:
        groups = list(sorted(set(int(g) for g in groups)))
        if not groups:
            raise ValueError("The list of groups to estimate is empty.")
        placeholders = ", ".join(f":g{i}" for i in range(len(groups)))
        query = text(f"SELECT * FROM {table} WHERE grp IN ({placeholders})")
        params = {f"g{i}": g for i, g in enumerate(groups)}

    log(f"Loading {table}...")
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        raise ValueError(f"No rows returned from table {table}.")

    df["dataset_id"] = int(model)
    df["dataset_label"] = dataset_label
    return df


# =============================================================================
# Preprocessing
# =============================================================================

def check_required_columns(df: pd.DataFrame) -> None:
    required = {
        "org", "dst", "grp",
        "cost1", "cost2", "cost3",
        "length1", "length2", "length3",
        "vduration1", "vduration2", "vduration3",
        "qty1", "qty2", "qty3",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def positive_fill(series: pd.Series, default: float) -> pd.Series:
    s = series.copy()
    s = s.where((~s.isna()) & (s > 0), other=default)
    return s.astype(float)


def make_weight(
    total_qty: pd.Series,
    weight_type: str,
    weight_alpha: Optional[float],
) -> pd.Series:
    q = total_qty.astype(float)
    if (q <= 0).any():
        raise ValueError("total_qty must be strictly positive before weights are created.")

    if weight_type == "raw":
        return q
    if weight_type == "sqrt":
        return np.sqrt(q)
    if weight_type == "log1p":
        return np.log1p(q)
    if weight_type == "none":
        return pd.Series(1.0, index=total_qty.index)
    if weight_type == "power":
        if weight_alpha is None:
            raise ValueError("weight_alpha must be provided when weight_type='power'.")
        if not np.isfinite(weight_alpha) or weight_alpha < 0:
            raise ValueError("weight_alpha must be finite and non-negative.")
        return q ** float(weight_alpha)

    raise ValueError("weight_type must be one of: raw, sqrt, log1p, none, power")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        raise ValueError("No valid observations for weighted mean.")
    return float(np.average(v[ok], weights=w[ok]))


def weighted_corr(df: pd.DataFrame, cols: List[str], weight_col: str) -> pd.DataFrame:
    x = df[cols].to_numpy(dtype=float)
    w = df[weight_col].to_numpy(dtype=float)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(w) & (w > 0)
    x = x[ok]
    w = w[ok]
    if x.shape[0] == 0:
        return pd.DataFrame(np.nan, index=cols, columns=cols)

    w = w / w.sum()
    mean = np.sum(w[:, None] * x, axis=0)
    xc = x - mean
    cov = (w[:, None] * xc).T @ xc
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)
    return pd.DataFrame(corr, index=cols, columns=cols)


def piecewise_segments_scaled(
    x: pd.Series,
    thresholds_scaled: Sequence[float],
) -> pd.DataFrame:
    thresholds = sorted(float(t) for t in thresholds_scaled)
    x_arr = x.to_numpy(dtype=float)

    if np.any(~np.isfinite(x_arr)) or np.any(x_arr < 0):
        raise ValueError("Distance spline variable must be finite and non-negative.")

    cols: Dict[str, np.ndarray] = {}
    lower = 0.0
    for i, upper in enumerate(thresholds, start=1):
        cols[f"dseg_raw_{i}"] = np.clip(x_arr - lower, 0.0, upper - lower)
        lower = upper
    cols[f"dseg_raw_{len(thresholds) + 1}"] = np.maximum(x_arr - lower, 0.0)
    return pd.DataFrame(cols, index=x.index)


def format_distance_boundary(value: float) -> str:
    """Return a stable boundary label used in distance-band strings and keys."""
    value = float(value)
    if abs(value - round(value)) < 1.0e-9:
        return str(int(round(value)))
    return ("%.12g" % value).replace(".", "p")


def make_distance_band(distance_km: pd.Series, thresholds_km: Sequence[float]) -> pd.Series:
    """Create distance-band labels from the command-line thresholds.

    The labels are deliberately aligned with the Java plugin. With thresholds
    [150, 300], the output labels are 0-150, 150-300, and 300+. The exporter
    later transforms them into key tokens 0_150, 150_300, and 300plus.

    The bins are left-closed and right-open, as in the Java implementation:
    a distance exactly equal to 150 km belongs to 150-300, and a distance
    exactly equal to 300 km belongs to 300+.
    """
    thresholds = [float(t) for t in thresholds_km]
    if not thresholds:
        raise ValueError("At least one distance threshold is required.")
    if any(t <= 0 for t in thresholds):
        raise ValueError("Distance thresholds must be strictly positive.")
    if thresholds != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError("Distance thresholds must be unique and increasing.")

    bins = [0.0] + thresholds + [np.inf]
    labels = []
    lower = 0.0
    for upper in thresholds:
        labels.append(f"{format_distance_boundary(lower)}-{format_distance_boundary(upper)}")
        lower = upper
    labels.append(f"{format_distance_boundary(lower)}+")

    return pd.cut(
        distance_km,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype(str)


def center_by_mode(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    availability_col: str,
    weight_col: str,
) -> float:
    mask = df[availability_col] == 1
    if not mask.any():
        raise ValueError(f"No available observations for {target_col}.")
    m = weighted_mean(df.loc[mask, source_col], df.loc[mask, weight_col])
    df[target_col] = df[source_col] - m
    return m



def _median_positive_or_default(
    df: pd.DataFrame,
    value_col: str,
    availability_col: str,
    default: Optional[float] = None,
) -> float:
    """Return a positive median, optionally falling back to a training default."""
    if default is not None and np.isfinite(default) and default > 0:
        return float(default)

    available = pd.to_numeric(df.loc[df[availability_col] == 1, value_col], errors="coerce")
    available = available[np.isfinite(available) & (available > 0)]
    if available.empty:
        raise ValueError(
            f"No positive available observations for {value_col}. "
            "This is only allowed when a training-set default is supplied."
        )
    return float(available.median())


def apply_or_compute_centering(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    availability_col: str,
    weight_col: str,
    centering_input: Optional[Dict[str, float]],
    centering_key: str,
) -> float:
    """Apply a supplied centering constant or compute it from the current dataframe."""
    if centering_input is not None:
        if centering_key not in centering_input:
            raise KeyError(f"Missing centering constant {centering_key!r}.")
        m = float(centering_input[centering_key])
        df[target_col] = df[source_col] - m
        return m
    return center_by_mode(df, source_col, target_col, availability_col, weight_col)


def preprocess_dataset(
    df: pd.DataFrame,
    dataset_id: int,
    dataset_label: str,
    weight_type: str,
    weight_alpha: Optional[float],
    dist_thresholds_km: Sequence[float],
    centering_input: Optional[Dict[str, float]] = None,
    safe_defaults_input: Optional[Dict[str, float]] = None,
    sample_label: str = "full",
) -> Tuple[pd.DataFrame, List[int], Dict[str, float], List[str], pd.DataFrame, Dict[str, float]]:
    """Preprocess one dataset and return model-ready dataframe and diagnostics.

    When train/test validation is used, the training set is preprocessed first.
    Its centering constants and safe fallback medians are then passed to the test
    set through centering_input and safe_defaults_input. This avoids leaking test
    information into the estimated model and keeps the test transformation exactly
    aligned with the training transformation.
    """
    check_required_columns(df)
    df = df.copy()

    for mode in (1, 2, 3):
        df[f"av{mode}"] = (
            df[f"cost{mode}"].notna()
            & df[f"length{mode}"].notna()
            & df[f"vduration{mode}"].notna()
            & (df[f"cost{mode}"] > 0)
            & (df[f"length{mode}"] > 0)
            & (df[f"vduration{mode}"] > 0)
        ).astype(int)

    road_bad = int((df["av1"] == 0).sum())
    if road_bad > 0:
        log(f"Warning: dropping {road_bad:,} rows where road is unavailable/invalid.")
        df = df.loc[df["av1"] == 1].copy()

    for mode in (1, 2, 3):
        bad = (df[f"av{mode}"] == 0) & (df[f"qty{mode}"] > 0)
        if bad.any():
            raise ValueError(
                f"Found {bad.sum()} rows with qty{mode} > 0 while mode {mode} is unavailable."
            )

    df["total_qty"] = df[["qty1", "qty2", "qty3"]].sum(axis=1)
    df = df.loc[df["total_qty"] > 0].copy()
    if df.empty:
        raise ValueError("No rows with strictly positive total flow.")

    df["dataset_id"] = int(dataset_id)
    df["dataset_label"] = dataset_label
    df["sample"] = sample_label
    df["grp"] = df["grp"].astype(int)
    groups = sorted(df["grp"].unique().tolist())

    for mode in (1, 2, 3):
        df[f"share{mode}"] = df[f"qty{mode}"] / df["total_qty"]

    df["weight"] = make_weight(df["total_qty"], weight_type, weight_alpha)
    df["dref_km"] = df["length1"].astype(float)
    df["dref_scaled"] = df["dref_km"] / DIST_SCALE

    safe_defaults: Dict[str, float] = dict(safe_defaults_input or {})
    for base in ("cost", "length", "vduration"):
        for mode in (1, 2, 3):
            col = f"{base}{mode}"
            default = _median_positive_or_default(
                df=df,
                value_col=col,
                availability_col=f"av{mode}",
                default=safe_defaults.get(col),
            )
            safe_defaults[col] = default
            df[f"{col}_safe"] = positive_fill(df[col], default=default)

    df["ln_cost_ratio_iww"] = np.log(df["cost2_safe"] / df["cost1_safe"])
    df["ln_cost_ratio_rail"] = np.log(df["cost3_safe"] / df["cost1_safe"])
    df["ln_move_ratio_iww"] = np.log(df["vduration2_safe"] / df["vduration1_safe"])
    df["ln_move_ratio_rail"] = np.log(df["vduration3_safe"] / df["vduration1_safe"])

    threshold_scaled = [t / DIST_SCALE for t in dist_thresholds_km]
    raw_segments = piecewise_segments_scaled(df["dref_scaled"], threshold_scaled)
    df = pd.concat([df, raw_segments], axis=1)
    raw_segment_cols = list(raw_segments.columns)

    centering: Dict[str, float] = {}
    centering["ln_cost_ratio_iww"] = apply_or_compute_centering(
        df, "ln_cost_ratio_iww", "c_ln_cost_ratio_iww", "av2", "weight",
        centering_input, "ln_cost_ratio_iww"
    )
    centering["ln_move_ratio_iww"] = apply_or_compute_centering(
        df, "ln_move_ratio_iww", "c_ln_move_ratio_iww", "av2", "weight",
        centering_input, "ln_move_ratio_iww"
    )
    centering["ln_cost_ratio_rail"] = apply_or_compute_centering(
        df, "ln_cost_ratio_rail", "c_ln_cost_ratio_rail", "av3", "weight",
        centering_input, "ln_cost_ratio_rail"
    )
    centering["ln_move_ratio_rail"] = apply_or_compute_centering(
        df, "ln_move_ratio_rail", "c_ln_move_ratio_rail", "av3", "weight",
        centering_input, "ln_move_ratio_rail"
    )

    centered_segment_cols: List[str] = []
    for raw_col in raw_segment_cols:
        idx = raw_col.split("_")[-1]
        iww_col = f"c_dseg_iww_{idx}"
        rail_col = f"c_dseg_rail_{idx}"
        centering[iww_col] = apply_or_compute_centering(
            df, raw_col, iww_col, "av2", "weight", centering_input, iww_col
        )
        centering[rail_col] = apply_or_compute_centering(
            df, raw_col, rail_col, "av3", "weight", centering_input, rail_col
        )
        centered_segment_cols.extend([iww_col, rail_col])

    # Diagnostic-only variables if present.
    if {"fduration1", "fduration2", "fduration3"}.issubset(df.columns):
        eps = 1e-6
        for mode in (1, 2, 3):
            col = f"fduration{mode}"
            available = df.loc[df[f"av{mode}"] == 1, col]
            if not available.empty:
                default = float(available.where(available > 0).median())
                if not np.isfinite(default) or default <= 0:
                    default = eps
            else:
                default = eps
            df[f"{col}_safe_diag"] = positive_fill(df[col], default=default)
        df["ln_hand_ratio_iww_diag"] = np.log(
            (df["fduration2_safe_diag"] + eps) / (df["fduration1_safe_diag"] + eps)
        )
        df["ln_hand_ratio_rail_diag"] = np.log(
            (df["fduration3_safe_diag"] + eps) / (df["fduration1_safe_diag"] + eps)
        )

    df["ln_detour_ratio_iww_diag"] = np.log(df["length2_safe"] / df["length1_safe"])
    df["ln_detour_ratio_rail_diag"] = np.log(df["length3_safe"] / df["length1_safe"])

    df["distance_band"] = make_distance_band(df["dref_km"], dist_thresholds_km)
    df["iww_availability"] = np.where(df["av2"] == 1, "IWW available", "IWW unavailable")

    required_model_cols = [
        "weight", "grp", "share1", "share2", "share3",
        "av1", "av2", "av3",
        "c_ln_cost_ratio_iww", "c_ln_move_ratio_iww",
        "c_ln_cost_ratio_rail", "c_ln_move_ratio_rail",
    ] + centered_segment_cols

    for col in required_model_cols:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values found in column {col}.")
        if col.startswith("share") and ((df[col] < 0).any() or (df[col] > 1).any()):
            raise ValueError(f"Invalid shares found in column {col}.")
        if col == "weight" and (df[col] <= 0).any():
            raise ValueError("Weights must be strictly positive.")

    dataset_summary = pd.DataFrame([{
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "sample": sample_label,
        "rows": int(len(df)),
        "total_tonnes": float(df["total_qty"].sum()),
        "iww_available_rows": int(df["av2"].sum()),
        "rail_available_rows": int(df["av3"].sum()),
        "iww_availability_rate": float(df["av2"].mean()),
        "rail_availability_rate": float(df["av3"].mean()),
        "observed_iww_tonnes": float(df["qty2"].sum()),
        "observed_rail_tonnes": float(df["qty3"].sum()),
        "observed_road_tonnes": float(df["qty1"].sum()),
        "observed_iww_share": float(df["qty2"].sum() / df["total_qty"].sum()),
        "observed_rail_share": float(df["qty3"].sum() / df["total_qty"].sum()),
        "observed_road_share": float(df["qty1"].sum() / df["total_qty"].sum()),
    }])

    print_preprocessing_diagnostics(
        df=df,
        dataset_id=dataset_id,
        dataset_label=dataset_label,
        sample_label=sample_label,
        groups=groups,
        weight_type=weight_type,
        weight_alpha=weight_alpha,
        dist_thresholds_km=dist_thresholds_km,
        centering=centering,
        raw_segment_cols=raw_segment_cols,
        dataset_summary=dataset_summary,
    )

    return df, groups, centering, raw_segment_cols, dataset_summary, safe_defaults

def print_preprocessing_diagnostics(
    df: pd.DataFrame,
    dataset_id: int,
    dataset_label: str,
    sample_label: str,
    groups: List[int],
    weight_type: str,
    weight_alpha: Optional[float],
    dist_thresholds_km: Sequence[float],
    centering: Dict[str, float],
    raw_segment_cols: List[str],
    dataset_summary: pd.DataFrame,
) -> None:
    log("\n==============================")
    if sample_label == "full":
        log(f"Preprocessing diagnostics: dataset {dataset_id} ({dataset_label})")
    else:
        log(f"Preprocessing diagnostics: dataset {dataset_id} ({dataset_label}), sample={sample_label}")
    log("==============================")
    log(f"Rows retained: {len(df):,}")
    log(f"Groups present: {groups}")
    if weight_type == "power":
        log(f"Weight type: {weight_type} (alpha={weight_alpha})")
    else:
        log(f"Weight type: {weight_type}")
    log(f"Distance thresholds in km: {list(dist_thresholds_km)}")

    log("\nDataset structure:")
    log(dataset_summary.to_string(index=False))

    log("\nAvailability counts:")
    for mode, label in [(1, "road"), (2, "IWW"), (3, "rail")]:
        log(f"  mode {mode} ({label}): {int(df[f'av{mode}'].sum()):,} available rows")

    centered = [
        "c_ln_cost_ratio_iww",
        "c_ln_move_ratio_iww",
        "c_ln_cost_ratio_rail",
        "c_ln_move_ratio_rail",
    ]
    log("\nWeighted correlations among centered variables:")
    log(weighted_corr(df, centered, "weight").round(3).to_string())

    log("\nCentering constants:")
    for key, value in sorted(centering.items()):
        log(f"  {key}: {value:.6f}")

    log("\nRaw distance spline segment summaries:")
    log(df[raw_segment_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())




# =============================================================================
# Train/test split helpers
# =============================================================================

def split_train_test_raw(
    df: pd.DataFrame,
    holdout_fraction: float,
    holdout_by: str,
    holdout_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split raw rows into training and test sets.

    The split is applied before centering and before Biogeme estimation. The
    training sample is used to estimate coefficients, centering constants, and
    calibration constants. The test sample is transformed with the training
    centering constants and is used only for out-of-sample validation.
    """
    if holdout_fraction <= 0:
        empty_summary = pd.DataFrame([{
            "holdout_enabled": False,
            "holdout_by": holdout_by,
            "holdout_fraction": holdout_fraction,
            "holdout_seed": holdout_seed,
            "train_rows_raw": int(len(df)),
            "test_rows_raw": 0,
        }])
        return df.copy(), df.iloc[0:0].copy(), empty_summary

    if not (0 < holdout_fraction < 1):
        raise ValueError("--holdout-fraction must be in [0,1). Use 0 to disable holdout.")

    rng = np.random.default_rng(int(holdout_seed))
    df = df.copy()

    if holdout_by == "row":
        test_mask = rng.random(len(df)) < holdout_fraction
    elif holdout_by in {"od", "od_group", "group"}:
        if holdout_by == "od":
            key_cols = ["org", "dst"]
        elif holdout_by == "od_group":
            key_cols = ["org", "dst", "grp"]
        else:
            key_cols = ["grp"]

        keys = df[key_cols].drop_duplicates().reset_index(drop=True)
        if len(keys) < 2:
            raise ValueError(f"Cannot hold out by {holdout_by}: fewer than two unique keys.")
        n_test = int(round(holdout_fraction * len(keys)))
        n_test = min(max(n_test, 1), len(keys) - 1)
        selected = np.zeros(len(keys), dtype=bool)
        selected[rng.choice(len(keys), size=n_test, replace=False)] = True
        keys["__is_test"] = selected
        df = df.merge(keys, on=key_cols, how="left")
        test_mask = df["__is_test"].fillna(False).to_numpy(dtype=bool)
        df = df.drop(columns=["__is_test"])
    else:
        raise ValueError("--holdout-by must be one of: row, od, od_group, group")

    if test_mask.all() or (~test_mask).all():
        raise ValueError("Holdout split produced an empty train or test sample.")

    train = df.loc[~test_mask].copy()
    test = df.loc[test_mask].copy()

    summary = pd.DataFrame([{
        "holdout_enabled": True,
        "holdout_by": holdout_by,
        "holdout_fraction": holdout_fraction,
        "holdout_seed": holdout_seed,
        "train_rows_raw": int(len(train)),
        "test_rows_raw": int(len(test)),
        "train_share_raw": float(len(train) / len(df)),
        "test_share_raw": float(len(test) / len(df)),
    }])

    log("\nHoldout split summary:")
    log(summary.to_string(index=False))
    return train, test, summary


def make_biogeme_dataframe(df: pd.DataFrame, raw_segment_cols: List[str]) -> pd.DataFrame:
    """Return only the numeric formula columns passed to Biogeme."""
    keep_cols = [
        "share1", "share2", "share3",
        "av1", "av2", "av3",
        "grp", "weight",
        "c_ln_cost_ratio_iww",
        "c_ln_move_ratio_iww",
        "c_ln_cost_ratio_rail",
        "c_ln_move_ratio_rail",
    ]

    for raw_col in raw_segment_cols:
        idx = raw_col.split("_")[-1]
        keep_cols.extend([f"c_dseg_iww_{idx}", f"c_dseg_rail_{idx}"])

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns required by Biogeme: {missing}")

    out = df[keep_cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="raise")

    bad = [c for c in out.columns if not np.isfinite(out[c].to_numpy(dtype=float)).all()]
    if bad:
        raise ValueError(f"Non-finite columns in Biogeme dataframe: {bad}")

    return out


# =============================================================================
# Biogeme model
# =============================================================================

def build_group_specific_expression(
    grp_var: Variable,
    groups: List[int],
    param_prefix: str,
    default_start: float,
    lower: Optional[float],
    upper: Optional[float],
    fixed: int = 0,
):
    betas = {
        g: Beta(f"{param_prefix}_g{g}", default_start, lower, upper, fixed)
        for g in groups
    }
    expr = sum(beta * (grp_var == g) for g, beta in betas.items())
    return expr, betas


def build_biogeme_model(
    df_model: pd.DataFrame,
    groups: List[int],
    model_name: str,
    weight_type: str,
    weight_alpha: Optional[float],
    raw_segment_cols: List[str],
    include_rail_cost: bool,
    generate_reports: bool,
    max_iterations: int,
) -> BIOGEME:
    df_biogeme = make_biogeme_dataframe(df_model, raw_segment_cols)
    #database = Database("freight_modal_choice_framework_dataset_specific", df_biogeme)
    database = Database("CalibratedODModalChoice", df_biogeme)
    

    SHARE1 = Variable("share1")
    SHARE2 = Variable("share2")
    SHARE3 = Variable("share3")
    AV1 = Variable("av1")
    AV2 = Variable("av2")
    AV3 = Variable("av3")

    C_COST_IWW = Variable("c_ln_cost_ratio_iww")
    C_MOVE_IWW = Variable("c_ln_move_ratio_iww")
    C_COST_RAIL = Variable("c_ln_cost_ratio_rail")
    C_MOVE_RAIL = Variable("c_ln_move_ratio_rail")
    WEIGHT = Variable("weight")
    GRP = Variable("grp")

    ASC_IWW_G, _ = build_group_specific_expression(
        GRP, groups, "ASC_IWW", 0.0, None, None, 0
    )
    ASC_RAIL_G, _ = build_group_specific_expression(
        GRP, groups, "ASC_RAIL", 0.0, None, None, 0
    )

    B_COST_IWW = Beta("B_COST_IWW", -1.0, None, 0.0, 0)
    B_MOVE_IWW = Beta("B_MOVE_IWW", -0.5, None, 0.0, 0)
    B_MOVE_RAIL = Beta("B_MOVE_RAIL", -0.5, None, 0.0, 0)
    B_RAIL_IWW_AVAILABLE = Beta("B_RAIL_IWW_AVAILABLE", 0.1, None, None, 0)

    if include_rail_cost:
        B_COST_RAIL = Beta("B_COST_RAIL", -0.1, None, 0.0, 0)
    else:
        B_COST_RAIL = 0

    dist_iww = 0
    dist_rail = 0
    for i in range(1, len(raw_segment_cols) + 1):
        seg_iww = Variable(f"c_dseg_iww_{i}")
        seg_rail = Variable(f"c_dseg_rail_{i}")
        dist_iww += Beta(f"B_DIST_IWW_{i}", 0.0, None, None, 0) * seg_iww
        dist_rail += Beta(f"B_DIST_RAIL_{i}", 0.0, None, None, 0) * seg_rail

    V1_base = 0
    V2_base = (
        ASC_IWW_G
        + B_COST_IWW * C_COST_IWW
        + B_MOVE_IWW * C_MOVE_IWW
        + dist_iww
    )
    V3_base = (
        ASC_RAIL_G
        + B_COST_RAIL * C_COST_RAIL
        + B_MOVE_RAIL * C_MOVE_RAIL
        + dist_rail
        + B_RAIL_IWW_AVAILABLE * AV2
    )

    V1 = AV1 * V1_base + (1 - AV1) * UNAVAILABLE_UTILITY
    V2 = AV2 * V2_base + (1 - AV2) * UNAVAILABLE_UTILITY
    V3 = AV3 * V3_base + (1 - AV3) * UNAVAILABLE_UTILITY
    V = {1: V1, 2: V2, 3: V3}

    row_log_like = (
        SHARE1 * loglogit(V, None, 1)
        + SHARE2 * loglogit(V, None, 2)
        + SHARE3 * loglogit(V, None, 3)
    )

    if weight_type == "power":
        weight_note = f"{weight_type}, alpha = {weight_alpha}"
    else:
        weight_note = weight_type

    user_notes = (
        "Sequential general freight modal-choice framework. "
        "Road utility normalized to zero. IWW uses centered cost and moving-time ratios. "
        "Rail uses centered moving-time ratio, distance spline, and rail market correction "
        "B_RAIL_IWW_AVAILABLE * I(IWW available). "
        "Rail cost included only if requested. "
        f"Weight type = {weight_note}."
    )

    biogeme = BIOGEME(
        database,
        {"log_like": row_log_like, "weight": WEIGHT},
        user_notes=user_notes,
        generate_html=generate_reports,
        generate_yaml=generate_reports,
        optimization_algorithm="simple_bounds_BFGS",
        max_iterations=max_iterations,
        second_derivatives=0.0,
        tolerance=1.0e-5,
        steptol=1.0e-9,
        numerically_safe=True,
        save_iterations=True,
    )
    biogeme.model_name = model_name
    return biogeme


# =============================================================================
# Prediction and validation
# =============================================================================

def extract_parameter_values(params: pd.DataFrame) -> Dict[str, float]:
    if "Name" not in params.columns or "Value" not in params.columns:
        raise KeyError(f"Expected columns 'Name' and 'Value'. Found {list(params.columns)}")
    return {str(row["Name"]): float(row["Value"]) for _, row in params.iterrows()}


def param_value(beta: Dict[str, float], name: str, default: float = 0.0) -> float:
    return float(beta.get(name, default))


def compute_base_utilities(
    df_model: pd.DataFrame,
    params: pd.DataFrame,
    raw_segment_cols: List[str],
    include_rail_cost: bool,
) -> pd.DataFrame:
    """
    Compute systematic utilities before any base-year calibration constants.
    """
    beta = extract_parameter_values(params)
    df = df_model.copy()

    n = len(df)
    v1 = np.zeros(n, dtype=float)
    v2 = np.zeros(n, dtype=float)
    v3 = np.zeros(n, dtype=float)

    for g in sorted(df["grp"].unique()):
        mask = df["grp"].to_numpy() == g
        v2[mask] += param_value(beta, f"ASC_IWW_g{int(g)}")
        v3[mask] += param_value(beta, f"ASC_RAIL_g{int(g)}")

    v2 += param_value(beta, "B_COST_IWW") * df["c_ln_cost_ratio_iww"].to_numpy(dtype=float)
    v2 += param_value(beta, "B_MOVE_IWW") * df["c_ln_move_ratio_iww"].to_numpy(dtype=float)

    if include_rail_cost:
        v3 += param_value(beta, "B_COST_RAIL") * df["c_ln_cost_ratio_rail"].to_numpy(dtype=float)
    v3 += param_value(beta, "B_MOVE_RAIL") * df["c_ln_move_ratio_rail"].to_numpy(dtype=float)
    v3 += param_value(beta, "B_RAIL_IWW_AVAILABLE") * df["av2"].to_numpy(dtype=float)

    for i in range(1, len(raw_segment_cols) + 1):
        v2 += param_value(beta, f"B_DIST_IWW_{i}") * df[f"c_dseg_iww_{i}"].to_numpy(dtype=float)
        v3 += param_value(beta, f"B_DIST_RAIL_{i}") * df[f"c_dseg_rail_{i}"].to_numpy(dtype=float)

    df["utility1_base"] = np.where(df["av1"].to_numpy(dtype=int) == 1, v1, UNAVAILABLE_UTILITY)
    df["utility2_base"] = np.where(df["av2"].to_numpy(dtype=int) == 1, v2, UNAVAILABLE_UTILITY)
    df["utility3_base"] = np.where(df["av3"].to_numpy(dtype=int) == 1, v3, UNAVAILABLE_UTILITY)
    return df


def add_probabilities_from_utilities(
    df_util: pd.DataFrame,
    utility1_col: str = "utility1_base",
    utility2_col: str = "utility2_base",
    utility3_col: str = "utility3_base",
) -> pd.DataFrame:
    """
    Add MNL probabilities and predicted modal tonnes from three utility columns.
    """
    df = df_util.copy()
    utilities = np.column_stack([
        df[utility1_col].to_numpy(dtype=float),
        df[utility2_col].to_numpy(dtype=float),
        df[utility3_col].to_numpy(dtype=float),
    ])

    max_u = np.max(utilities, axis=1)
    exp_u = np.exp(utilities - max_u[:, None])
    probs = exp_u / exp_u.sum(axis=1)[:, None]

    df["prob1"] = probs[:, 0]
    df["prob2"] = probs[:, 1]
    df["prob3"] = probs[:, 2]

    for mode in (1, 2, 3):
        df[f"pred_qty{mode}"] = df[f"prob{mode}"] * df["total_qty"]
        df[f"share_error{mode}"] = df[f"prob{mode}"] - df[f"share{mode}"]
        df[f"abs_share_error{mode}"] = df[f"share_error{mode}"].abs()
        df[f"sq_share_error{mode}"] = df[f"share_error{mode}"] ** 2

    return df


def simulate_probabilities(
    df_model: pd.DataFrame,
    params: pd.DataFrame,
    raw_segment_cols: List[str],
    include_rail_cost: bool,
) -> pd.DataFrame:
    """
    Simulate probabilities without post-estimation base-year calibration.
    """
    df_util = compute_base_utilities(df_model, params, raw_segment_cols, include_rail_cost)
    return add_probabilities_from_utilities(df_util)


def mode_long_frame(df_pred: pd.DataFrame) -> pd.DataFrame:
    mode_labels = {1: "road", 2: "IWW", 3: "rail"}
    frames = []
    base_cols = [
        "dataset_id", "dataset_label", "org", "dst", "grp", "dref_km",
        "distance_band", "iww_availability", "total_qty", "weight", "av2", "av3",
    ]
    base_cols = [c for c in base_cols if c in df_pred.columns]

    for mode in (1, 2, 3):
        tmp = df_pred[base_cols].copy()
        tmp["mode"] = mode_labels[mode]
        tmp["mode_id"] = mode
        tmp["observed_share"] = df_pred[f"share{mode}"]
        tmp["predicted_share"] = df_pred[f"prob{mode}"]
        tmp["observed_tonnes"] = df_pred[f"qty{mode}"]
        tmp["predicted_tonnes"] = df_pred[f"pred_qty{mode}"]
        tmp["share_error"] = df_pred[f"share_error{mode}"]
        tmp["abs_share_error"] = df_pred[f"abs_share_error{mode}"]
        tmp["sq_share_error"] = df_pred[f"sq_share_error{mode}"]
        frames.append(tmp)

    return pd.concat(frames, ignore_index=True)


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return float(np.average(v[ok], weights=w[ok]))


def aggregate_validation(long_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    group_keys = group_cols + ["mode"]

    for keys, sub in long_df.groupby(group_keys, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_keys, keys))

        total_tonnes = float(sub["total_qty"].sum())
        obs_tonnes = float(sub["observed_tonnes"].sum())
        pred_tonnes = float(sub["predicted_tonnes"].sum())
        tonnes_error = pred_tonnes - obs_tonnes

        mae_share = weighted_average(sub["abs_share_error"], sub["weight"])
        mse_share = weighted_average(sub["sq_share_error"], sub["weight"])
        rmse_share = float(np.sqrt(mse_share)) if np.isfinite(mse_share) else np.nan
        mean_share_error = weighted_average(sub["share_error"], sub["weight"])

        rows.append({
            **key_dict,
            "n_rows": int(len(sub)),
            "total_tonnes": total_tonnes,
            "observed_tonnes": obs_tonnes,
            "predicted_tonnes": pred_tonnes,
            "tonnes_error": tonnes_error,
            "tonnes_error_pct_of_observed": tonnes_error / obs_tonnes if obs_tonnes > 0 else np.nan,
            "observed_share_of_total": obs_tonnes / total_tonnes if total_tonnes > 0 else np.nan,
            "predicted_share_of_total": pred_tonnes / total_tonnes if total_tonnes > 0 else np.nan,
            "share_point_error": tonnes_error / total_tonnes if total_tonnes > 0 else np.nan,
            "weighted_mean_share_error": mean_share_error,
            "weighted_mae_share": mae_share,
            "weighted_rmse_share": rmse_share,
        })

    return pd.DataFrame(rows)




def calibration_key_columns(calibration_level: str) -> List[str]:
    """
    Return dataframe columns that define the base-year calibration cells.

    Calibration constants are always mode-specific for IWW and rail. The level
    controls the additional dimensions used to segment the constants.
    """
    mapping = {
        "none": [],
        "mode": [],
        "mode_group": ["grp"],
        "mode_distance": ["distance_band"],
        "mode_group_distance": ["grp", "distance_band"],
        "mode_iww_availability": ["iww_availability"],
        "mode_group_iww_availability": ["grp", "iww_availability"],
        "mode_od": ["org", "dst"],
        "mode_od_group": ["org", "dst", "grp"],
    }
    if calibration_level not in mapping:
        raise ValueError(
            f"Unknown calibration level {calibration_level!r}. "
            f"Expected one of {sorted(mapping)}."
        )
    return mapping[calibration_level]


def _make_calibration_ids(df: pd.DataFrame, key_cols: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Factorize calibration cells and return integer cell ids plus key dataframe.
    """
    if not key_cols:
        ids = np.zeros(len(df), dtype=np.int64)
        key_df = pd.DataFrame({"calibration_id": [0]})
        return ids, key_df

    key_frame = df[key_cols].copy()
    # Factorize a MultiIndex for compact integer ids. pd.factorize drops
    # MultiIndex names, so we rebuild the key dataframe with explicit columns.
    mi = pd.MultiIndex.from_frame(key_frame)
    codes, uniques = pd.factorize(mi, sort=False)
    key_df = pd.DataFrame(list(uniques), columns=key_cols)
    key_df.insert(0, "calibration_id", np.arange(len(key_df), dtype=np.int64))
    return codes.astype(np.int64), key_df


def apply_base_year_calibration(
    df_model: pd.DataFrame,
    params: pd.DataFrame,
    raw_segment_cols: List[str],
    include_rail_cost: bool,
    calibration_level: str,
    max_iterations: int,
    damping: float,
    tolerance: float,
    epsilon_tonnes: float,
    max_step: float,
    max_abs_constant: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply additive base-year calibration constants to IWW and rail utilities.

    Constants are calibrated iteratively by cell using:
        delta_m(cell) += damping * log((observed_m + eps) / (predicted_m + eps))

    Road remains the normalized base alternative. Adjusting IWW and rail
    constants is sufficient to shift probability mass away from or toward road.

    The constants are not behavioral parameters. They are base-year correction
    factors designed to improve matrix reproduction.
    """
    if calibration_level == "none":
        df_uncal = simulate_probabilities(df_model, params, raw_segment_cols, include_rail_cost)
        return df_uncal, pd.DataFrame(), pd.DataFrame()

    if not (0 < damping <= 1):
        raise ValueError("calibration_damping must be in (0, 1].")
    if max_iterations < 1:
        raise ValueError("calibration_max_iterations must be >= 1.")
    if epsilon_tonnes < 0:
        raise ValueError("calibration_epsilon_tonnes must be non-negative.")
    if max_step <= 0 or max_abs_constant <= 0:
        raise ValueError("calibration max step and max abs constant must be positive.")

    key_cols = calibration_key_columns(calibration_level)
    df_util = compute_base_utilities(df_model, params, raw_segment_cols, include_rail_cost)
    calib_id, key_df = _make_calibration_ids(df_util, key_cols)
    n_cells = int(calib_id.max()) + 1 if len(calib_id) else 0

    total_qty = df_util["total_qty"].to_numpy(dtype=float)
    obs2 = np.bincount(calib_id, weights=df_util["qty2"].to_numpy(dtype=float), minlength=n_cells)
    obs3 = np.bincount(calib_id, weights=df_util["qty3"].to_numpy(dtype=float), minlength=n_cells)

    base_u1 = df_util["utility1_base"].to_numpy(dtype=float)
    base_u2 = df_util["utility2_base"].to_numpy(dtype=float)
    base_u3 = df_util["utility3_base"].to_numpy(dtype=float)

    c2 = np.zeros(n_cells, dtype=float)
    c3 = np.zeros(n_cells, dtype=float)

    history_records = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        u1 = base_u1
        u2 = base_u2 + c2[calib_id]
        u3 = base_u3 + c3[calib_id]

        util = np.column_stack([u1, u2, u3])
        max_u = np.max(util, axis=1)
        exp_u = np.exp(util - max_u[:, None])
        prob = exp_u / exp_u.sum(axis=1)[:, None]

        pred2_row = prob[:, 1] * total_qty
        pred3_row = prob[:, 2] * total_qty

        pred2 = np.bincount(calib_id, weights=pred2_row, minlength=n_cells)
        pred3 = np.bincount(calib_id, weights=pred3_row, minlength=n_cells)

        with np.errstate(divide="ignore", invalid="ignore"):
            step2_raw = np.log((obs2 + epsilon_tonnes) / (pred2 + epsilon_tonnes))
            step3_raw = np.log((obs3 + epsilon_tonnes) / (pred3 + epsilon_tonnes))

        step2_raw = np.nan_to_num(step2_raw, nan=0.0, posinf=max_step, neginf=-max_step)
        step3_raw = np.nan_to_num(step3_raw, nan=0.0, posinf=max_step, neginf=-max_step)

        step2 = np.clip(step2_raw, -max_step, max_step)
        step3 = np.clip(step3_raw, -max_step, max_step)

        # If both observed and predicted are essentially zero, do not move.
        zero2 = (obs2 <= epsilon_tonnes) & (pred2 <= epsilon_tonnes)
        zero3 = (obs3 <= epsilon_tonnes) & (pred3 <= epsilon_tonnes)
        step2[zero2] = 0.0
        step3[zero3] = 0.0

        c2 = np.clip(c2 + damping * step2, -max_abs_constant, max_abs_constant)
        c3 = np.clip(c3 + damping * step3, -max_abs_constant, max_abs_constant)

        max_abs_step = float(max(np.max(np.abs(damping * step2)), np.max(np.abs(damping * step3))))
        total_abs_error = float(np.sum(np.abs(pred2 - obs2)) + np.sum(np.abs(pred3 - obs3)))
        total_obs_nonroad = float(np.sum(obs2) + np.sum(obs3))
        nonroad_wape = total_abs_error / total_obs_nonroad if total_obs_nonroad > 0 else np.nan

        history_records.append({
            "iteration": iteration,
            "max_abs_constant_step": max_abs_step,
            "nonroad_total_abs_error": total_abs_error,
            "nonroad_wape_by_calibration_cells": nonroad_wape,
            "total_observed_iww": float(np.sum(obs2)),
            "total_predicted_iww": float(np.sum(pred2)),
            "total_observed_rail": float(np.sum(obs3)),
            "total_predicted_rail": float(np.sum(pred3)),
        })

        if iteration == 1 or iteration % 10 == 0:
            log(
                f"  calibration iter {iteration:03d}: "
                f"max step={max_abs_step:.6g}, non-road WAPE={nonroad_wape:.6g}"
            )

        if max_abs_step < tolerance:
            converged = True
            log(f"  calibration converged after {iteration} iterations.")
            break

    if not converged:
        log(
            f"  calibration stopped after {max_iterations} iterations "
            f"without reaching tolerance {tolerance}."
        )

    # Final calibrated predictions.
    df_cal = df_util.copy()
    df_cal["calibration_id"] = calib_id
    df_cal["calib_const_iww"] = c2[calib_id]
    df_cal["calib_const_rail"] = c3[calib_id]
    df_cal["utility1_calibrated"] = base_u1
    df_cal["utility2_calibrated"] = base_u2 + df_cal["calib_const_iww"].to_numpy(dtype=float)
    df_cal["utility3_calibrated"] = base_u3 + df_cal["calib_const_rail"].to_numpy(dtype=float)
    df_cal = add_probabilities_from_utilities(
        df_cal,
        utility1_col="utility1_calibrated",
        utility2_col="utility2_calibrated",
        utility3_col="utility3_calibrated",
    )

    pred2_final = np.bincount(calib_id, weights=df_cal["pred_qty2"].to_numpy(dtype=float), minlength=n_cells)
    pred3_final = np.bincount(calib_id, weights=df_cal["pred_qty3"].to_numpy(dtype=float), minlength=n_cells)

    constants = key_df.copy()
    constants["calibration_level"] = calibration_level
    constants["calib_const_iww"] = c2
    constants["observed_iww_tonnes"] = obs2
    constants["predicted_iww_tonnes_after_calibration"] = pred2_final
    constants["iww_tonnes_error_after_calibration"] = pred2_final - obs2
    constants["calib_const_rail"] = c3
    constants["observed_rail_tonnes"] = obs3
    constants["predicted_rail_tonnes_after_calibration"] = pred3_final
    constants["rail_tonnes_error_after_calibration"] = pred3_final - obs3

    history = pd.DataFrame(history_records)
    return df_cal, constants, history



def aggregate_matrix_cell_metrics(long_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Aggregate OD-matrix cell-level validation metrics.

    Each row in long_df is an OD-commodity-mode cell. The metrics compare:
        observed_tonnes = observed modal OD matrix cell
        predicted_tonnes = model-estimated modal OD matrix cell

    MAPE/weighted MAPE are computed only on cells with observed_tonnes > 0,
    because percentage error is undefined for zero observed cells. WAPE is also
    reported because it is robust to many zero cells:
        WAPE = sum(abs(predicted - observed)) / sum(observed)
    """
    rows = []
    group_keys = group_cols + ["mode"]

    for keys, sub in long_df.groupby(group_keys, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_keys, keys))

        obs = sub["observed_tonnes"].to_numpy(dtype=float)
        pred = sub["predicted_tonnes"].to_numpy(dtype=float)
        w = sub["weight"].to_numpy(dtype=float)

        err = pred - obs
        abs_err = np.abs(err)
        sq_err = err ** 2

        ok_w = np.isfinite(w) & (w > 0)
        if ok_w.any():
            weighted_mean_error = float(np.average(err[ok_w], weights=w[ok_w]))
            weighted_mae = float(np.average(abs_err[ok_w], weights=w[ok_w]))
            weighted_rmse = float(np.sqrt(np.average(sq_err[ok_w], weights=w[ok_w])))
        else:
            weighted_mean_error = np.nan
            weighted_mae = np.nan
            weighted_rmse = np.nan

        positive_obs = obs > 0
        if positive_obs.any():
            ape = abs_err[positive_obs] / obs[positive_obs]
            mape = float(np.mean(ape))
            median_ape = float(np.median(ape))
            ok_w_pos = ok_w[positive_obs]
            if ok_w_pos.any():
                weighted_mape = float(np.average(ape[ok_w_pos], weights=w[positive_obs][ok_w_pos]))
            else:
                weighted_mape = np.nan
        else:
            mape = np.nan
            median_ape = np.nan
            weighted_mape = np.nan

        obs_sum = float(np.sum(obs))
        pred_sum = float(np.sum(pred))
        err_sum = float(np.sum(err))
        abs_err_sum = float(np.sum(abs_err))

        rows.append({
            **key_dict,
            "n_cells": int(len(sub)),
            "n_positive_observed_cells": int(positive_obs.sum()),
            "observed_tonnes": obs_sum,
            "predicted_tonnes": pred_sum,
            "tonnes_error": err_sum,
            "tonnes_abs_error": abs_err_sum,
            "mean_error_tonnes": float(np.mean(err)),
            "mae_tonnes": float(np.mean(abs_err)),
            "rmse_tonnes": float(np.sqrt(np.mean(sq_err))),
            "weighted_mean_error_tonnes": weighted_mean_error,
            "weighted_mae_tonnes": weighted_mae,
            "weighted_rmse_tonnes": weighted_rmse,
            "mape_nonzero_observed": mape,
            "median_ape_nonzero_observed": median_ape,
            "weighted_mape_nonzero_observed": weighted_mape,
            "wape": abs_err_sum / obs_sum if obs_sum > 0 else np.nan,
            "bias_pct_of_observed": err_sum / obs_sum if obs_sum > 0 else np.nan,
            "predicted_to_observed_ratio": pred_sum / obs_sum if obs_sum > 0 else np.nan,
        })

    return pd.DataFrame(rows)



def _build_validation_tables(long_df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Build aggregate-share validation and OD-matrix cell validation tables.
    """
    reports = {
        "overall_by_mode": aggregate_validation(long_df, []),
        "by_group_mode": aggregate_validation(long_df, ["grp"]),
        "by_distance_band_mode": aggregate_validation(long_df, ["distance_band"]),
        "by_iww_availability_mode": aggregate_validation(long_df, ["iww_availability"]),
        "by_group_distance_band_mode": aggregate_validation(long_df, ["grp", "distance_band"]),
    }

    matrix_reports = {
        "matrix_cells_overall_by_mode": aggregate_matrix_cell_metrics(long_df, []),
        "matrix_cells_by_group_mode": aggregate_matrix_cell_metrics(long_df, ["grp"]),
        "matrix_cells_by_distance_band_mode": aggregate_matrix_cell_metrics(long_df, ["distance_band"]),
        "matrix_cells_by_iww_availability_mode": aggregate_matrix_cell_metrics(long_df, ["iww_availability"]),
        "matrix_cells_by_group_distance_band_mode": aggregate_matrix_cell_metrics(long_df, ["grp", "distance_band"]),
    }

    return reports, matrix_reports


def _write_validation_tables(
    model_name: str,
    tag: str,
    reports: Dict[str, pd.DataFrame],
    matrix_reports: Dict[str, pd.DataFrame],
) -> None:
    """
    Write validation tables with a tag such as 'precalibration' or 'calibrated'.
    """
    for suffix, table in reports.items():
        out_path = f"{model_name}_validation_{tag}_{suffix}.csv"
        table.to_csv(out_path, index=False)
        log(f"Saved validation table to: {out_path}")

    for suffix, table in matrix_reports.items():
        out_path = f"{model_name}_validation_{tag}_{suffix}.csv"
        table.to_csv(out_path, index=False)
        log(f"Saved matrix-cell validation table to: {out_path}")


def _print_validation_summary(
    title: str,
    reports: Dict[str, pd.DataFrame],
    matrix_reports: Dict[str, pd.DataFrame],
) -> None:
    log("\n" + title)
    log("-" * len(title))

    log("Aggregate shares / aggregate tonnes, overall by mode:")
    display_cols = [
        "mode", "observed_share_of_total", "predicted_share_of_total",
        "share_point_error", "weighted_mae_share", "weighted_rmse_share",
        "observed_tonnes", "predicted_tonnes", "tonnes_error",
    ]
    log(reports["overall_by_mode"][display_cols].to_string(index=False))

    log("\nOD-matrix cell-level validation, overall by mode:")
    matrix_display_cols = [
        "mode", "n_cells", "n_positive_observed_cells",
        "observed_tonnes", "predicted_tonnes", "tonnes_error",
        "mae_tonnes", "rmse_tonnes", "weighted_mae_tonnes", "weighted_rmse_tonnes",
        "mape_nonzero_observed", "weighted_mape_nonzero_observed", "wape",
        "bias_pct_of_observed",
    ]
    log(matrix_reports["matrix_cells_overall_by_mode"][matrix_display_cols].to_string(index=False))


def write_validation_reports(
    df_model: pd.DataFrame,
    params: pd.DataFrame,
    raw_segment_cols: List[str],
    model_name: str,
    include_rail_cost: bool,
    save_row_predictions: bool,
    calibration_level: str,
    calibration_max_iterations: int,
    calibration_damping: float,
    calibration_tolerance: float,
    calibration_epsilon_tonnes: float,
    calibration_max_step: float,
    calibration_max_abs_constant: float,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Write pre-calibration validation, optionally apply base-year calibration, and
    write post-calibration validation.

    The returned reports are the final reports used by the combined comparison:
    calibrated reports if calibration_level != 'none', otherwise pre-calibration.
    """
    log("\n==============================")
    log("Validation report: observed vs predicted, in sample")
    log("==============================")

    # Always write pre-calibration diagnostics.
    df_pre = simulate_probabilities(df_model, params, raw_segment_cols, include_rail_cost)
    long_pre = mode_long_frame(df_pre)
    reports_pre, matrix_pre = _build_validation_tables(long_pre)

    _print_validation_summary("Pre-calibration validation", reports_pre, matrix_pre)
    _write_validation_tables(model_name, "precalibration", reports_pre, matrix_pre)

    calibration_constants = pd.DataFrame()

    if calibration_level == "none":
        final_df = df_pre
        final_long = long_pre
        final_reports = reports_pre
        final_matrix_reports = matrix_pre
    else:
        # The pre-calibration row-level objects can be very large. Once their
        # compact reports have been written, release them before starting
        # calibration. This matters for mode_od and mode_od_group.
        del df_pre, long_pre
        gc.collect()

        log("\n==============================")
        log(f"Base-year calibration constants: level={calibration_level}")
        log("==============================")

        df_cal, constants, history = apply_base_year_calibration(
            df_model=df_model,
            params=params,
            raw_segment_cols=raw_segment_cols,
            include_rail_cost=include_rail_cost,
            calibration_level=calibration_level,
            max_iterations=calibration_max_iterations,
            damping=calibration_damping,
            tolerance=calibration_tolerance,
            epsilon_tonnes=calibration_epsilon_tonnes,
            max_step=calibration_max_step,
            max_abs_constant=calibration_max_abs_constant,
        )

        constants_path = f"{model_name}_calibration_constants_{calibration_level}.csv"
        history_path = f"{model_name}_calibration_history_{calibration_level}.csv"
        calibration_constants = constants.copy()
        constants.to_csv(constants_path, index=False)
        history.to_csv(history_path, index=False)
        log(f"Saved calibration constants to: {constants_path}")
        log(f"Saved calibration history to: {history_path}")

        long_cal = mode_long_frame(df_cal)
        reports_cal, matrix_cal = _build_validation_tables(long_cal)

        _print_validation_summary("Post-calibration validation", reports_cal, matrix_cal)
        _write_validation_tables(model_name, "calibrated", reports_cal, matrix_cal)

        final_df = df_cal
        final_long = long_cal
        final_reports = reports_cal
        final_matrix_reports = matrix_cal

    if save_row_predictions:
        row_cols = [
            "dataset_id", "dataset_label", "org", "dst", "grp", "dref_km",
            "distance_band", "iww_availability", "total_qty",
            "share1", "share2", "share3", "prob1", "prob2", "prob3",
            "qty1", "qty2", "qty3", "pred_qty1", "pred_qty2", "pred_qty3",
            "share_error1", "share_error2", "share_error3", "av2", "av3",
            "calibration_id", "calib_const_iww", "calib_const_rail",
        ]
        row_cols = [c for c in row_cols if c in final_df.columns]
        out_rows = f"{model_name}_validation_final_row_predictions.csv"
        final_df[row_cols].to_csv(out_rows, index=False)
        log(f"Saved final row-level predictions to: {out_rows}")

        long_out = f"{model_name}_validation_final_modal_od_cells_long.csv"
        long_cols = [
            "dataset_id", "dataset_label", "org", "dst", "grp",
            "dref_km", "distance_band", "iww_availability", "mode", "mode_id",
            "total_qty", "observed_share", "predicted_share",
            "observed_tonnes", "predicted_tonnes", "share_error",
            "abs_share_error", "sq_share_error", "weight",
        ]
        long_cols = [c for c in long_cols if c in final_long.columns]
        final_long[long_cols].to_csv(long_out, index=False)
        log(f"Saved final long-form modal OD matrix cells to: {long_out}")

    # The combined comparison reports only need the compact validation tables.
    # Returning the full long-form OD cell dataframe would keep millions of rows
    # alive across datasets and can cause the operating system to kill Python
    # silently before later datasets are estimated.
    del final_long
    if "final_df" in locals():
        del final_df
    gc.collect()

    return pd.DataFrame(), final_reports, final_matrix_reports, calibration_constants




def apply_existing_calibration_constants(
    df_model: pd.DataFrame,
    params: pd.DataFrame,
    raw_segment_cols: List[str],
    include_rail_cost: bool,
    calibration_level: str,
    calibration_constants: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply calibration constants estimated on the training set to another set.

    Missing constants are interpreted as zero. This is intentional for holdout
    validation: for example, if OD pairs are held out and calibration_level is
    mode_od_group, the held-out OD x group constants do not exist and the test
    predictions fall back to the behavioral utility for those cells.
    """
    df_util = compute_base_utilities(df_model, params, raw_segment_cols, include_rail_cost)

    hit_summary = pd.DataFrame()
    if calibration_level == "none" or calibration_constants is None or calibration_constants.empty:
        df_out = add_probabilities_from_utilities(df_util)
        return df_out, hit_summary

    key_cols = calibration_key_columns(calibration_level)
    df_cal = df_util.copy()

    if not key_cols:
        c_iww = float(calibration_constants["calib_const_iww"].iloc[0])
        c_rail = float(calibration_constants["calib_const_rail"].iloc[0])
        df_cal["calib_const_iww"] = c_iww
        df_cal["calib_const_rail"] = c_rail
        hit_summary = pd.DataFrame([{
            "calibration_level": calibration_level,
            "test_rows": int(len(df_cal)),
            "matched_rows": int(len(df_cal)),
            "match_rate": 1.0,
        }])
    else:
        needed_cols = key_cols + ["calib_const_iww", "calib_const_rail"]
        missing = [c for c in needed_cols if c not in calibration_constants.columns]
        if missing:
            raise KeyError(f"Calibration constants table is missing columns: {missing}")

        lookup = calibration_constants[needed_cols].copy()
        lookup["__calibration_matched"] = 1
        before = len(df_cal)
        df_cal = df_cal.merge(lookup, on=key_cols, how="left")
        if len(df_cal) != before:
            raise RuntimeError("Calibration merge changed the number of validation rows.")

        matched = df_cal["__calibration_matched"].fillna(0).astype(int)
        hit_summary = pd.DataFrame([{
            "calibration_level": calibration_level,
            "test_rows": int(len(df_cal)),
            "matched_rows": int(matched.sum()),
            "match_rate": float(matched.mean()) if len(matched) else np.nan,
        }])
        df_cal["calib_const_iww"] = df_cal["calib_const_iww"].fillna(0.0)
        df_cal["calib_const_rail"] = df_cal["calib_const_rail"].fillna(0.0)
        df_cal = df_cal.drop(columns=["__calibration_matched"])

    df_cal["utility1_calibrated"] = df_cal["utility1_base"]
    df_cal["utility2_calibrated"] = df_cal["utility2_base"] + df_cal["calib_const_iww"]
    df_cal["utility3_calibrated"] = df_cal["utility3_base"] + df_cal["calib_const_rail"]
    df_cal = add_probabilities_from_utilities(
        df_cal,
        utility1_col="utility1_calibrated",
        utility2_col="utility2_calibrated",
        utility3_col="utility3_calibrated",
    )
    return df_cal, hit_summary


def write_holdout_validation_reports(
    df_test: pd.DataFrame,
    params: pd.DataFrame,
    raw_segment_cols: List[str],
    model_name: str,
    include_rail_cost: bool,
    calibration_level: str,
    calibration_constants: pd.DataFrame,
    save_row_predictions: bool,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """Validate the model on held-out rows not used for estimation/calibration."""
    log("\n==============================")
    log("Holdout validation report: observed vs predicted, test sample")
    log("==============================")

    df_pre = simulate_probabilities(df_test, params, raw_segment_cols, include_rail_cost)
    long_pre = mode_long_frame(df_pre)
    reports_pre, matrix_pre = _build_validation_tables(long_pre)
    _print_validation_summary("Holdout pre-calibration validation", reports_pre, matrix_pre)
    _write_validation_tables(model_name, "holdout_precalibration", reports_pre, matrix_pre)

    if calibration_level == "none":
        final_df = df_pre
        final_long = long_pre
        final_reports = reports_pre
        final_matrix_reports = matrix_pre
        hit_summary = pd.DataFrame()
    else:
        del df_pre, long_pre
        gc.collect()
        df_cal, hit_summary = apply_existing_calibration_constants(
            df_model=df_test,
            params=params,
            raw_segment_cols=raw_segment_cols,
            include_rail_cost=include_rail_cost,
            calibration_level=calibration_level,
            calibration_constants=calibration_constants,
        )
        if not hit_summary.empty:
            log("\nHoldout calibration-key match summary:")
            log(hit_summary.to_string(index=False))
            hit_summary.to_csv(f"{model_name}_validation_holdout_calibration_key_match.csv", index=False)

        long_cal = mode_long_frame(df_cal)
        reports_cal, matrix_cal = _build_validation_tables(long_cal)
        _print_validation_summary("Holdout calibrated validation", reports_cal, matrix_cal)
        _write_validation_tables(model_name, "holdout_calibrated", reports_cal, matrix_cal)
        final_df = df_cal
        final_long = long_cal
        final_reports = reports_cal
        final_matrix_reports = matrix_cal

    if save_row_predictions:
        row_cols = [
            "dataset_id", "dataset_label", "sample", "org", "dst", "grp", "dref_km",
            "distance_band", "iww_availability", "total_qty",
            "share1", "share2", "share3", "prob1", "prob2", "prob3",
            "qty1", "qty2", "qty3", "pred_qty1", "pred_qty2", "pred_qty3",
            "share_error1", "share_error2", "share_error3", "av2", "av3",
            "calib_const_iww", "calib_const_rail",
        ]
        row_cols = [c for c in row_cols if c in final_df.columns]
        out_rows = f"{model_name}_validation_holdout_final_row_predictions.csv"
        final_df[row_cols].to_csv(out_rows, index=False)
        log(f"Saved holdout row-level predictions to: {out_rows}")

        long_out = f"{model_name}_validation_holdout_final_modal_od_cells_long.csv"
        long_cols = [
            "dataset_id", "dataset_label", "sample", "org", "dst", "grp",
            "dref_km", "distance_band", "iww_availability", "mode", "mode_id",
            "total_qty", "observed_share", "predicted_share",
            "observed_tonnes", "predicted_tonnes", "share_error",
            "abs_share_error", "sq_share_error", "weight",
        ]
        long_cols = [c for c in long_cols if c in final_long.columns]
        final_long[long_cols].to_csv(long_out, index=False)
        log(f"Saved holdout long-form modal OD matrix cells to: {long_out}")

    del final_long, final_df
    gc.collect()
    return final_reports, final_matrix_reports, hit_summary


# =============================================================================
# Optional export of Nodus plugin parameters to MySQL
# =============================================================================


def _sql_identifier(name: str) -> str:
    """Validate a SQL identifier used as a generated parameter-table name."""
    import re

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Invalid SQL table name {name!r}. Use only letters, digits and underscores, "
            "and do not start with a digit."
        )
    return name


def parameter_table_name(
    explicit_table: Optional[str],
    table_prefix: str,
    dataset_id: int,
    number_of_models: int,
) -> str:
    """Return the MySQL table used to store plugin parameters for one dataset."""
    if explicit_table:
        base = _sql_identifier(explicit_table)
        if number_of_models > 1:
            return _sql_identifier(f"{base}_model{dataset_id}")
        return base
    return _sql_identifier(f"{_sql_identifier(table_prefix)}_model{dataset_id}")


def _format_param_value(value: object) -> str:
    """Format a scalar value for storage in the key-value parameter table."""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _add_nodus_param(
    rows: List[Dict[str, object]],
    key: str,
    value: object,
    param_type: str,
) -> None:
    """Append one key-value parameter row for the Nodus modal-choice plugin.

    The table consumed by the Java plugin is intentionally simple: every piece
    of information is stored as a key-value row. Dataset metadata such as the
    dataset id, dataset label, and model name are therefore stored as ordinary
    parameters, not as repeated columns.
    """
    if value is None:
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return
    rows.append({
        "param_key": str(key),
        "param_value": _format_param_value(value),
        "param_type": str(param_type),
    })


def _calibration_key_part(value: object) -> str:
    """Convert OD/group/category values to stable key tokens used by the Java plugin."""
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).strip().replace(".0", "").replace(" ", "_")


def _distance_band_key(value: object) -> str:
    """Convert Python distance-band labels to Java-plugin key strings.

    Distance-band labels are generated by make_distance_band(...). With
    thresholds 150,300 they are 0-150, 150-300 and 300+. The Java plugin
    requests the corresponding keys 0_150, 150_300 and 300plus.
    """
    s = str(value).strip()
    if s.endswith("+"):
        return s[:-1].replace("-", "_").replace(".", "p") + "plus"
    return s.replace("-", "_").replace(".", "p")


def _iww_availability_key(value: object) -> str:
    """Convert IWW availability labels to Java-plugin key strings."""
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "iww available", "available"}:
        return "1"
    return "0"


def _calibration_suffix(row: pd.Series, calibration_level: str) -> str:
    """Build the suffix after mc.cal.<mode> for the selected calibration level."""
    if calibration_level == "mode":
        return ""
    if calibration_level == "mode_group":
        return "." + _calibration_key_part(row["grp"])
    if calibration_level == "mode_distance":
        return "." + _distance_band_key(row["distance_band"])
    if calibration_level == "mode_group_distance":
        return "." + _calibration_key_part(row["grp"]) + "." + _distance_band_key(row["distance_band"])
    if calibration_level == "mode_iww_availability":
        return "." + _iww_availability_key(row["iww_availability"])
    if calibration_level == "mode_group_iww_availability":
        return "." + _calibration_key_part(row["grp"]) + "." + _iww_availability_key(row["iww_availability"])
    if calibration_level == "mode_od":
        return "." + _calibration_key_part(row["org"]) + "." + _calibration_key_part(row["dst"])
    if calibration_level == "mode_od_group":
        return (
            "." + _calibration_key_part(row["org"])
            + "." + _calibration_key_part(row["dst"])
            + "." + _calibration_key_part(row["grp"])
        )
    raise ValueError(f"Unsupported calibration level for export: {calibration_level}")


def build_nodus_parameter_table(
    dataset_id: int,
    dataset_label: str,
    model_name: str,
    params: pd.DataFrame,
    centering: Dict[str, float],
    groups: List[int],
    raw_segment_cols: List[str],
    dist_thresholds_km: Sequence[float],
    include_rail_cost: bool,
    calibration_level: str,
    calibration_constants: pd.DataFrame,
    calibration_max_abs_constant: float,
) -> pd.DataFrame:
    """Build the key-value table read by the Nodus Java modal-choice plugin.

    The table stores only param_key, param_value, and param_type. Dataset
    metadata are included as ordinary rows, not as separate columns. Keeping all
    plugin inputs in the same key-value structure makes the database table match
    the Java plugin's loading mechanism directly and avoids very large .costs
    files when calibration_level is mode_od or mode_od_group.
    """
    beta = extract_parameter_values(params)
    rows: List[Dict[str, object]] = []

    def add(key: str, value: object, param_type: str = "coefficient") -> None:
        _add_nodus_param(rows, key, value, param_type)

    # Plugin settings and run metadata.
    add("@MC.Plugin", "CalibratedODModalChoice", "setting")
    add("@MC.Model", dataset_id, "setting")
    add("@MC.DatasetLabel", dataset_label, "setting")
    add("@MC.ModelName", model_name, "setting")
    add("@MC.IncludeRailCost", str(bool(include_rail_cost)).lower(), "setting")
    add("@MC.UseCalibration", str(calibration_level != "none").lower(), "setting")
    add("@MC.CalibrationLevel", calibration_level, "setting")
    add("@MC.CalibrationMaxAbsConstant", calibration_max_abs_constant, "setting")
    add("@MC.DistanceScaleKm", DIST_SCALE, "setting")
    add("@MC.DistanceThresholdsKm", ",".join(_format_param_value(x) for x in dist_thresholds_km), "setting")
    add("@MC.SpreadPathsByInverseCost", "true", "setting")
    add("mc.distance.segment.count", len(raw_segment_cols), "setting")

    # Commodity-specific constants.
    for g in groups:
        add(f"mc.asc.iww.{int(g)}", beta.get(f"ASC_IWW_g{int(g)}", 0.0))
        add(f"mc.asc.rail.{int(g)}", beta.get(f"ASC_RAIL_g{int(g)}", 0.0))

    # Generic behavioral coefficients.
    for key, name in [
        ("mc.b_cost_iww", "B_COST_IWW"),
        ("mc.b_move_iww", "B_MOVE_IWW"),
        ("mc.b_cost_rail", "B_COST_RAIL"),
        ("mc.b_move_rail", "B_MOVE_RAIL"),
        ("mc.b_rail_iww_available", "B_RAIL_IWW_AVAILABLE"),
    ]:
        add(key, beta.get(name, 0.0))

    # Centering constants for relative variables.
    for key, variable in [
        ("mc.center.ln_cost_ratio_iww", "ln_cost_ratio_iww"),
        ("mc.center.ln_move_ratio_iww", "ln_move_ratio_iww"),
        ("mc.center.ln_cost_ratio_rail", "ln_cost_ratio_rail"),
        ("mc.center.ln_move_ratio_rail", "ln_move_ratio_rail"),
    ]:
        add(key, centering.get(variable, 0.0), "centering")

    # Distance-spline coefficients and centering constants.
    for i in range(1, len(raw_segment_cols) + 1):
        add(f"mc.b_dist_iww.{i}", beta.get(f"B_DIST_IWW_{i}", 0.0))
        add(f"mc.b_dist_rail.{i}", beta.get(f"B_DIST_RAIL_{i}", 0.0))
        add(f"mc.center.dseg_iww.{i}", centering.get(f"c_dseg_iww_{i}", 0.0), "centering")
        add(f"mc.center.dseg_rail.{i}", centering.get(f"c_dseg_rail_{i}", 0.0), "centering")

    # Calibration constants. Only non-zero constants are stored to reduce table
    # size. Missing keys are interpreted as zero by the Java plugin.
    if calibration_level != "none" and calibration_constants is not None and not calibration_constants.empty:
        for _, row in calibration_constants.iterrows():
            suffix = _calibration_suffix(row, calibration_level)
            ciww = float(row.get("calib_const_iww", 0.0))
            crail = float(row.get("calib_const_rail", 0.0))
            if np.isfinite(ciww) and abs(ciww) > 0.0:
                add(f"mc.cal.iww{suffix}", ciww, "calibration")
            if np.isfinite(crail) and abs(crail) > 0.0:
                add(f"mc.cal.rail{suffix}", crail, "calibration")

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["param_key"], keep="last")
        out = out.sort_values(["param_type", "param_key"]).reset_index(drop=True)
    return out


def store_nodus_parameters_to_mysql(
    parameter_rows: pd.DataFrame,
    table_name: str,
    user: str,
    password: str,
    host: str,
    database: str,
) -> None:
    """Create or replace the Nodus modal-choice key-value parameter table in MySQL."""
    table_name = _sql_identifier(table_name)
    if parameter_rows.empty:
        log(f"Parameter table {table_name} not created because there are no rows to store.")
        return

    url = URL.create(
        drivername="mysql+mysqlconnector",
        username=user,
        password=password,
        host=host,
        database=database,
    )
    engine = create_engine(url)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(
            f"""
            CREATE TABLE {table_name} (
                param_key VARCHAR(512) NOT NULL,
                param_value TEXT NOT NULL,
                param_type VARCHAR(32) NOT NULL,
                PRIMARY KEY (param_key)
            )
            """
        ))
        parameter_rows.to_sql(
            table_name,
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )

    log(f"Stored {len(parameter_rows):,} Nodus modal-choice parameters in MySQL table {table_name}.")
    log(f"Add this line to the Nodus .costs file for this dataset: @paramTable={table_name}")


# =============================================================================
# Dataset estimation and combined reports
# =============================================================================

def build_model_name(
    dataset_id: int,
    dataset_label: str,
    groups: List[int],
    dist_thresholds_km: Sequence[float],
    weight_type: str,
    weight_alpha: Optional[float],
    include_rail_cost: bool,
) -> str:
    """Build a compact base filename for all outputs of one dataset.

    The dataset label is intentionally not included in the filename. It remains
    present inside CSV contents and terminal summaries, but omitting it from the
    filename keeps paths shorter and less fragile.
    """
    groups_label = "_".join(str(g) for g in groups)
    knots_label = "_".join(
        str(int(t)) if float(t).is_integer() else str(t).replace(".", "p")
        for t in dist_thresholds_km
    )
    if weight_type == "power":
        weight_label = f"power{str(weight_alpha).replace('.', 'p')}w"
    else:
        weight_label = f"{weight_type}w"
    rail_cost_label = "withrailcost" if include_rail_cost else "norailcost"

    return build_output_name(
        MODEL_NAME_PREFIX,
        f"model{dataset_id}",
        f"grps_{groups_label}",
        f"knots_{knots_label}",
        weight_label,
        rail_cost_label,
    )



def estimate_one_dataset(
    dataset_id: int,
    dataset_label: str,
    user: str,
    password: str,
    host: str,
    database: str,
    groups: Optional[Iterable[int]],
    weight_type: str,
    weight_alpha: Optional[float],
    dist_thresholds_km: Sequence[float],
    include_rail_cost: bool,
    generate_reports: bool,
    max_iterations: int,
    save_row_predictions: bool,
    calibration_level: str,
    calibration_max_iterations: int,
    calibration_damping: float,
    calibration_tolerance: float,
    calibration_epsilon_tonnes: float,
    calibration_max_step: float,
    calibration_max_abs_constant: float,
    store_parameters_in_table: bool,
    parameter_table: Optional[str],
    parameter_table_prefix: str,
    number_of_models: int,
    holdout_fraction: float,
    holdout_by: str,
    holdout_seed: int,
) -> Dict[str, object]:
    log("\n" + "=" * 80)
    log(f"Estimating dataset {dataset_id} ({dataset_label})")
    log("=" * 80)

    raw = load_dataset_from_mysql(
        model=dataset_id,
        dataset_label=dataset_label,
        user=user,
        password=password,
        host=host,
        database=database,
        groups=groups,
    )

    holdout_enabled = holdout_fraction > 0
    if holdout_enabled:
        raw_train, raw_test, holdout_split_summary = split_train_test_raw(
            raw,
            holdout_fraction=holdout_fraction,
            holdout_by=holdout_by,
            holdout_seed=holdout_seed,
        )
    else:
        raw_train = raw
        raw_test = raw.iloc[0:0].copy()
        holdout_split_summary = pd.DataFrame()

    df_model, groups_present, centering, raw_segment_cols, dataset_summary_train, safe_defaults = preprocess_dataset(
        raw_train,
        dataset_id=dataset_id,
        dataset_label=dataset_label,
        weight_type=weight_type,
        weight_alpha=weight_alpha,
        dist_thresholds_km=dist_thresholds_km,
        sample_label="train" if holdout_enabled else "full",
    )

    df_test = pd.DataFrame()
    dataset_summary_test = pd.DataFrame()
    if holdout_enabled:
        df_test, test_groups, test_centering, test_raw_segment_cols, dataset_summary_test, _ = preprocess_dataset(
            raw_test,
            dataset_id=dataset_id,
            dataset_label=dataset_label,
            weight_type=weight_type,
            weight_alpha=weight_alpha,
            dist_thresholds_km=dist_thresholds_km,
            centering_input=centering,
            safe_defaults_input=safe_defaults,
            sample_label="test",
        )
        if test_raw_segment_cols != raw_segment_cols:
            raise RuntimeError("Train and test samples produced different distance segment columns.")
        if holdout_by == "group":
            missing_groups = sorted(set(test_groups) - set(groups_present))
            if missing_groups:
                log(
                    "Warning: group holdout contains groups not estimated in Biogeme: "
                    f"{missing_groups}. Their group-specific ASCs default to zero in test simulation."
                )

    model_name = build_model_name(
        dataset_id=dataset_id,
        dataset_label=dataset_label,
        groups=groups_present,
        dist_thresholds_km=dist_thresholds_km,
        weight_type=weight_type,
        weight_alpha=weight_alpha,
        include_rail_cost=include_rail_cost,
    )
    model_name = f"{model_name}_calib_{calibration_level}"
    if holdout_enabled:
        frac_label = str(holdout_fraction).replace(".", "p")
        model_name = f"{model_name}_holdout_{holdout_by}_{frac_label}_seed{holdout_seed}"

    biogeme_object = build_biogeme_model(
        df_model=df_model,
        groups=groups_present,
        model_name=model_name,
        weight_type=weight_type,
        weight_alpha=weight_alpha,
        raw_segment_cols=raw_segment_cols,
        include_rail_cost=include_rail_cost,
        generate_reports=generate_reports,
        max_iterations=max_iterations,
    )

    dropExistentOutput(model_name)

    log(f"\nStarting Biogeme estimation for dataset {dataset_id} ({dataset_label})...")
    try:
        results = biogeme_object.estimate()
    except BaseException as exc:
        log(f"\nERROR during Biogeme estimation for dataset {dataset_id}: {exc}")
        traceback.print_exc()
        raise
    log(f"Finished Biogeme estimation for dataset {dataset_id} ({dataset_label}).")

    log("\n==============================")
    log("Biogeme summary")
    log("==============================")
    log(results.short_summary())

    params = get_pandas_estimated_parameters(estimation_results=results)
    params.insert(0, "dataset_id", dataset_id)
    params.insert(1, "dataset_label", dataset_label)

    log("\n==============================")
    log("Estimated parameters")
    log("==============================")
    log(params.to_string())

    out_params = f"{model_name}_parameters.csv"
    params.to_csv(out_params, index=False)
    log(f"Saved parameter table to: {out_params}")

    out_centering = f"{model_name}_centering_constants.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "dataset_label": dataset_label,
                "sample": "train" if holdout_enabled else "full",
                "variable": key,
                "weighted_mean": value,
            }
            for key, value in centering.items()
        ]
    ).sort_values(["dataset_id", "variable"]).to_csv(out_centering, index=False)
    log(f"Saved centering constants to: {out_centering}")

    long_df, reports, matrix_reports, calibration_constants = write_validation_reports(
        df_model=df_model,
        params=params,
        raw_segment_cols=raw_segment_cols,
        model_name=model_name,
        include_rail_cost=include_rail_cost,
        save_row_predictions=save_row_predictions,
        calibration_level=calibration_level,
        calibration_max_iterations=calibration_max_iterations,
        calibration_damping=calibration_damping,
        calibration_tolerance=calibration_tolerance,
        calibration_epsilon_tonnes=calibration_epsilon_tonnes,
        calibration_max_step=calibration_max_step,
        calibration_max_abs_constant=calibration_max_abs_constant,
    )

    holdout_reports: Dict[str, pd.DataFrame] = {}
    holdout_matrix_reports: Dict[str, pd.DataFrame] = {}
    holdout_hit_summary = pd.DataFrame()
    if holdout_enabled:
        holdout_reports, holdout_matrix_reports, holdout_hit_summary = write_holdout_validation_reports(
            df_test=df_test,
            params=params,
            raw_segment_cols=raw_segment_cols,
            model_name=model_name,
            include_rail_cost=include_rail_cost,
            calibration_level=calibration_level,
            calibration_constants=calibration_constants,
            save_row_predictions=save_row_predictions,
        )

    if store_parameters_in_table:
        if holdout_enabled:
            log(
                "Warning: --store-parameters-in-table is active in a holdout run. "
                "The exported table contains coefficients and calibration constants estimated on the training sample only."
            )
        table_name = parameter_table_name(
            explicit_table=parameter_table,
            table_prefix=parameter_table_prefix,
            dataset_id=dataset_id,
            number_of_models=number_of_models,
        )
        parameter_rows = build_nodus_parameter_table(
            dataset_id=dataset_id,
            dataset_label=dataset_label,
            model_name=model_name,
            params=params,
            centering=centering,
            groups=groups_present,
            raw_segment_cols=raw_segment_cols,
            dist_thresholds_km=dist_thresholds_km,
            include_rail_cost=include_rail_cost,
            calibration_level=calibration_level,
            calibration_constants=calibration_constants,
            calibration_max_abs_constant=calibration_max_abs_constant,
        )
        audit_csv = f"{model_name}_nodus_parameter_table_{table_name}.csv"
        parameter_rows.to_csv(audit_csv, index=False)
        log(f"Saved Nodus parameter-table audit CSV to: {audit_csv}")
        store_nodus_parameters_to_mysql(
            parameter_rows=parameter_rows,
            table_name=table_name,
            user=user,
            password=password,
            host=host,
            database=database,
        )

    dataset_summary = pd.concat(
        [x for x in [dataset_summary_train, dataset_summary_test] if x is not None and not x.empty],
        ignore_index=True,
    )
    if holdout_enabled and not holdout_split_summary.empty:
        split_path = f"{model_name}_holdout_split_summary.csv"
        holdout_split_summary.to_csv(split_path, index=False)
        log(f"Saved holdout split summary to: {split_path}")

    # Keep only compact objects in memory. Full model dataframes and long-form
    # validation rows have already been written to disk if requested.
    del raw, raw_train, raw_test, df_model, df_test, biogeme_object, results
    gc.collect()

    return {
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "model_name": model_name,
        "params": params,
        "centering": centering,
        "dataset_summary": dataset_summary,
        "validation_reports": reports,
        "matrix_validation_reports": matrix_reports,
        "holdout_validation_reports": holdout_reports,
        "holdout_matrix_validation_reports": holdout_matrix_reports,
        "holdout_calibration_hit_summary": holdout_hit_summary,
        "long_validation": pd.DataFrame(),
        "groups": groups_present,
    }

def make_key_coefficient_table(results_by_dataset: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    key_params = [
        "B_COST_IWW",
        "B_MOVE_IWW",
        "B_COST_RAIL",
        "B_MOVE_RAIL",
        "B_RAIL_IWW_AVAILABLE",
        "B_DIST_IWW_1",
        "B_DIST_IWW_2",
        "B_DIST_IWW_3",
        "B_DIST_RAIL_1",
        "B_DIST_RAIL_2",
        "B_DIST_RAIL_3",
    ]

    for result in results_by_dataset:
        dataset_id = int(result["dataset_id"])
        dataset_label = str(result["dataset_label"])
        params = result["params"].copy()
        params_by_name = params.set_index("Name")

        for name in key_params:
            if name in params_by_name.index:
                p = params_by_name.loc[name]
                active = p.get("Active bound", np.nan)
                rows.append({
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_label,
                    "parameter": name,
                    "value": float(p["Value"]),
                    "robust_std_err": float(p["Robust std err."]) if "Robust std err." in p else np.nan,
                    "robust_t_stat": float(p["Robust t-stat."]) if "Robust t-stat." in p else np.nan,
                    "robust_p_value": float(p["Robust p-value"]) if "Robust p-value" in p else np.nan,
                    "active_bound": active,
                })
            else:
                rows.append({
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_label,
                    "parameter": name,
                    "value": np.nan,
                    "robust_std_err": np.nan,
                    "robust_t_stat": np.nan,
                    "robust_p_value": np.nan,
                    "active_bound": np.nan,
                })

    return pd.DataFrame(rows)


def write_combined_reports(results_by_dataset: List[Dict[str, object]], prefix: str) -> None:
    log("\n" + "=" * 80)
    log("Writing combined comparison reports")
    log("=" * 80)

    dataset_structure = pd.concat(
        [r["dataset_summary"] for r in results_by_dataset],
        ignore_index=True,
    )
    dataset_structure.to_csv(f"{prefix}_comparison_dataset_structure.csv", index=False)

    key_coeffs = make_key_coefficient_table(results_by_dataset)
    key_coeffs.to_csv(f"{prefix}_comparison_key_coefficients_long.csv", index=False)

    key_coeffs_wide = key_coeffs.pivot_table(
        index="parameter",
        columns=["dataset_id", "dataset_label"],
        values="value",
        aggfunc="first",
    )
    key_coeffs_wide.to_csv(f"{prefix}_comparison_key_coefficients_wide.csv")

    all_overall = []
    all_iww_avail = []
    all_group = []
    all_dist = []
    all_matrix_overall = []
    all_matrix_group = []
    all_matrix_dist = []
    all_matrix_iww_avail = []

    for r in results_by_dataset:
        did = int(r["dataset_id"])
        label = str(r["dataset_label"])
        reports = r["validation_reports"]
        matrix_reports = r["matrix_validation_reports"]

        for name, collector in [
            ("overall_by_mode", all_overall),
            ("by_iww_availability_mode", all_iww_avail),
            ("by_group_mode", all_group),
            ("by_distance_band_mode", all_dist),
        ]:
            table = reports[name].copy()
            table.insert(0, "dataset_label", label)
            table.insert(0, "dataset_id", did)
            collector.append(table)

        for name, collector in [
            ("matrix_cells_overall_by_mode", all_matrix_overall),
            ("matrix_cells_by_group_mode", all_matrix_group),
            ("matrix_cells_by_distance_band_mode", all_matrix_dist),
            ("matrix_cells_by_iww_availability_mode", all_matrix_iww_avail),
        ]:
            table = matrix_reports[name].copy()
            table.insert(0, "dataset_label", label)
            table.insert(0, "dataset_id", did)
            collector.append(table)

    pd.concat(all_overall, ignore_index=True).to_csv(
        f"{prefix}_comparison_validation_overall_by_dataset_mode.csv",
        index=False,
    )
    pd.concat(all_iww_avail, ignore_index=True).to_csv(
        f"{prefix}_comparison_validation_iww_availability_by_dataset_mode.csv",
        index=False,
    )
    pd.concat(all_group, ignore_index=True).to_csv(
        f"{prefix}_comparison_validation_group_by_dataset_mode.csv",
        index=False,
    )
    pd.concat(all_dist, ignore_index=True).to_csv(
        f"{prefix}_comparison_validation_distance_by_dataset_mode.csv",
        index=False,
    )

    pd.concat(all_matrix_overall, ignore_index=True).to_csv(
        f"{prefix}_comparison_matrix_cells_overall_by_dataset_mode.csv",
        index=False,
    )
    pd.concat(all_matrix_group, ignore_index=True).to_csv(
        f"{prefix}_comparison_matrix_cells_group_by_dataset_mode.csv",
        index=False,
    )
    pd.concat(all_matrix_dist, ignore_index=True).to_csv(
        f"{prefix}_comparison_matrix_cells_distance_by_dataset_mode.csv",
        index=False,
    )
    pd.concat(all_matrix_iww_avail, ignore_index=True).to_csv(
        f"{prefix}_comparison_matrix_cells_iww_availability_by_dataset_mode.csv",
        index=False,
    )

    log(f"Saved {prefix}_comparison_dataset_structure.csv")
    log(f"Saved {prefix}_comparison_key_coefficients_long.csv")
    log(f"Saved {prefix}_comparison_key_coefficients_wide.csv")
    log(f"Saved {prefix}_comparison_validation_overall_by_dataset_mode.csv")
    log(f"Saved {prefix}_comparison_validation_iww_availability_by_dataset_mode.csv")
    log(f"Saved {prefix}_comparison_validation_group_by_dataset_mode.csv")
    log(f"Saved {prefix}_comparison_validation_distance_by_dataset_mode.csv")
    log(f"Saved {prefix}_comparison_matrix_cells_overall_by_dataset_mode.csv")
    log(f"Saved {prefix}_comparison_matrix_cells_group_by_dataset_mode.csv")
    log(f"Saved {prefix}_comparison_matrix_cells_distance_by_dataset_mode.csv")
    log(f"Saved {prefix}_comparison_matrix_cells_iww_availability_by_dataset_mode.csv")

    log("\nCombined key coefficients:")
    log(key_coeffs_wide.to_string())

    overall = pd.concat(all_overall, ignore_index=True)
    display_cols = [
        "dataset_id", "dataset_label", "mode",
        "observed_share_of_total", "predicted_share_of_total",
        "share_point_error", "observed_tonnes", "predicted_tonnes", "tonnes_error",
    ]
    log("\nCombined validation: overall by dataset and mode")
    log(overall[display_cols].to_string(index=False))

    matrix_overall = pd.concat(all_matrix_overall, ignore_index=True)
    matrix_display_cols = [
        "dataset_id", "dataset_label", "mode", "n_cells", "n_positive_observed_cells",
        "observed_tonnes", "predicted_tonnes", "tonnes_error",
        "mae_tonnes", "rmse_tonnes", "weighted_mae_tonnes", "weighted_rmse_tonnes",
        "mape_nonzero_observed", "weighted_mape_nonzero_observed", "wape",
        "bias_pct_of_observed",
    ]
    log("\nCombined OD-matrix cell-level validation: overall by dataset and mode")
    log(matrix_overall[matrix_display_cols].to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate, calibrate, and validate a freight modal-choice model "
            "for one or several aggregate OD-matrix datasets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical examples
----------------

1) Behavioral estimation with moderate calibration:

   python CalibratedODModalChoice.py \
       --models 1,2,3 \
       --dataset-labels Europe_NUTS2,Benelux_NUTS3,Germany_NUTS3 \
       --groups 0,1,2,3,4,5,6,7,8,9 \
       --weight-type power \
       --weight-alpha 0.9 \
       --dist-thresholds-km 150,300 \
       --calibration-level mode_group_distance

2) Base-year OD-matrix reproduction:

   python CalibratedODModalChoice.py \
       --models 1,2,3 \
       --dataset-labels Europe_NUTS2,Benelux_NUTS3,Germany_NUTS3 \
       --groups 0,1,2,3,4,5,6,7,8,9 \
       --weight-type raw \
       --dist-thresholds-km 150,300 \
       --calibration-level mode_od_group \
       --calibration-max-abs-constant 8

4) Train/test validation by OD pair:

   python CalibratedODModalChoice.py \
       --models 1,2,3 \
       --dataset-labels Europe_NUTS2,Benelux_NUTS3,Germany_NUTS3 \
       --weight-type raw \
       --dist-thresholds-km 150,300 \
       --calibration-level mode_group_distance \
       --holdout-fraction 0.2 \
       --holdout-by od \
       --holdout-seed 12345

Use --include-rail-cost only as a sensitivity test unless rail cost is known to
be robustly identified in the selected datasets.

3) Store Nodus plugin parameters in database tables after estimation:

   python CalibratedODModalChoice.py \
       --models 1,2,3 \
       --dataset-labels Europe_NUTS2,Benelux_NUTS3,Germany_NUTS3 \
       --weight-type raw \
       --dist-thresholds-km 150,300 \
       --calibration-level mode_od_group \
       --calibration-max-abs-constant 8 \
       --store-parameters-in-table \
       --parameter-table-prefix modal_choice_params

   The Nodus .costs file can then contain a line such as:
       @paramTable=modal_choice_params_model1
""",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated table suffixes, e.g. '1,2,3'.",
    )
    parser.add_argument(
        "--dataset-labels",
        type=str,
        default=None,
        help=(
            "Optional comma-separated labels matching --models, for example "
            "'Europe_NUTS2,Benelux_NUTS3,Germany_NUTS3'."
        ),
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Optional comma-separated subset of grp values, e.g. '0,1,2'.",
    )
    parser.add_argument(
        "--weight-type",
        choices=["raw", "sqrt", "log1p", "none", "power"],
        default="power",
        help=(
            "How to weight each OD-commodity row in the likelihood. "
            "Use raw for base-year matrix reproduction; power is often better "
            "for less tonne-dominated behavioral estimation."
        ),
    )
    parser.add_argument(
        "--weight-alpha",
        type=float,
        default=0.9,
        help="Exponent alpha for --weight-type power: weight = total_qty ** alpha.",
    )
    parser.add_argument(
        "--dist-thresholds-km",
        type=str,
        default=",".join(str(int(t)) for t in DEFAULT_DIST_THRESHOLDS_KM),
        help=(
            "Comma-separated internal road-distance spline thresholds in km. "
            "For example, 150,300 creates segments 0-150, 150-300, and 300+ km."
        ),
    )
    parser.add_argument(
        "--include-rail-cost",
        action="store_true",
        help="Include B_COST_RAIL. Default excludes rail cost.",
    )
    parser.add_argument(
        "--calibration-level",
        choices=[
            "none",
            "mode",
            "mode_group",
            "mode_distance",
            "mode_group_distance",
            "mode_iww_availability",
            "mode_group_iww_availability",
            "mode_od",
            "mode_od_group",
        ],
        default="mode_group_distance",
        help=(
            "Post-estimation base-year calibration constants. "
            "Default mode_group_distance calibrates IWW and rail constants by "
            "commodity group and distance band. Use none to disable. "
            "mode_od_group is the most granular and may overfit."
        ),
    )
    parser.add_argument(
        "--calibration-max-iterations",
        type=int,
        default=80,
        help="Maximum iterations for base-year calibration constants.",
    )
    parser.add_argument(
        "--calibration-damping",
        type=float,
        default=0.7,
        help="Damping factor for calibration updates in (0,1].",
    )
    parser.add_argument(
        "--calibration-tolerance",
        type=float,
        default=1.0e-5,
        help="Stop calibration when the maximum absolute damped update is below this value.",
    )
    parser.add_argument(
        "--calibration-epsilon-tonnes",
        type=float,
        default=1.0e-6,
        help=(
            "Small pseudo-tonnage used in log-ratio calibration updates. "
            "Needed for zero observed/predicted cells."
        ),
    )
    parser.add_argument(
        "--calibration-max-step",
        type=float,
        default=2.0,
        help="Maximum absolute log-constant update per calibration iteration.",
    )
    parser.add_argument(
        "--calibration-max-abs-constant",
        type=float,
        default=12.0,
        help="Maximum absolute value allowed for any calibration constant.",
    )
    parser.add_argument(
        "--generate-reports",
        action="store_true",
        help="Ask Biogeme to generate HTML/YAML reports. Off by default for speed and robustness.",
    )
    parser.add_argument(
        "--save-row-predictions",
        action="store_true",
        help="Save row-level prediction files. These can be large.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of raw rows/keys to reserve as a test sample. "
            "Use 0 to disable train/test validation."
        ),
    )
    parser.add_argument(
        "--holdout-by",
        choices=["row", "od", "od_group", "group"],
        default="od",
        help=(
            "Unit used for the holdout split. 'od' holds out complete OD pairs; "
            "'od_group' holds out OD x group cells; 'row' holds out individual rows; "
            "'group' holds out whole commodity groups."
        ),
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=12345,
        help="Random seed used for the train/test split.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5000,
        help="Maximum number of Biogeme optimization iterations.",
    )
    parser.add_argument("--user", type=str, required=True, default="nodus")
    parser.add_argument("--password", type=str, required=True, default="nodus")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--model-name-prefix",
        type=str,
        default=MODEL_NAME_PREFIX,
        help=(
            "Optional prefix prepended to generated output filenames. "
            "The default is blank, which keeps filenames shorter. Dataset labels "
            "are not included in filenames."
        ),
    )
    parser.add_argument(
        "--store-parameters-in-table",
        action="store_true",
        help=(
            "Store the coefficients, centering constants, settings, and calibration "
            "constants needed by the Nodus modal-choice plugin in a MySQL key-value table."
        ),
    )
    parser.add_argument(
        "--parameter-table",
        type=str,
        default=None,
        help=(
            "Explicit base name for the Nodus plugin parameter table. If several datasets "
            "are estimated, _modelX is appended automatically. If omitted, "
            "--parameter-table-prefix is used."
        ),
    )
    parser.add_argument(
        "--parameter-table-prefix",
        type=str,
        default="modal_choice_params",
        help=(
            "Prefix used to generate Nodus plugin parameter tables when --parameter-table "
            "is not supplied. Default creates tables such as modal_choice_params_model1."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue estimating remaining datasets if one dataset fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global MODEL_NAME_PREFIX
    MODEL_NAME_PREFIX = sanitize_output_token(args.model_name_prefix)

    models = parse_int_list(args.models)
    dataset_labels = parse_labels(args.dataset_labels, models)
    groups = parse_groups(args.groups)
    dist_thresholds_km = parse_thresholds(args.dist_thresholds_km)
    weight_alpha = args.weight_alpha if args.weight_type == "power" else None

    if args.weight_type == "power":
        comparison_weight_label = f"power{str(weight_alpha).replace('.', 'p')}w"
    else:
        comparison_weight_label = f"{args.weight_type}w"
    knots_label = "_".join(
        str(int(t)) if float(t).is_integer() else str(t).replace(".", "p")
        for t in dist_thresholds_km
    )
    rail_cost_label = "withrailcost" if args.include_rail_cost else "norailcost"
    calibration_label = f"calib_{args.calibration_level}"
    comparison_prefix = build_output_name(
        MODEL_NAME_PREFIX,
        f"models_{'_'.join(map(str, models))}",
        f"knots_{knots_label}",
        comparison_weight_label,
        rail_cost_label,
        calibration_label,
    )
    if args.holdout_fraction > 0:
        frac_label = str(args.holdout_fraction).replace(".", "p")
        comparison_prefix = build_output_name(
            comparison_prefix,
            "holdout",
            args.holdout_by,
            frac_label,
            f"seed{args.holdout_seed}",
        )

    wd = (Path.cwd() / args.output_dir).resolve()
    terminal_log_path, terminal_log_file, old_stdout, old_stderr = start_terminal_log(
        wd, comparison_prefix
    )

    try:
        os.chdir(wd)
        log(f"Terminal output is being written to: {terminal_log_path}")

        print_run_parameters(
            args=args,
            models=models,
            dataset_labels=dataset_labels,
            groups=groups,
            dist_thresholds_km=dist_thresholds_km,
            weight_alpha=weight_alpha,
        )

        all_results: List[Dict[str, object]] = []

        for dataset_id in models:
            dataset_label = dataset_labels[dataset_id]
            log(f"\nPreparing dataset {dataset_id}; compact results currently retained: {len(all_results)}")
            gc.collect()
            try:
                result = estimate_one_dataset(
                    dataset_id=dataset_id,
                    dataset_label=dataset_label,
                    user=args.user,
                    password=args.password,
                    host=args.host,
                    database=args.database,
                    groups=groups,
                    weight_type=args.weight_type,
                    weight_alpha=weight_alpha,
                    dist_thresholds_km=dist_thresholds_km,
                    include_rail_cost=args.include_rail_cost,
                    generate_reports=args.generate_reports,
                    max_iterations=args.max_iterations,
                    save_row_predictions=args.save_row_predictions,
                    calibration_level=args.calibration_level,
                    calibration_max_iterations=args.calibration_max_iterations,
                    calibration_damping=args.calibration_damping,
                    calibration_tolerance=args.calibration_tolerance,
                    calibration_epsilon_tonnes=args.calibration_epsilon_tonnes,
                    calibration_max_step=args.calibration_max_step,
                    calibration_max_abs_constant=args.calibration_max_abs_constant,
                    store_parameters_in_table=args.store_parameters_in_table,
                    parameter_table=args.parameter_table,
                    parameter_table_prefix=args.parameter_table_prefix,
                    number_of_models=len(models),
                    holdout_fraction=args.holdout_fraction,
                    holdout_by=args.holdout_by,
                    holdout_seed=args.holdout_seed,
                )
                all_results.append(result)
                gc.collect()
            except BaseException as exc:
                log(f"\nFAILED dataset {dataset_id} ({dataset_label}): {exc}")
                traceback.print_exc()
                if not args.continue_on_error:
                    raise

        if all_results:
            write_combined_reports(all_results, comparison_prefix)
            print("Done.")
        else:
            raise RuntimeError("No datasets were estimated successfully.")
    except BaseException:
        log("\nUnhandled error in main run.")
        traceback.print_exc()
        raise
    finally:
        stop_terminal_log(terminal_log_path, terminal_log_file, old_stdout, old_stderr)


if __name__ == "__main__":
    main()
