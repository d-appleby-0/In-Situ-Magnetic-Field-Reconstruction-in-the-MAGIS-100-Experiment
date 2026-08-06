"""
parse_table_files.py
====================
Parses Opera Simulia .table files from the MAGIS-100 magnetic shielding simulation.

Supports two export modes from Opera (auto-detected from the file header):

  SCALAR mode  — one FIELU column = |H| magnitude (original export)
    Header example:  '1 15 15 2'
    Data columns:    x  y  z  |H|
    Output columns:  field_value (= |H|)

  VECTOR mode  — three FIELU columns = Hx, Hy, Hz components (Option A export)
    Header example:  '1 15 15 4'
    Data columns:    x  y  z  Hx  Hy  Hz
    Output columns:  field_Hx  field_Hy  field_Hz  field_value (= sqrt(Hx²+Hy²+Hz²))

Output CSV columns (both modes):
    scenario_id | coil_density | slice_id | x | y | z |
    field_value | field_Hx | field_Hy | field_Hz |
    r | region | file | export_mode

    In scalar mode, field_Hx / field_Hy / field_Hz are NaN.
    In vector mode, field_value = sqrt(Hx²+Hy²+Hz²) for backward compatibility
    with all downstream code that reads field_value.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# Geometry constants (inches)
R_TUBE_INNER  = 2.75   # inner radius: free-space vacuum tube interior
R_TUBE_OUTER  = 3.0    # outer radius: tube wall boundary
R_SHIELD      = 7.0    # inner face of shield material


def classify_point(x: float, y: float) -> str:
    """Return the geometric region label for a cross-sectional point (x, y)."""
    r2 = x * x + y * y
    if r2 <= R_TUBE_INNER ** 2:
        return "tube_interior"
    elif r2 <= R_TUBE_OUTER ** 2:
        return "tube_wall"
    elif r2 < R_SHIELD ** 2:
        return "sensor_zone"
    else:
        return "shield_material"


def parse_filename(filename: str) -> tuple:
    """
    Extract (scenario_id, coil_density, slice_index) from a filename.

    Pattern: Results_DefField_{J}Ain-2.{slice}
    Returns (scenario_id, coil_density_float, slice_int)
    or (None, None, None) if the filename does not match.
    """
    m = re.search(r'Results_DefField_((\d*)(?:\.(\d+))?)Ain-2\.(\d+)', filename)
    if not m:
        return None, None, None
    J         = float(m.group(1))
    slice_idx = int(m.group(4))
    return f"J{J:05.3f}_s{slice_idx}", float(J), slice_idx


def _parse_header(lines: list) -> dict:
    """
    Parse the Opera .table header block and detect export mode.

    Returns dict with keys:
      nx, ny           : grid dimensions
      n_data_cols      : raw value from line 0
      export_mode      : 'scalar' or 'vector'
      fielu_count      : number of FIELU columns found (1 or 3)
      n_header_lines   : total header lines to skip before data begins
    """
    parts = lines[0].split()
    nx          = int(parts[1])
    ny          = int(parts[2])
    n_data_cols = int(parts[3])

    # Count FIELU occurrences across all non-terminator descriptor lines
    fielu_count = sum(
        1 for l in lines[1:]
        if l.strip() != '0' and 'FIELU' in l
    )

    if fielu_count == 1:
        export_mode = "scalar"
    elif fielu_count == 3:
        export_mode = "vector"
    else:
        raise ValueError(
            f"Unexpected number of FIELU columns in header: {fielu_count}. "
            f"Expected 1 (scalar |H|) or 3 (vector Hx,Hy,Hz). "
            f"Check your Opera table export settings."
        )

    # Header ends at the '0' terminator line
    n_header_lines = next(
        i for i, l in enumerate(lines) if l.strip() == '0'
    ) + 1

    return {
        "nx"            : nx,
        "ny"            : ny,
        "n_data_cols"   : n_data_cols,
        "export_mode"   : export_mode,
        "fielu_count"   : fielu_count,
        "n_header_lines": n_header_lines,
    }


def parse_table_file(filepath) -> pd.DataFrame | None:
    """
    Parse a single Opera .table file into a tidy DataFrame.

    Auto-detects scalar vs vector export from the header.
    Returns None if the filename does not match the expected convention.
    """
    filepath = Path(filepath)
    scenario_id, coil_density, slice_idx = parse_filename(filepath.name)
    if scenario_id is None:
        print(f"  [SKIP] Unrecognised filename: {filepath.name}")
        return None

    with open(filepath, "r") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]

    if len(lines) < 7:
        print(f"  [SKIP] Too few lines: {filepath.name}")
        return None

    try:
        header = _parse_header(lines)
    except ValueError as e:
        print(f"  [SKIP] Header error in {filepath.name}: {e}")
        return None

    export_mode   = header["export_mode"]
    n_skip        = header["n_header_lines"]
    expected_cols = 4 if export_mode == "scalar" else 6

    records = []
    for line in lines[n_skip:]:
        cols = line.split()
        if len(cols) < expected_cols:
            continue
        try:
            x = float(cols[0])
            y = float(cols[1])
            z = float(cols[2])
        except ValueError:
            continue

        if export_mode == "scalar":
            try:
                val = float(cols[3])
            except ValueError:
                continue
            Hx = np.nan; Hy = np.nan; Hz = np.nan

        else:  # vector
            try:
                Hx = float(cols[3])
                Hy = float(cols[4])
                Hz = float(cols[5])
            except ValueError:
                continue
            val = float(np.sqrt(Hx**2 + Hy**2 + Hz**2))

        records.append({
            "scenario_id"  : scenario_id,
            "coil_density" : coil_density,
            "slice_id"     : slice_idx,
            "x"            : x,
            "y"            : y,
            "z"            : z,
            "field_value"  : val,    # |H| in A/m — always populated
            "field_Hx"     : Hx,     # NaN in scalar mode
            "field_Hy"     : Hy,     # NaN in scalar mode
            "field_Hz"     : Hz,     # NaN in scalar mode
            "r"            : round(np.sqrt(x*x + y*y), 6),
            "region"       : classify_point(x, y),
            "export_mode"  : export_mode,
        })

    if not records:
        print(f"  [SKIP] No valid data rows in {filepath.name}")
        return None

    df = pd.DataFrame(records)
    df["file"] = filepath.name
    return df


def parse_all_table_files(directory, pattern="Results_DefField_*Ain-2.*",
                          verbose=True) -> pd.DataFrame:
    """
    Parse every matching .table file in a directory into one combined DataFrame.

    Mixed scalar/vector files in the same directory are supported but will
    trigger a warning — downstream code expects a consistent export_mode
    across all files. Use the scalar mode for legacy CSVs and vector mode
    for new exports; do not mix them in the same run.

    Parameters
    ----------
    directory : str or Path
    pattern   : glob to match filenames
    verbose   : print progress and summary

    Returns
    -------
    pd.DataFrame sorted by coil_density, slice_id, x, y.
    Columns always include field_value; field_Hx/Hy/Hz are NaN
    in scalar-mode files and populated in vector-mode files.
    """
    directory = Path(directory)
    files     = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {directory}"
        )

    frames = []
    for fp in files:
        if verbose:
            print(f"  Parsing: {fp.name}")
        df = parse_table_file(fp)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No valid data parsed.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["coil_density", "slice_id", "x", "y"])
        .reset_index(drop=True)
    )
    combined["region"] = combined["region"].astype("category")

    # Check for mixed export modes
    modes = combined["export_mode"].unique()
    if len(modes) > 1:
        print(f"\n  X WARNING: Mixed export modes in dataset: {list(modes)}.")
        print(f"    All files should be either scalar or vector.")
        print(f"    Vector-mode files have Hx/Hy/Hz; scalar-mode files have NaN.")
        print(f"    Downstream multipole fitting (Mode B) requires all files")
        print(f"    to be vector-mode. Re-export all files from Opera in vector mode.")

    if verbose:
        mode_str = modes[0] if len(modes) == 1 else "MIXED"
        print(f"\n   {len(files)} file(s) → {len(combined):,} rows  [export_mode={mode_str}]")
        print("\nRegion counts:")
        print(combined["region"].value_counts().to_string())
        print("\nScenario index (coil_density × slice):")
        summary = (
            combined[["scenario_id", "coil_density", "slice_id", "z"]]
            .drop_duplicates()
            .sort_values(["coil_density", "slice_id"])
        )
        print(summary.to_string(index=False))

    return combined


def has_vector_data(df: pd.DataFrame) -> bool:
    """
    Return True if the DataFrame contains valid (non-NaN) vector H components.
    Use this to decide whether to pass mode='vector' to fit_all_scenarios.
    """
    return (
        "field_Hx" in df.columns
        and df["field_Hx"].notna().any()
    )


# Region accessors

def get_tube_interior(df: pd.DataFrame) -> pd.DataFrame:
    """Points inside the vacuum tube — regression targets."""
    return df[df["region"] == "tube_interior"].copy()

def get_sensor_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Points in the annular sensor gap — regression inputs."""
    return df[df["region"] == "sensor_zone"].copy()

def get_tube_wall(df: pd.DataFrame) -> pd.DataFrame:
    """Points inside the tube wall (usually discarded)."""
    return df[df["region"] == "tube_wall"].copy()

def get_shield_material(df: pd.DataFrame) -> pd.DataFrame:
    """Points inside the mu-metal shield body (always discarded)."""
    return df[df["region"] == "shield_material"].copy()


# CLI

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Parsing Opera .table files in: {data_dir}\n")

    df = parse_all_table_files(data_dir)

    # Report per-mode field stats
    for mode, grp in df.groupby("export_mode", observed=True):
        print(f"\n── Field stats [{mode} mode] by region ──")
        print(
            grp.groupby("region", observed=True)["field_value"]
            .agg(count="count", min="min", mean="mean", max="max")
            .round(5).to_string()
        )
        if mode == "vector":
            for comp in ["field_Hx", "field_Hy", "field_Hz"]:
                print(f"\n  {comp} range: "
                      f"{grp[comp].min():.4f} to {grp[comp].max():.4f} A/m")

    out = Path(data_dir) / "parsed_fields.csv"
    try:
        df.to_csv(out, index=False)
    except OSError:
        out = Path("parsed_fields.csv")
        df.to_csv(out, index=False)
    print(f"\n✓  Saved → {out}")
    if has_vector_data(df):
        print("  Vector components (field_Hx, field_Hy, field_Hz) are populated.")
        print("  Use mode='vector' in fit_all_scenarios() for Mode B fitting.")
    else:
        print("  Scalar export detected — field_Hx/Hy/Hz are NaN.")
        print("  Re-export from Opera with Hx,Hy,Hz components for Mode B fitting.")
