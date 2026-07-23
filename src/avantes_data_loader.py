from pathlib import Path
import numpy as np


def load_folder(folder_path, max_rows=None):
    """
    Load all Avantes Rwd8 .txt files in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing the .txt files.
    max_rows : int or None
        Maximum number of rows to read from each file.

    Returns
    -------
    dict
        {filename: {"x": np.ndarray, "y": np.ndarray}}
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    txt_files = sorted(folder.glob("*.txt"))

    all_data = {}

    for file in txt_files:
        all_data[file.name] = load_single_file(file, max_rows)

    return all_data


def load_single_file(txt_file, max_rows=None):
    """
    Extract wavelength (column 1) and signal (column 5)
    from an Avantes Rwd8 text export.
    """

    txt_file = Path(txt_file)
    lines = txt_file.read_text(errors="ignore").splitlines()

    header_index = None

    # Find table header line containing "Wave"
    for i, line in enumerate(lines):
        if "Wave" in line and ";" in line:
            header_index = i
            break

    if header_index is None:
        raise ValueError(f"Could not find data table in {txt_file.name}")

    data_start = header_index + 1

    # Skip units line if present
    if data_start < len(lines) and "[" in lines[data_start]:
        data_start += 1

    x_vals = []
    y_vals = []

    for line in lines[data_start:]:

        s = line.strip()
        if not s:
            continue

        parts = [p.strip() for p in s.split(";")]

        if len(parts) < 5:
            continue

        try:
            x = float(parts[0])
            y = float(parts[4])
        except ValueError:
            continue

        x_vals.append(x)
        y_vals.append(y)

        if max_rows is not None and len(x_vals) >= max_rows:
            break

    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    return {"x": x, "y": y}