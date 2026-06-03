"""CuTS engine — ported from the host's wrappers/cuts_wrapper.py.

Standalone (no host `wrappers/` imports) so the plugin is self-contained inside
the submodule. CuTS trains a base denoiser unconditionally and enforces
constraints through its DSL program at synthesis time, so train/generate
decouple: `train` fits + caches the base model under model_dir (symlinked into
CuTS's expected layout); `generate` loads that cache, finetunes/synthesises under
the DSL constraints, and never retrains.

CuTS uses relative paths and requires its repo root as the working directory, so
both entry points chdir into the CuTS repo and restore cwd afterwards. The host
runs this plugin in the `cuts` conda env (declared in meta.json).
"""
from __future__ import annotations

import datetime
import importlib
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

CUTS_DIR = str(Path(__file__).resolve().parents[1])

logger = logging.getLogger("cuts.sdcontract")

META_NAME = "meta.json"
MARKER_NAME = "cuts_denoiser.pt"
DATA_STASH = "cuts_data.csv"


# --- inlined link helper (was wrappers.utils.common) ------------------------

def _ensure_dir_link(src: str, dst: str) -> None:
    """Symlink dst -> src (a directory), replacing whatever is at dst."""
    if os.path.islink(dst) or os.path.isfile(dst):
        os.unlink(dst)
    elif os.path.isdir(dst):
        import shutil
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.abspath(src), dst)


# --- CuTS env robustness (ported verbatim) ----------------------------------

def _validate_cuts_python_dependencies() -> None:
    required = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "pyparsing", "torch"]
    missing = []
    for name in required:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "CuTS environment is missing required Python modules: " + ", ".join(missing))


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy-2 pickles to load in NumPy-1 CuTS environments."""
    try:
        import numpy.core as numpy_core
    except Exception:
        return
    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("multiarray", "_multiarray_umath", "numeric", "fromnumeric",
                 "umath", "numerictypes", "overrides"):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


def _install_torch_pickle_compat(device: str) -> None:
    """Map cached CUDA tensors to CPU when CuTS is forced onto CPU."""
    if device != "cpu":
        return
    try:
        import io
        import torch
        import torch.storage
    except Exception:
        return
    if getattr(torch.storage._load_from_bytes, "_cuts_cpu_patch", False):
        return

    def _load_from_bytes_cpu(buffer):
        return torch.load(io.BytesIO(buffer), map_location=torch.device("cpu"))

    _load_from_bytes_cpu._cuts_cpu_patch = True
    torch.storage._load_from_bytes = _load_from_bytes_cpu


def _resolve_cuts_device(requested: str) -> str:
    """CuTS-like CUDA probe; fall back to CPU on runtime failure."""
    if requested != "cuda":
        return requested
    try:
        import torch
    except Exception:
        logger.warning("PyTorch import failed during CuTS device probe; using CPU.")
        return "cpu"
    if not torch.cuda.is_available():
        logger.info("CuTS: CUDA unavailable; using CPU.")
        return "cpu"
    try:
        module = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.BatchNorm1d(16), torch.nn.ReLU()).cuda()
        module.train()
        out = module(torch.randn(4, 8, device="cuda"))
        out.sum().backward()
        torch.cuda.synchronize()
        return "cuda"
    except Exception as exc:
        logger.warning("CuTS: CUDA probe failed (%s); falling back to CPU.", exc)
        return "cpu"


# --- DSL helpers ------------------------------------------------------------

def _dsl_dataset_name(dataset: str) -> str:
    if dataset.lower() == "default":
        return "DefaultAnonymized"
    if "_" in dataset:
        return "".join(part.capitalize() for part in dataset.split("_"))
    return dataset.capitalize()


def _build_program(dsl_dataset: str, constraints: list[tuple[str, str]]) -> str:
    lines = [f"SYNTHESIZE: {dsl_dataset};"]
    for col, val in constraints:
        val = val.strip()
        if any(ch in val for ch in (" ", "\t", ";", "=", "\n")):
            logger.warning(
                "CuTS DSL: value %r for column %r contains special characters that "
                "the DSL parser splits into tokens (BUG-007: no quoted literals).", val, col)
        lines.append("    ENFORCE: ROW CONSTRAINT: PARAM 1:")
        lines.append(f"        {col} == {val};")
    lines.append("END;")
    return "\n".join(lines)


def _link_trained_models(abs_model_dir: str, dataset: str) -> None:
    """Symlink model_dir into CuTS's trained_models slot for this dataset."""
    dataset_dir_name = dataset.lower().replace("_", "")
    target = os.path.join(CUTS_DIR, "experiment_data",
                          "customizable_synthesizer_experiments", dataset_dir_name,
                          "trained_models")
    _ensure_dir_link(abs_model_dir, target)


def _prepare_env(train_csv: str) -> None:
    abs_csv = os.path.abspath(train_csv)
    os.environ["CUTS_TRAIN_CSV"] = abs_csv
    os.environ["CUTS_TEST_CSV"] = abs_csv


# --- contract entry points --------------------------------------------------

def train(req: dict) -> Path:
    """Fit + cache the CuTS base denoiser; always leaves a contract meta.json."""
    abs_model_dir = os.path.abspath(req["output_model_dir"])
    os.makedirs(abs_model_dir, exist_ok=True)
    dataset = req["dataset"]
    train_csv = os.path.abspath(req["train_csv"])
    label_col = req.get("label_column")
    columns = list(pd.read_csv(train_csv, nrows=0).columns)

    # Stash the training CSV so generate can rebuild the dataset from model_dir.
    import shutil
    shutil.copy(train_csv, os.path.join(abs_model_dir, DATA_STASH))

    _link_trained_models(abs_model_dir, dataset)
    dsl_dataset = _dsl_dataset_name(dataset)

    original_cwd = os.getcwd()
    os.chdir(CUTS_DIR)
    if CUTS_DIR not in sys.path:
        sys.path.insert(0, CUTS_DIR)
    try:
        _prepare_env(train_csv)
        _validate_cuts_python_dependencies()
        _install_numpy_pickle_compat()
        from customizable_synthesizer import CuTS

        device = _resolve_cuts_device(req.get("device") or "cuda")
        _install_torch_pickle_compat(device)

        # Train the base denoiser with an unconstrained program; constraints are
        # applied later at generate time (the cached base model is constraint-free).
        program = _build_program(dsl_dataset, [])
        cuts = CuTS(program, device=device)
        logger.info("Training CuTS denoiser for %s (device=%s)...", dataset, device)
        cuts.fit(verbose=True)
        Path(abs_model_dir, MARKER_NAME).touch()
    finally:
        os.chdir(original_cwd)

    meta = {
        "schema_version": "1.0",
        "method": req["method"],
        "dataset": dataset,
        "label_column": label_col,
        "columns": columns,
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "sdcontract_core_version": "1.0",
        # CuTS-specific (consumed by generate):
        "dsl_dataset": dsl_dataset,
        "device": device,
    }
    Path(abs_model_dir, META_NAME).write_text(json.dumps(meta, indent=2))
    return Path(abs_model_dir)


def generate(req: dict, native_constraints: list[str]) -> Path:
    """Synthesise to the host-chosen output_csv_path under DSL constraints; loads
    the cached base denoiser and never retrains."""
    abs_model_dir = os.path.abspath(req["model_dir"])
    meta = json.loads(Path(abs_model_dir, META_NAME).read_text())
    dataset = meta["dataset"]
    dsl_dataset = meta["dsl_dataset"]
    n_samples = int(req["n_samples"])
    abs_output_path = os.path.abspath(req["output_csv_path"])
    stash_csv = os.path.join(abs_model_dir, DATA_STASH)
    if not Path(abs_model_dir, MARKER_NAME).exists():
        raise RuntimeError(f"No CuTS base denoiser under {abs_model_dir}; train before generate.")

    constraints = []
    for raw in native_constraints:
        if "=" in raw:
            k, v = raw.split("=", 1)
            constraints.append((k.strip(), v.strip()))

    _link_trained_models(abs_model_dir, dataset)

    original_cwd = os.getcwd()
    os.chdir(CUTS_DIR)
    if CUTS_DIR not in sys.path:
        sys.path.insert(0, CUTS_DIR)
    try:
        _prepare_env(stash_csv)
        _validate_cuts_python_dependencies()
        _install_numpy_pickle_compat()
        from customizable_synthesizer import CuTS

        device = _resolve_cuts_device(req.get("device") or meta.get("device") or "cuda")
        _install_torch_pickle_compat(device)

        program = _build_program(dsl_dataset, constraints)
        cuts = CuTS(program, device=device)
        # fit(verbose=False) restores the cached base model/workloads and
        # initialises finetuned_model for this program (no base retraining).
        cuts.fit(verbose=False)
        syndata = cuts.generate_data(n_samples)

        full_oh_width = sum(len(v) for v in cuts.dataset.full_one_hot_index_map.values())
        train_oh_width = sum(len(v) for v in cuts.dataset.train_full_one_hot_index_map.values())
        if syndata.shape[1] == full_oh_width:
            with_label = True
        elif syndata.shape[1] == train_oh_width:
            with_label = False
        else:
            logger.warning("CuTS width %d matches neither one-hot width; assuming with_label.",
                           syndata.shape[1])
            with_label = True

        decoded = cuts.dataset.decode_full_one_hot_batch(
            syndata, buckets=32, with_label=with_label, input_torch=True)
        columns = (cuts.dataset.features.keys() if with_label
                   else cuts.dataset.train_features.keys())
        df = pd.DataFrame(decoded, columns=columns)

        # CuTS decodes numericals as bin-midpoint floats; snap back using the stash.
        if os.path.exists(stash_csv):
            real = pd.read_csv(stash_csv)
            for col in df.columns:
                if col not in real.columns:
                    continue
                if pd.api.types.is_bool_dtype(real[col]):
                    df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
                elif pd.api.types.is_numeric_dtype(real[col]):
                    col_min, col_max = float(real[col].min()), float(real[col].max())
                    df[col] = df[col].astype(float).clip(col_min, col_max)
                    if pd.api.types.is_integer_dtype(real[col]):
                        df[col] = df[col].round().astype(real[col].dtype)
                    else:
                        unique_vals = real[col].dropna().unique()
                        if len(unique_vals) <= 20:
                            import numpy as np
                            arr = np.sort(unique_vals)
                            df[col] = arr[np.argmin(np.abs(arr[:, None] - df[col].values[None, :]), axis=0)]
    finally:
        os.chdir(original_cwd)

    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    df.to_csv(abs_output_path, index=False)
    logger.info("Saved %d rows -> %s", len(df), abs_output_path)
    return Path(abs_output_path)
