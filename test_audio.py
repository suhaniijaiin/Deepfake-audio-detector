"""
test_audio.py — CLI script to test audio samples against the trained ensemble model

Usage:
    python test_audio.py --file path/to/audio.wav
    python test_audio.py --file path/to/audio.wav --threshold 0.4
    python test_audio.py --file path/to/audio.wav --json
    python test_audio.py --folder path/to/audio_folder/
"""

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TF logs

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_DIR     = os.getenv("MODEL_DIR", "./outputs")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble_softvote.keras")
META_PATH     = os.path.join(MODEL_DIR, "ensemble_meta.json")
SUPPORTED_EXT = {".wav", ".flac", ".mp3"}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _register_custom_layers():
    import keras as ks
    @ks.saving.register_keras_serializable()
    class SumPool1D(ks.layers.Layer):
        def call(self, x):    return ks.ops.sum(x, axis=1)
        def get_config(self): return super().get_config()
    return {"SumPool1D": SumPool1D}


def load_model_and_meta():
    """Load the ensemble model and its metadata / config."""
    if not os.path.isfile(ENSEMBLE_PATH):
        sys.exit(
            f"\n[ERROR] Model not found: {ENSEMBLE_PATH}\n"
            "  Train the model in the notebook first, then make sure the\n"
            "  outputs/ folder is in the same directory as test_audio.py.\n"
        )
    if not os.path.isfile(META_PATH):
        sys.exit(f"\n[ERROR] Metadata not found: {META_PATH}\n")

    print("Loading model ...", end=" ", flush=True)
    from tensorflow.keras.models import load_model
    model = load_model(ENSEMBLE_PATH, custom_objects=_register_custom_layers())
    with open(META_PATH) as f:
        meta = json.load(f)
    print("done.")
    return model, meta


def load_audio(path: str, sr: int = 16_000, n_samples: int = 48_000) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        start = (len(y) - n_samples) // 2
        y = y[start: start + n_samples]
    return y.astype(np.float32)


def extract_mfcc_sequence(
    y: np.ndarray,
    sr: int = 16_000,
    n_mfcc: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    T: int = 300,
) -> np.ndarray:
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
    if mfcc.shape[1] < T:
        mfcc = np.pad(mfcc, ((0, 0), (0, T - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :T]
    return mfcc.T.astype(np.float32)   # (T, n_mfcc)


def predict_file(file_path: str, model, meta: dict, threshold: float) -> dict:
    """Run prediction on a single audio file. Returns a result dict."""
    cfg = meta.get("cfg", {})

    y   = load_audio(file_path,
                     sr=cfg.get("sr", 16_000),
                     n_samples=cfg.get("n_samples", 48_000))
    seq = extract_mfcc_sequence(y,
                                sr=cfg.get("sr", 16_000),
                                n_mfcc=cfg.get("n_mfcc", 40),
                                n_fft=cfg.get("n_fft", 512),
                                hop_length=cfg.get("hop_length", 160),
                                T=cfg.get("T", 300))
    x      = seq[np.newaxis, ...]                        # (1, T, n_mfcc)
    p_fake = float(model.predict(x, verbose=0)[0, 0])
    p_real = 1.0 - p_fake
    is_fake = p_fake >= threshold

    return {
        "file"      : str(file_path),
        "verdict"   : "DEEPFAKE" if is_fake else "GENUINE",
        "label"     : "Deepfake (AI-Generated)" if is_fake else "Genuine (Human)",
        "is_fake"   : is_fake,
        "confidence": round((p_fake if is_fake else p_real) * 100, 2),
        "p_fake"    : round(p_fake, 6),
        "p_real"    : round(p_real, 6),
        "threshold" : threshold,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════
DIVIDER = "─" * 52

def _verdict_color(is_fake: bool) -> tuple[str, str]:
    """Return ANSI color codes for the verdict (green=genuine, red=fake)."""
    if is_fake:
        return "\033[91m", "\033[0m"   # bright red
    return "\033[92m", "\033[0m"       # bright green


def print_result(result: dict):
    start, end = _verdict_color(result["is_fake"])
    icon = "🔴" if result["is_fake"] else "🟢"
    print(f"\n{DIVIDER}")
    print(f"  File       : {Path(result['file']).name}")
    print(f"  Verdict    : {start}{icon}  {result['verdict']} ({result['label'].split('(')[1][:-1]}){end}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  P(fake)    : {result['p_fake']:.4f}")
    print(f"  P(real)    : {result['p_real']:.4f}")
    print(f"  Threshold  : {result['threshold']}")
    print(DIVIDER)


def print_batch_summary(results: list[dict]):
    total   = len(results)
    n_fake  = sum(1 for r in results if r["is_fake"])
    n_real  = total - n_fake
    avg_conf = sum(r["confidence"] for r in results) / total if total else 0

    print(f"\n{'═'*52}")
    print(f"  BATCH SUMMARY  ({total} file{'s' if total != 1 else ''})")
    print(f"{'─'*52}")
    print(f"  Genuine  : {n_real:>4}  ({n_real/total*100:.1f}%)")
    print(f"  Deepfake : {n_fake:>4}  ({n_fake/total*100:.1f}%)")
    print(f"  Avg Conf : {avg_conf:.1f}%")
    print(f"{'═'*52}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test_audio.py",
        description="Test audio files against the deepfake detection ensemble.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Path to a single audio file (.wav / .flac / .mp3)",
    )
    group.add_argument(
        "--folder", "-d",
        metavar="DIR",
        help="Path to a folder — all supported audio files inside are tested",
    )
    p.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Decision threshold [0.0–1.0] (default: value from ensemble_meta.json, usually 0.535)\n"
             "Lower → more sensitive to fakes.  Higher → more conservative.",
    )
    p.add_argument(
        "--json", "-j",
        action="store_true",
        help="Print the result(s) as JSON instead of the formatted table",
    )
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Save JSON results to this file (e.g. results.json)",
    )
    return p


def collect_audio_files(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        sys.exit(f"\n[ERROR] Not a directory: {folder}\n")
    files = sorted(
        p for p in folder_path.iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    )
    if not files:
        sys.exit(
            f"\n[ERROR] No supported audio files found in: {folder}\n"
            f"  Supported formats: {', '.join(SUPPORTED_EXT)}\n"
        )
    return files


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    model, meta = load_model_and_meta()
    threshold   = args.threshold if args.threshold is not None \
                  else float(meta.get("threshold", 0.535))

    if not (0.0 <= threshold <= 1.0):
        sys.exit("\n[ERROR] Threshold must be between 0.0 and 1.0\n")

    # ── Collect files ───────────────────────────────────────────────────────
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            sys.exit(f"\n[ERROR] File not found: {args.file}\n")
        if file_path.suffix.lower() not in SUPPORTED_EXT:
            sys.exit(
                f"\n[ERROR] Unsupported format: '{file_path.suffix}'\n"
                f"  Supported: {', '.join(SUPPORTED_EXT)}\n"
            )
        audio_files = [file_path]
    else:
        audio_files = collect_audio_files(args.folder)
        print(f"\nFound {len(audio_files)} audio file(s) in '{args.folder}'")

    # ── Run predictions ─────────────────────────────────────────────────────
    results = []
    for i, fp in enumerate(audio_files, 1):
        if len(audio_files) > 1:
            print(f"[{i}/{len(audio_files)}] {fp.name} ...", end=" ", flush=True)
        try:
            result = predict_file(str(fp), model, meta, threshold)
            results.append(result)
            if len(audio_files) > 1:
                verdict_str = "DEEPFAKE" if result["is_fake"] else "GENUINE"
                print(f"{verdict_str}  ({result['confidence']}%)")
        except Exception as e:
            print(f"FAILED — {e}")
            results.append({"file": str(fp), "error": str(e)})

    # ── Output ──────────────────────────────────────────────────────────────
    if args.json:
        output = results[0] if args.file else results
        print(json.dumps(output, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"\n[ERROR] {r['file']}: {r['error']}")
            else:
                print_result(r)
        if len(results) > 1:
            valid = [r for r in results if "error" not in r]
            if valid:
                print_batch_summary(valid)

    # ── Save to file ─────────────────────────────────────────────────────────
    if args.output:
        output_data = results[0] if (args.file and len(results) == 1) else results
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
