# save_kernelbook.py
import os
import argparse
import re
from datasets import load_dataset
import matplotlib.pyplot as plt

def slugify(s: str) -> str:
    """Make a safe filename from a string."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="kernelbook_py", help="Output directory for .py files and plot")
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to load_dataset")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load train split
    ds = load_dataset(
        "GPUMODE/KernelBook",
        split="train",
        trust_remote_code=args.trust_remote_code
    )

    # Zero-padding for filenames
    total = len(ds)
    pad_width = max(1, len(str(max(0, total - 1))))

    line_counts = []
    saved = 0

    for i, ex in enumerate(ds):
        code = ex.get("python_code")
        assert code, f"Missing python_code for item {i}"

        entry_point = ex.get("entry_point")
        assert entry_point, f"Missing entry_point for item {i}"

        # Save file: <padded_index>_<entry_point>.py
        safe_name = slugify(entry_point)
        filename = f"{str(i).zfill(pad_width)}_{safe_name}.py"
        path = os.path.join(args.outdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        saved += 1

        # Count lines
        line_counts.append(len(code.strip("\n").splitlines()))

    print(f"Saved {saved} files to {args.outdir}")

    # Plot histogram of line counts (limit x-axis to 200)
    plt.figure(figsize=(9, 5))
    plt.hist(line_counts, bins=300, edgecolor="black")
    plt.xlabel("Number of lines in python_code")
    plt.ylabel("Frequency")
    plt.title("KernelBook (train): Python code line count distribution")
    plt.xlim(0, 200)  # cap x-axis at 200
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    png_path = os.path.join(args.outdir, "kernelbook_linecount_hist.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved histogram to {png_path}")

    try:
        plt.show()  # No-op if headless
    except Exception:
        pass

if __name__ == "__main__":
    main()

