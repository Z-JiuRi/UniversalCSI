import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def model_parts(name):
    if name.endswith("_cnn_residual"):
        return name[:-len("_cnn_residual")], "cnn_residual"
    if name.endswith("_transnet"):
        return name[:-len("_transnet")], "transnet"
    if name.endswith("_hybrid"):
        return name[:-len("_hybrid")], "hybrid"
    return name, "unknown"


def find_pair(pairwise, left, right):
    a, b = sorted([left, right])
    for row in pairwise:
        if row["left"] == a and row["right"] == b:
            return row
    return None


def top_rows(rows, key, reverse=True, limit=10):
    return sorted(rows, key=lambda row: f(row, key), reverse=reverse)[:limit]


def render_split(split_dir):
    split = split_dir.name
    summary = read_csv(split_dir / "summary.csv")
    pairwise = read_csv(split_dir / "pairwise_distances.csv")

    by_encoder = defaultdict(list)
    by_decoder = defaultdict(list)
    for row in summary:
        encoder, decoder = model_parts(row["name"])
        row["encoder"] = encoder
        row["decoder"] = decoder
        by_encoder[encoder].append(row)
        by_decoder[decoder].append(row)

    lines = [
        f"## Split: `{split}`",
        "",
        f"Models: `{len(summary)}`",
        "",
        "### Code Scale Ranking",
        "",
        "| rank | model | std | l2_norm_mean | effective_rank_32 | top5_pca |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(top_rows(summary, "std", reverse=True), start=1):
        lines.append(
            f"| {idx} | {row['name']} | {f(row, 'std'):.4e} | "
            f"{f(row, 'l2_norm_mean'):.4e} | "
            f"{f(row, 'effective_rank_32'):.4e} | "
            f"{f(row, 'pca_top5_ratio'):.4e} |"
        )

    lines.extend([
        "",
        "### Highest Effective-Rank Code Spaces",
        "",
        "| rank | model | effective_rank_32 | top10_pca | dim_std_mean |",
        "|---:|---|---:|---:|---:|",
    ])
    for idx, row in enumerate(top_rows(summary, "effective_rank_32",
                                       reverse=True), start=1):
        lines.append(
            f"| {idx} | {row['name']} | {f(row, 'effective_rank_32'):.4e} | "
            f"{f(row, 'pca_top10_ratio'):.4e} | "
            f"{f(row, 'dim_std_mean'):.4e} |"
        )

    lines.extend([
        "",
        "### Decoder Effect Within Same Encoder",
        "",
        "| encoder | transnet vs cnn_residual | transnet vs hybrid | cnn_residual vs hybrid |",
        "|---|---:|---:|---:|",
    ])
    for encoder in sorted(by_encoder):
        models = {row["decoder"]: row["name"] for row in by_encoder[encoder]}
        values = []
        for left, right in [
            ("transnet", "cnn_residual"),
            ("transnet", "hybrid"),
            ("cnn_residual", "hybrid"),
        ]:
            if left in models and right in models:
                row = find_pair(pairwise, models[left], models[right])
                values.append(f"{f(row, 'centroid_l2'):.4e}" if row else "NA")
            else:
                values.append("NA")
        lines.append(f"| {encoder} | {values[0]} | {values[1]} | {values[2]} |")

    lines.extend([
        "",
        "### Encoder Effect Within Same Decoder",
        "",
        "| decoder | mean centroid_l2 | max centroid_l2 | max pair |",
        "|---|---:|---:|---|",
    ])
    for decoder in sorted(by_decoder):
        names = sorted(row["name"] for row in by_decoder[decoder])
        rows = []
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                row = find_pair(pairwise, left, right)
                if row:
                    rows.append(row)
        if not rows:
            continue
        mean_value = sum(f(row, "centroid_l2") for row in rows) / len(rows)
        max_row = max(rows, key=lambda row: f(row, "centroid_l2"))
        lines.append(
            f"| {decoder} | {mean_value:.4e} | "
            f"{f(max_row, 'centroid_l2'):.4e} | "
            f"{max_row['left']} vs {max_row['right']} |"
        )

    lines.extend([
        "",
        "### Closest Architecture Pairs",
        "",
        "| rank | left | right | centroid_l2 | centroid_cosine | std_profile_l2 |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for idx, row in enumerate(top_rows(pairwise, "centroid_l2",
                                       reverse=False), start=1):
        lines.append(
            f"| {idx} | {row['left']} | {row['right']} | "
            f"{f(row, 'centroid_l2'):.4e} | "
            f"{f(row, 'centroid_cosine'):.4e} | "
            f"{f(row, 'std_profile_l2'):.4e} |"
        )

    lines.extend([
        "",
        "### Farthest Architecture Pairs",
        "",
        "| rank | left | right | centroid_l2 | centroid_cosine | std_profile_l2 |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for idx, row in enumerate(top_rows(pairwise, "centroid_l2",
                                       reverse=True), start=1):
        lines.append(
            f"| {idx} | {row['left']} | {row['right']} | "
            f"{f(row, 'centroid_l2'):.4e} | "
            f"{f(row, 'centroid_cosine'):.4e} | "
            f"{f(row, 'std_profile_l2'):.4e} |"
        )
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Summarize codeword analysis CSV files into Markdown.")
    parser.add_argument("--analysis_root",
                        default="exps/real_matrix_2epoch/codeword_analysis")
    parser.add_argument("--out",
                        default="exps/real_matrix_2epoch/codeword_analysis/summary_report.md")
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root)
    split_dirs = [
        path for path in [analysis_root / "train"]
        if (path / "summary.csv").is_file()
        and (path / "pairwise_distances.csv").is_file()
    ]
    if not split_dirs:
        raise SystemExit(f"No analysis CSV files found under {analysis_root}")

    lines = [
        "# Codeword Cross-Architecture Summary",
        "",
        "This report is generated from the per-split `summary.csv` and "
        "`pairwise_distances.csv` files.",
        "",
    ]
    for split_dir in split_dirs:
        lines.extend(render_split(split_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
