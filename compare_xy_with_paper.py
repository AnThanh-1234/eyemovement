import ast
import os
import sys

import pandas as pd


sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_PATH = os.path.join(BASE_DIR, "grid_search_results.csv")
REPORT_PATH = os.path.join(BASE_DIR, "xy_vs_jemr_comparison.csv")

# Table 1, "Proposed", from J. Eye Mov. Res. 2025, 18, 28, p. 10.
PAPER = {
    "Accuracy": (0.9439, 0.0208),
    "Precision": (0.9531, 0.0371),
    "Recall": (0.9498, 0.0393),
    "F1": (0.9492, 0.0238),
}


def parse_features(value):
    """Parse the CSV representation of a Python list safely."""
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"L1_FEATURES is not a list: {value!r}")
    return parsed


def main():
    if not os.path.exists(GRID_PATH):
        raise FileNotFoundError(f"Không tìm thấy: {GRID_PATH}")

    grid = pd.read_csv(GRID_PATH)
    required = {
        "L1_FEATURES",
        "Mean_Accuracy",
        "Std_Accuracy",
        "Mean_F1",
        "Mean_Precision",
        "Mean_Recall",
    }
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"grid_search_results.csv thiếu cột: {sorted(missing)}")

    grid["_features"] = grid["L1_FEATURES"].map(parse_features)
    xy = grid[
        grid["_features"].map(lambda features: features == ["x", "y"])
    ].copy()
    if xy.empty:
        raise ValueError("Không có cấu hình tầng 1 dùng đúng ['x', 'y'].")

    best = xy.sort_values(
        by=["Mean_Accuracy", "Mean_F1"],
        ascending=False,
    ).iloc[0]

    current = {
        "Accuracy": (best["Mean_Accuracy"], best["Std_Accuracy"]),
        "Precision": (best["Mean_Precision"], None),
        "Recall": (best["Mean_Recall"], None),
        "F1": (best["Mean_F1"], None),
    }

    rows = []
    for metric, (value, std) in current.items():
        paper_mean, paper_std = PAPER[metric]
        rows.append(
            {
                "Metric": metric,
                "XY_best_mean": value,
                "XY_best_std": std,
                "Paper_mean": paper_mean,
                "Paper_std": paper_std,
                "Difference_percentage_points": (value - paper_mean) * 100,
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)

    print("=" * 72)
    print("SO SÁNH CẤU HÌNH TẦNG 1 ['x', 'y'] VỚI PAPER JEMR")
    print("=" * 72)
    print(f"Grid search: {GRID_PATH}")
    print("Paper: J. Eye Mov. Res. 2025, 18, 28, Table 1, Proposed")
    print("\nCấu hình ['x', 'y'] tốt nhất theo Mean_Accuracy:")
    for column in [
        "L1_FEATURES",
        "COV_TYPE",
        "N_MIX",
        "MIN_SEG_LEN",
        "STANDARDIZE",
        "Mean_Accuracy",
        "Std_Accuracy",
        "Mean_F1",
        "Mean_Precision",
        "Mean_Recall",
    ]:
        print(f"  {column}: {best[column]}")

    print("\nSo sánh (đơn vị chênh lệch: điểm phần trăm):")
    for row in rows:
        xy_mean = row["XY_best_mean"] * 100
        paper_mean = row["Paper_mean"] * 100
        difference = row["Difference_percentage_points"]
        print(
            f"  {row['Metric']:<10} "
            f"grid={xy_mean:6.2f}% | paper={paper_mean:6.2f}% | "
            f"chênh lệch={difference:+6.2f} điểm"
        )

    print(f"\nĐã lưu báo cáo: {REPORT_PATH}")
    print(
        "\nLưu ý: đây là so sánh tham khảo. Grid search hiện tại dùng "
        "trung bình macro trên các file dataset của bạn, còn paper báo cáo "
        "kết quả Proposed theo thiết lập thực nghiệm của tác giả."
    )


if __name__ == "__main__":
    main()
