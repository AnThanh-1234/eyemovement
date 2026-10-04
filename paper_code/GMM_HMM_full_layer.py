# -*- coding: utf-8 -*-
"""Ổn định hóa pipeline GMM-HMM hai tầng, không dùng Ground Truth để dự đoán.

Các bước:
1. Chọn K bằng Elbow trên SSE của K-Means tự cài đặt.
2. GMM-HMM tầng 1 với nhiều lần restart, chọn log-likelihood tốt nhất.
3. GMM-HMM tầng 2 (3 state) cho từng section đủ dài.
4. Gom mean velocity của các state trên toàn file, K-Means thành 3 nhóm.
5. Ánh xạ vận tốc thấp/trung bình/cao thành Fixation/Pursuit/Saccade.

Ground Truth, nếu có trong cột thứ tư, chỉ được dùng để đánh giá sau dự đoán.
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


CFG = {
    "seed": 42,
    "k_max": 9,
    "kmeans_max_iter": 100,
    "kmeans_restarts": 10,
    "l1_features": ("x", "y"),
    "l2_features": ("v",),
    "n_mix": 2,
    "covariance_type": "full",
    "n_iter": 50,
    "tol": 1e-5,
    "n_restarts": 5,
    "min_seg_len": 15,
    "standardize": True,
}


def load_data(path):
    matrix = np.atleast_2d(np.loadtxt(path))
    if matrix.shape[1] < 3:
        raise ValueError(f"{path} cần ít nhất 3 cột [x, y, v].")
    features = pd.DataFrame(matrix[:, :3], columns=("x", "y", "v"))
    ground_truth = (
        matrix[:, 3].astype(int) if matrix.shape[1] >= 4 else None
    )
    return features, ground_truth


def kmeans_sse(data, n_clusters, max_iter, restarts, rng):
    """Tính SSE bằng K-Means tự cài đặt, không dùng sklearn KMeans."""
    best_sse = np.inf
    for _ in range(restarts):
        centers = data[rng.choice(len(data), n_clusters, replace=False)].copy()
        for _ in range(max_iter):
            distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(2)
            labels = distances.argmin(1)
            new_centers = np.array([
                data[labels == cluster].mean(0)
                if np.any(labels == cluster) else centers[cluster]
                for cluster in range(n_clusters)
            ])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        sse = float(((data - centers[labels]) ** 2).sum())
        best_sse = min(best_sse, sse)
    return best_sse


def choose_k(data, cfg, requested_k=None):
    if requested_k is not None:
        if requested_k < 1 or requested_k > len(data):
            raise ValueError("K phải nằm trong khoảng [1, số điểm].")
        return requested_k

    max_k = min(cfg["k_max"], len(data))
    ks = np.arange(1, max_k + 1)
    if len(ks) < 3:
        return int(ks[-1])

    rng = np.random.default_rng(cfg["seed"])
    sse = np.array([
        kmeans_sse(
            data,
            int(k),
            cfg["kmeans_max_iter"],
            cfg["kmeans_restarts"],
            rng,
        )
        for k in ks
    ])
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (sse - sse.min()) / (sse.max() - sse.min() + 1e-12)
    distance = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y
                      + x[-1] * y[0] - y[-1] * x[0])
    return int(ks[int(distance.argmax())])


def kmeans_preview(data, n_clusters, cfg):
    """Tạo nhãn/tâm K-Means để trực quan hóa mà không đổi pipeline huấn luyện."""
    rng = np.random.default_rng(cfg["seed"])
    centers = data[rng.choice(len(data), n_clusters, replace=False)].copy()
    for _ in range(cfg["kmeans_max_iter"]):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = distances.argmin(1)
        new_centers = np.array([
            data[labels == cluster].mean(0)
            if np.any(labels == cluster) else centers[cluster]
            for cluster in range(n_clusters)
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def plot_kmeans(data, cfg, selected_k):
    """Hiển thị Elbow và kết quả K-Means trước khi chạy GMM-HMM."""
    max_k = min(cfg["k_max"], len(data))
    ks = np.arange(1, max_k + 1)
    rng = np.random.default_rng(cfg["seed"])
    sse = np.array([
        kmeans_sse(
            data, int(k), cfg["kmeans_max_iter"],
            cfg["kmeans_restarts"], rng
        )
        for k in ks
    ])

    plt.figure(figsize=(8, 5))
    plt.plot(ks, sse, "bo-", markerfacecolor="red")
    plt.axvline(selected_k, color="green", linestyle="--",
                label=f"Selected K={selected_k}")
    plt.xlabel("Number of clusters K")
    plt.ylabel("SSE")
    plt.title("Elbow method - K-Means")
    plt.legend()
    plt.tight_layout()
    plt.show()

    labels, centers = kmeans_preview(data, selected_k, cfg)
    plt.figure(figsize=(9, 6))
    for cluster in range(selected_k):
        mask = labels == cluster
        if np.any(mask):
            plt.scatter(
                data[mask, 0], data[mask, 1], s=18,
                label=f"Cluster {cluster}",
            )
    plt.scatter(
        centers[:, 0], centers[:, 1], c="black", marker="X",
        s=100, label="K-Means centers",
    )
    plt.plot(data[:, 0], data[:, 1], color="gray",
             alpha=0.3, linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"K-Means clustering before GMM-HMM (K={selected_k})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def fit_gmmhmm(data, n_states, cfg, seed):
    """Thử nhiều restart; nếu n_mix=2 thất bại thì fallback về n_mix=1."""
    mix_values = list(dict.fromkeys((cfg["n_mix"], 1)))
    for n_mix in mix_values:
        best_model = None
        best_score = -np.inf
        for restart in range(cfg["n_restarts"]):
            try:
                model = GMMHMM(
                    n_components=n_states,
                    n_mix=n_mix,
                    covariance_type=cfg["covariance_type"],
                    n_iter=cfg["n_iter"],
                    tol=cfg["tol"],
                    min_covar=1e-3,
                    random_state=seed + restart,
                    verbose=False,
                )
                model.fit(data)
                score = float(model.score(data))
                if np.isfinite(score) and score > best_score:
                    best_model, best_score = model, score
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
        if best_model is not None:
            return best_model
    return None


def scale(data, enabled):
    return StandardScaler().fit_transform(data) if enabled else data


def velocity_mapping(stats, cfg):
    """Tạo mapping state cục bộ -> lớp vận tốc từ toàn bộ section."""
    velocities = np.log10(np.maximum(np.asarray(stats)[:, 2], 0) + 1e-6)
    if len(velocities) >= 3:
        model = KMeans(
            n_clusters=3,
            n_init=10,
            random_state=cfg["seed"],
        ).fit(velocities.reshape(-1, 1))
        order = np.argsort(model.cluster_centers_.ravel())
        cluster_to_class = {
            int(cluster): rank for rank, cluster in enumerate(order)
        }
        return {
            (int(section), int(state)): cluster_to_class[int(cluster)]
            for (section, state, _), cluster in zip(stats, model.labels_)
        }, model.cluster_centers_.ravel()[order]

    # Không đủ state để chạy K-Means 3 cụm: xếp trực tiếp theo vận tốc.
    order = np.argsort(velocities)
    return {
        (int(stats[index][0]), int(stats[index][1])): min(rank, 2)
        for rank, index in enumerate(order)
    }, velocities[order]


def classify(data, cfg, requested_k=None):
    xy = data[["x", "y"]].to_numpy()
    k = choose_k(xy, cfg, requested_k)
    layer1_input = scale(data[list(cfg["l1_features"])].to_numpy(),
                         cfg["standardize"])
    layer1 = fit_gmmhmm(layer1_input, k, cfg, cfg["seed"])
    if layer1 is None:
        raise RuntimeError("Không fit được GMM-HMM tầng 1 sau mọi restart.")

    _, segment_states = layer1.decode(layer1_input, algorithm="viterbi")
    velocity = data["v"].to_numpy()
    local_states = np.full(len(data), -1, dtype=int)
    stats = []
    tiny_sections = []

    for section in np.unique(segment_states):
        indices = np.flatnonzero(segment_states == section)
        if len(indices) < cfg["min_seg_len"]:
            tiny_sections.append(indices)
            continue
        layer2_input = scale(
            data.iloc[indices][list(cfg["l2_features"])].to_numpy(),
            cfg["standardize"],
        )
        layer2 = fit_gmmhmm(
            layer2_input,
            3,
            cfg,
            cfg["seed"] + 100 + int(section),
        )
        if layer2 is None:
            tiny_sections.append(indices)
            continue
        _, states = layer2.decode(layer2_input, algorithm="viterbi")
        local_states[indices] = states
        for state in np.unique(states):
            state_velocity = velocity[indices[states == state]]
            stats.append((section, state, float(state_velocity.mean())))

    if not stats:
        raise RuntimeError("Không có section đủ dữ liệu để phân loại tầng 2.")

    mapping, centers = velocity_mapping(stats, cfg)
    prediction = np.full(len(data), -1, dtype=int)
    for index in np.flatnonzero(local_states >= 0):
        key = (int(segment_states[index]), int(local_states[index]))
        if key not in mapping:
            raise RuntimeError(f"Thiếu mapping cho state {key}.")
        prediction[index] = mapping[key]

    # Section ngắn/fail dùng tâm velocity global, không dùng Ground Truth.
    for indices in tiny_sections:
        distances = np.abs(
            np.log10(np.maximum(velocity[indices], 0) + 1e-6)[:, None]
            - centers[None, :]
        )
        prediction[indices] = distances.argmin(1).clip(max=2)

    if np.any(prediction < 0):
        raise RuntimeError("Một số điểm chưa được gán nhãn dự đoán.")
    return prediction, k, len(tiny_sections), segment_states, local_states


def plot_sections(data, segment_states, k):
    """Hiển thị phân đoạn tầng 1 trên mặt phẳng x-y."""
    plt.figure(figsize=(9, 6))
    for section in range(k):
        mask = segment_states == section
        if np.any(mask):
            plt.scatter(
                data.loc[mask, "x"],
                data.loc[mask, "y"],
                s=18,
                label=f"Section {section}",
            )
    plt.plot(data["x"], data["y"], color="gray", alpha=0.35, linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GMM-HMM tầng 1: phân đoạn chuỗi điểm nhìn")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_micro_states(data, local_states):
    """Hiển thị state thô của GMM-HMM tầng 2 trước semantic mapping."""
    plt.figure(figsize=(9, 6))
    for state in range(3):
        mask = local_states == state
        if np.any(mask):
            plt.scatter(
                data.loc[mask, "x"],
                data.loc[mask, "y"],
                s=18,
                label=f"Micro state {state}",
            )
    plt.plot(data["x"], data["y"], color="gray",
             alpha=0.35, linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GMM-HMM tầng 2: state thô trước khi gán lớp")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_classification(data, prediction, ground_truth=None):
    """Hiển thị nhãn dự đoán và Ground Truth nếu file có cột thứ tư."""
    plots = [("Kết quả phân loại GMM-HMM", prediction)]
    if ground_truth is not None:
        plots.append(("Ground Truth", ground_truth))

    fig, axes = plt.subplots(
        1, len(plots), figsize=(8 * len(plots), 6), squeeze=False
    )
    for axis, (title, labels) in zip(axes[0], plots):
        labels = np.asarray(labels)
        for label in range(3):
            mask = labels == label
            if np.any(mask):
                axis.scatter(
                    data.loc[mask, "x"],
                    data.loc[mask, "y"],
                    s=20,
                    label=("Fixation", "Smooth Pursuit", "Saccade")[label],
                )
        axis.plot(data["x"], data["y"], color="gray", alpha=0.35, linewidth=0.5)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_title(title)
        axis.legend()
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    axis = fig.add_subplot(111, projection="3d")
    for label in range(3):
        mask = np.asarray(prediction) == label
        if np.any(mask):
            axis.scatter(
                data.loc[mask, "x"],
                data.loc[mask, "y"],
                data.loc[mask, "v"],
                s=18,
                label=("Fixation", "Smooth Pursuit", "Saccade")[label],
            )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Velocity")
    axis.set_title("Kết quả phân loại 3D")
    axis.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(
            Path(__file__).parent / "dataset" / "testdataset" / "tester11_1.txt"
        ),
        help="Một file dữ liệu [x, y, v] hoặc [x, y, v, ground_truth].",
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=CFG["seed"])
    parser.add_argument("--min-seg-len", type=int, default=CFG["min_seg_len"])
    args = parser.parse_args()

    cfg = CFG.copy()
    cfg["seed"] = args.seed
    cfg["min_seg_len"] = args.min_seg_len
    path = Path(args.data)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    data, ground_truth = load_data(path)
    print(f"[*] Loaded {len(data)} points from '{path}'.")
    cfg["seed"] = args.seed
    prediction, k, tiny, segment_states, local_states = classify(
        data, cfg, args.k
    )
    print(f"[*] Selected K: {k}")
    print(f"[*] Fallback sections: {tiny}")
    plot_kmeans(data[["x", "y"]].to_numpy(), cfg, k)
    plot_sections(data, segment_states, k)
    plot_micro_states(data, local_states)
    plot_classification(data, prediction, ground_truth)

    if ground_truth is not None:
        accuracy = accuracy_score(ground_truth, prediction)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth,
            prediction,
            labels=[0, 1, 2],
            zero_division=0,
        )
        print(f"[*] Accuracy: {accuracy * 100:.2f}%")
        print(f"[*] Macro Precision: {precision.mean():.4f}")
        print(f"[*] Macro Recall: {recall.mean():.4f}")
        print(f"[*] Macro F1: {f1.mean():.4f}")


if __name__ == "__main__":
    main()
