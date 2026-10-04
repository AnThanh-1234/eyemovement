"""
Chạy Step 0 (Kmeans-SSE) + Step 1 (GMM-HMM tầng 1) của Algorithm 2, rồi visualize.

Cần để cùng thư mục: sample_create.py, Average_deviation.py, dataplot.py
Dùng:
    python run_round1.py                       # k tự chọn từ elbow, HMM khởi tạo từ K-means
    python run_round1.py --k 4
    python run_round1.py --init random --n_mix 2 --features xyv   # kiểu main.py cũ
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GMMHMM

import Average_deviation as AD
import dataplot

# cùng bảng màu với dataplot.path_classification_plot (key 0..5)
COLORS = ["#00DDAA", "#FF5511", "#0000CD", "#FFA500", "#C0C0C0", "#FFD700"]


def load_data(path):
    """Đọc file [N,3] (x,y,v) hoặc [N,4] (x,y,v,label). Trả về (feats [N,3], gt_label hoặc None)."""
    mat = np.atleast_2d(np.loadtxt(path))        # thêm delimiter=',' nếu là CSV
    if mat.shape[1] < 3:
        raise ValueError(f"File chỉ có {mat.shape[1]} cột, cần ít nhất 3 (x, y, v)")
    gt = mat[:, 3].astype(int) if mat.shape[1] >= 4 else None
    return mat[:, :3], gt


def make_features(datas, mode):
    """xy: chỉ tọa độ (đúng mô tả paper cho tầng 1). xyv: thêm v, có chuẩn hóa vì v ~1e-3 còn x,y ~1."""
    if mode == "xy":
        return datas[:, :2]
    return (datas - datas.mean(0)) / (datas.std(0) + 1e-12)


def kmeans_init_gmmhmm(X, km_labels, n_iter, stay=0.9):
    """GMM-HMM mà mỗi state khởi tạo từ 1 cụm K-means (mean, covariance), n_mix=1.
    State được đánh số theo thứ tự cụm xuất hiện đầu tiên theo thời gian."""
    order = []
    for l in km_labels:
        if l not in order:
            order.append(l)
    k, d = len(order), X.shape[1]
    means = np.zeros((k, 1, d))
    covars = np.zeros((k, 1, d, d))
    for s, l in enumerate(order):
        pts = X[km_labels == l]
        cov = np.cov(pts.T) if len(pts) > d else np.eye(d) * 1e-3
        means[s, 0] = pts.mean(0)
        covars[s, 0] = np.atleast_2d(cov) + 1e-4 * np.eye(d)
    transmat = np.full((k, k), (1 - stay) / max(k - 1, 1))
    np.fill_diagonal(transmat, stay)
    startprob = np.full(k, 0.1 / max(k - 1, 1))
    startprob[0] = 0.9
    startprob /= startprob.sum()

    model = GMMHMM(n_components=k, n_mix=1, covariance_type="full", n_iter=n_iter,
                   tol=1e-5, init_params="", params="stmcw", verbose=True)
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.weights_ = np.ones((k, 1))
    model.means_ = means
    model.covars_ = covars
    return model


def auto_elbow(Ks, centers, labels, xy):
    """Chọn k tại điểm xa đường nối đầu-cuối nhất của đường cong độ phân tán."""
    Ks = list(Ks)
    disp = np.array([np.linalg.norm(xy - centers[k][labels[k]], axis=1).mean() for k in Ks])
    if len(Ks) < 3:
        return Ks[0]
    x = (np.array(Ks) - Ks[0]) / (Ks[-1] - Ks[0])
    y = (disp - disp.min()) / (disp.max() - disp.min() + 1e-12)
    dist = np.abs(x + y - 1) / np.sqrt(2)
    return Ks[int(np.argmax(dist))]


def relabel_by_first_appearance(states):
    """Đổi nhãn sao cho cụm xuất hiện đầu tiên theo thời gian là C1, tiếp theo C2..."""
    mapping = {}
    out = np.empty(len(states), dtype=int)
    for i, s in enumerate(states):
        if s not in mapping:
            mapping[s] = len(mapping)
        out[i] = mapping[s]
    return out, mapping


def plot_clusters_2d(datas, labels, k):
    plt.figure(10)
    for c in range(k):
        m = labels == c
        if m.any():
            plt.scatter(datas[m, 0], datas[m, 1], s=25,
                        c=COLORS[c % len(COLORS)], label=f"C{c + 1} ({m.sum()} điểm)")
    plt.plot(datas[:, 0], datas[:, 1], color="gray", lw=0.4, alpha=0.5)
    plt.xlabel("X (0~1)")
    plt.ylabel("Y (0~1)")
    plt.title(f"GMM-HMM tầng 1: {k} cụm (2D x,y)")
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/testdataset/tester11_1.txt")
    ap.add_argument("--k", type=int, default=None, help="số cụm; bỏ trống = tự chọn từ elbow")
    ap.add_argument("--kmax", type=int, default=9)
    ap.add_argument("--n_mix", type=int, default=1, help="số mixture/state (chỉ dùng khi --init random)")
    ap.add_argument("--features", choices=["xy", "xyv"], default="xy")
    ap.add_argument("--init", choices=["kmeans", "random"], default="kmeans")
    ap.add_argument("--n_iter", type=int, default=50)
    args = ap.parse_args()

    # ---- Load data ----
    datas, gt = load_data(args.data)
    print(f"Đã đọc {len(datas)} điểm (x, y, v)" + (" + cột label" if gt is not None else ""))

    # ---- Step 0: Kmeans-SSE (Average_deviation) ----
    Ks = range(1, args.kmax + 1)
    centers, kl = AD.km_deviation(Ks, datas[:, 0:2])   # hiện đồ thị elbow
    k = args.k if args.k is not None else auto_elbow(Ks, centers, kl, datas[:, 0:2])
    print(f"Số cụm k = {k}" + (" (tự chọn từ elbow)" if args.k is None else " (người dùng chọn)"))
    if k > len(COLORS):
        print(f"Cảnh báo: dataplot chỉ có {len(COLORS)} màu, k={k} sẽ thiếu màu.")

    center_order = AD.Remove_redundant_data(kl[k], centers[k])
    AD.cluster_center_plot(datas[:, 0:2], center_order)

    # ---- Step 1: GMM-HMM tầng 1 ----
    X = make_features(datas, args.features)
    if args.init == "kmeans":
        model = kmeans_init_gmmhmm(X, kl[k], args.n_iter)
    else:
        model = GMMHMM(n_components=k, n_mix=args.n_mix, covariance_type="full",
                       n_iter=args.n_iter, tol=1e-5, random_state=0, verbose=True)
    model.fit(X)                           # Baum-Welch
    _, raw_states = model.decode(X)        # Viterbi
    labels, _ = relabel_by_first_appearance(raw_states)

    # ---- In chuỗi cụm ----
    print("\nChuỗi cụm theo thời gian:")
    print(", ".join(f"C{s + 1}" for s in labels))

    runs = []
    for s in labels:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    print("\nTóm tắt theo đoạn:")
    print(" -> ".join(f"C{s + 1}x{n}" for s, n in runs))

    if gt is not None:
        print("\nLabel gốc trong file (cột 4):")
        print(", ".join(str(g) for g in gt))

    np.savetxt("round1_clusters.txt", labels + 1, fmt="%d")
    print("\nĐã lưu nhãn vào round1_clusters.txt")

    # ---- Visualize ----
    plot_clusters_2d(datas, labels, k)
    states_info = {c: datas[labels == c] for c in range(k) if (labels == c).any()}
    dataplot.path_classification_plot(states_info)


if __name__ == "__main__":
    main()