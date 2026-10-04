# -*- coding: utf-8 -*-
"""
Chương trình: GMM_HMM_layer1_eva.py
Chức năng:
- Chạy Step 0 (Kmeans-SSE / Average_deviation) + Step 1 (GMM-HMM tầng 1) của Algorithm 2.
- Phân đoạn chuỗi chuyển động mắt thành k cụm/phân đoạn theo thời gian.
- Trực quan hóa kết quả dự đoán và vẽ nhãn thật (Ground Truth) bên cạnh để xem thử và so sánh:
  + Biểu đồ 2D (X, Y): Dự đoán GMM-HMM vs Nhãn thật (Ground Truth).
  + Biểu đồ 3D (X, Y, V): Dự đoán GMM-HMM vs Nhãn thật (Ground Truth).

Cách dùng:
    python GMM_HMM_layer1_eva.py                                     # Mặc định lấy tester11_3.txt, k tự chọn từ elbow
    python GMM_HMM_layer1_eva.py --data dataset/testdataset/tester01_1.txt
    python GMM_HMM_layer1_eva.py --k 4                               # Chỉ định k = 4 cụm
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hmmlearn.hmm import GMMHMM

import Average_deviation as AD
import dataplot

# Bảng màu chuẩn dùng cho các phân đoạn/nhãn
COLORS = [
    "#00DDAA", "#FF5511", "#0000CD", "#FFA500", "#9932CC",
    "#FFD700", "#00CED1", "#FF1493", "#32CD32", "#808080"
]


def load_data(path):
    """
    Đọc file dữ liệu điểm nhìn [N, 3] (x, y, v) hoặc [N, 4] (x, y, v, label).
    Trả về:
        feats (np.ndarray): Ma trận đặc trưng [N, 3] chứa [x, y, v].
        gt (np.ndarray hoặc None): Nhãn thực tế [N] nếu file có cột thứ 4.
    """
    mat = np.atleast_2d(np.loadtxt(path))
    if mat.shape[1] < 3:
        raise ValueError(f"File chỉ có {mat.shape[1]} cột, cần ít nhất 3 cột (x, y, v)")
    
    feats = mat[:, :3]
    gt = mat[:, 3].astype(int) if mat.shape[1] >= 4 else None
    return feats, gt


def make_features(datas, mode):
    """
    Trích xuất đặc trưng cho tầng 1:
    - 'xy': Chỉ lấy tọa độ 2D [x, y] (chuẩn theo mô tả của bài báo).
    - 'xyv': Lấy cả 3 chiều [x, y, v] và chuẩn hóa Z-score.
    """
    if mode == "xy":
        return datas[:, :2]
    return (datas - datas.mean(0)) / (datas.std(0) + 1e-12)


def kmeans_init_gmmhmm(X, km_labels, n_iter, stay=0.9):
    """
    Khởi tạo GMM-HMM với các trạng thái lấy từ tâm và hiệp phương sai của cụm K-Means.
    Các trạng thái được sắp xếp theo trình tự thời gian xuất hiện đầu tiên.
    """
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

    model = GMMHMM(
        n_components=k,
        n_mix=1,
        covariance_type="full",
        n_iter=n_iter,
        tol=1e-5,
        init_params="",
        params="stmcw",
        verbose=False
    )
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.weights_ = np.ones((k, 1))
    model.means_ = means
    model.covars_ = covars
    return model


def auto_elbow(Ks, centers, labels, xy):
    """
    Tự động chọn số cụm k tại điểm khuỷu tay (Elbow point) xa đường chéo nhất.
    """
    Ks = list(Ks)
    disp = np.array([np.linalg.norm(xy - centers[k][labels[k]], axis=1).mean() for k in Ks])
    if len(Ks) < 3:
        return Ks[0]
    
    x = (np.array(Ks) - Ks[0]) / (Ks[-1] - Ks[0])
    y = (disp - disp.min()) / (disp.max() - disp.min() + 1e-12)
    dist = np.abs(x + y - 1) / np.sqrt(2)
    return Ks[int(np.argmax(dist))]


def relabel_by_first_appearance(states):
    """
    Đánh lại nhãn cụm sao cho cụm xuất hiện đầu tiên theo thời gian là C1, tiếp theo là C2, C3...
    """
    mapping = {}
    out = np.empty(len(states), dtype=int)
    for i, s in enumerate(states):
        if s not in mapping:
            mapping[s] = len(mapping)
        out[i] = mapping[s]
    return out, mapping


def plot_clusters_2d_compare(datas, pred_labels, gt_labels, k):
    """
    Vẽ biểu đồ 2D (X, Y):
    - Nếu có cột nhãn thật: Vẽ 2 hình cạnh nhau (Dự đoán vs Nhãn thật).
    - Nếu không có nhãn thật: Chỉ vẽ 1 hình dự đoán.
    """
    if gt_labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Hình 1 (Trái): Dự đoán GMM-HMM Tầng 1
        ax1 = axes[0]
        for c in range(k):
            m = pred_labels == c
            if m.any():
                ax1.scatter(datas[m, 0], datas[m, 1], s=25,
                            c=COLORS[c % len(COLORS)], label=f"C{c + 1} ({m.sum()} điểm)")
        ax1.plot(datas[:, 0], datas[:, 1], color="gray", lw=0.4, alpha=0.5)
        ax1.set_xlabel("X (0~1)")
        ax1.set_ylabel("Y (0~1)")
        ax1.set_title(f"GMM-HMM Tầng 1 (Dự đoán: {k} cụm)", fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.4)
        
        # Hình 2 (Phải): Nhãn thật (Ground Truth)
        ax2 = axes[1]
        gt_unique = np.unique(gt_labels)
        for idx, g in enumerate(gt_unique):
            m = gt_labels == g
            if m.any():
                ax2.scatter(datas[m, 0], datas[m, 1], s=25,
                            c=COLORS[idx % len(COLORS)], label=f"Nhãn thật {g} ({m.sum()} điểm)")
        ax2.plot(datas[:, 0], datas[:, 1], color="gray", lw=0.4, alpha=0.5)
        ax2.set_xlabel("X (0~1)")
        ax2.set_ylabel("Y (0~1)")
        ax2.set_title(f"Nhãn thật trong file (Ground Truth: {len(gt_unique)} nhóm)", fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        plt.suptitle("So sánh 2D: Dự đoán phân cụm Tầng 1 vs Nhãn thật", fontsize=15)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        for c in range(k):
            m = pred_labels == c
            if m.any():
                plt.scatter(datas[m, 0], datas[m, 1], s=25,
                            c=COLORS[c % len(COLORS)], label=f"C{c + 1} ({m.sum()} điểm)")
        plt.plot(datas[:, 0], datas[:, 1], color="gray", lw=0.4, alpha=0.5)
        plt.xlabel("X (0~1)")
        plt.ylabel("Y (0~1)")
        plt.title(f"GMM-HMM Tầng 1: {k} cụm (2D x,y)", fontsize=13, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()


def plot_clusters_3d_compare(datas, pred_labels, gt_labels, k):
    """
    Vẽ biểu đồ không gian 3D (X, Y, V):
    - Nếu có cột nhãn thật: Vẽ 2 hình 3D cạnh nhau (Dự đoán vs Nhãn thật).
    - Nếu không có nhãn thật: Chỉ vẽ 1 hình 3D dự đoán.
    """
    if gt_labels is not None:
        fig = plt.figure(figsize=(16, 7))
        
        # Subplot 1 (Trái): Dự đoán 3D
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        for c in range(k):
            m = pred_labels == c
            if m.any():
                ax1.scatter(datas[m, 0], datas[m, 1], datas[m, 2],
                            zdir="z", c=COLORS[c % len(COLORS)], marker="o", s=30, label=f"C{c + 1}")
        ax1.set_xlabel("X (0~1)")
        ax1.set_ylabel("Y (0~1)")
        ax1.set_zlabel("Velocity (V)")
        ax1.set_title("Dự đoán phân đoạn 3D (GMM-HMM)", fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')
        
        # Subplot 2 (Phải): Nhãn thật 3D
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        gt_unique = np.unique(gt_labels)
        for idx, g in enumerate(gt_unique):
            m = gt_labels == g
            if m.any():
                ax2.scatter(datas[m, 0], datas[m, 1], datas[m, 2],
                            zdir="z", c=COLORS[idx % len(COLORS)], marker="o", s=30, label=f"Nhãn thật {g}")
        ax2.set_xlabel("X (0~1)")
        ax2.set_ylabel("Y (0~1)")
        ax2.set_zlabel("Velocity (V)")
        ax2.set_title("Nhãn thật 3D (Ground Truth)", fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right')
        
        plt.suptitle("So sánh không gian 3D (X, Y, V): Dự đoán vs Nhãn thật", fontsize=15)
        plt.tight_layout()
        plt.show()
    else:
        states_info = {c: datas[pred_labels == c] for c in range(k) if (pred_labels == c).any()}
        dataplot.path_classification_plot(states_info)


def main():
    ap = argparse.ArgumentParser(description="Chạy Step 0 + Step 1 (GMM-HMM tầng 1) và hiển thị so sánh với nhãn thật")
    ap.add_argument("--data", default="dataset/testdataset/tester11_1.txt", help="Đường dẫn file dữ liệu")
    ap.add_argument("--k", type=int, default=None, help="Số cụm k; bỏ trống = tự chọn từ elbow")
    ap.add_argument("--kmax", type=int, default=9, help="Số cụm tối đa khảo sát K-means Elbow")
    ap.add_argument("--n_mix", type=int, default=1, help="Số mixture/state (khi dùng init random)")
    ap.add_argument("--features", choices=["xy", "xyv"], default="xy", help="Bộ đặc trưng (xy hoặc xyv)")
    ap.add_argument("--init", choices=["kmeans", "random"], default="kmeans", help="Cách khởi tạo mô hình")
    ap.add_argument("--n_iter", type=int, default=50, help="Số vòng lặp Baum-Welch")
    args = ap.parse_args()

    # ---- 1. Đọc dữ liệu ----
    datas, gt = load_data(args.data)
    print(f"Đã đọc {len(datas)} điểm từ '{args.data}'" + (" (có cột nhãn thật)" if gt is not None else " (không có nhãn thật)"))

    # ---- 2. Step 0: Kmeans-SSE (Average_deviation) ----
    Ks = range(1, args.kmax + 1)
    centers, kl = AD.km_deviation(Ks, datas[:, 0:2])   # Hiển thị đồ thị Elbow
    k = args.k if args.k is not None else auto_elbow(Ks, centers, kl, datas[:, 0:2])
    print(f"Số cụm k = {k}" + (" (tự chọn từ elbow)" if args.k is None else " (người dùng chọn)"))
    if k > len(COLORS):
        print(f"Cảnh báo: dataplot chỉ có {len(COLORS)} màu, k={k} sẽ xoay vòng màu.")

    center_order = AD.Remove_redundant_data(kl[k], centers[k])
    AD.cluster_center_plot(datas[:, 0:2], center_order)

    # ---- 3. Step 1: Huấn luyện GMM-HMM tầng 1 ----
    X = make_features(datas, args.features)
    if args.init == "kmeans":
        model = kmeans_init_gmmhmm(X, kl[k], args.n_iter)
    else:
        model = GMMHMM(n_components=k, n_mix=args.n_mix, covariance_type="full",
                       n_iter=args.n_iter, tol=1e-5, random_state=0, verbose=True)
    model.fit(X)                           # Baum-Welch
    _, raw_states = model.decode(X)        # Viterbi
    labels, _ = relabel_by_first_appearance(raw_states)

    # ---- 4. In chuỗi nhãn phân đoạn ----
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
        print("\nNhãn thật gốc trong file (cột 4):")
        print(", ".join(str(g) for g in gt))

    np.savetxt("round1_clusters.txt", labels + 1, fmt="%d")
    print("\nĐã lưu nhãn vào round1_clusters.txt")

    # ---- 5. Trực quan hóa so sánh cạnh nhau (Dự đoán vs Nhãn thật) ----
    print("\nĐang mở biểu đồ 2D và 3D (Dự đoán vs Nhãn thật)...")
    plot_clusters_2d_compare(datas, labels, gt, k)
    plot_clusters_3d_compare(datas, labels, gt, k)


if __name__ == "__main__":
    main()