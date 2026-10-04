# -*- coding: utf-8 -*-
"""
Module: Average_deviation.py
Chức năng:
- Xác định số lượng phân đoạn / số cụm tối ưu K (Optimal Number of Clusters) bằng thuật toán K-Means
  và phương pháp Elbow (độ lệch bình phương trung bình / SSE).
- Trích xuất và sắp xếp các tâm cụm (Cluster Centers) theo thứ tự xuất hiện theo thời gian của đường nhìn.
- Trực quan hóa các tâm phân đoạn (Section centers) và các điểm nhìn (Gaze points) trên mặt phẳng 2D.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import sample_create
from matplotlib import rcParams

# Cấu hình font chữ và hiển thị cho biểu đồ Matplotlib
config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
    'axes.unicode_minus': False  # Hiển thị đúng dấu trừ âm
}


def km_deviation(Ks, datas):
    """
    Tính toán độ lệch phân tán trung bình (Mean Dispersion / SSE) cho các giá trị K khác nhau
    để tìm số lượng cụm tối ưu theo phương pháp Elbow.
    
    Tham số:
        Ks (iterable): Danh sách các giá trị K cần thử nghiệm (ví dụ: range(1, 10)).
        datas (np.ndarray): Ma trận dữ liệu điểm nhìn (tọa độ không gian 2D [x, y]).
        
    Trả về:
        cluster_center (dict): Từ điển chứa tọa độ các tâm cụm ứng với từng K.
        cluster_labels (dict): Từ điển chứa nhãn phân cụm cho từng điểm dữ liệu ứng với từng K.
    """
    meanDispersions = []  # Danh sách lưu độ lệch phân tán trung bình tương ứng với từng K
    cluster_center = {}   # Lưu tâm cụm cho từng K
    cluster_labels = {}   # Lưu nhãn cụm cho từng K

    # Lặp qua từng số cụm k trong danh sách Ks
    for k in Ks:
        # Khởi tạo mô hình K-Means với k cụm
        # init='k-means++': Chọn tâm khởi đầu thông minh giúp tăng tốc hội tụ
        # n_init=10: Chạy 10 lần với các tâm khởi đầu khác nhau để chọn kết quả tốt nhất
        # max_iter=300: Số lần lặp tối đa cho một lần chạy
        km = KMeans(n_clusters=k,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)

        # Huấn luyện K-Means trên dữ liệu
        km.fit(datas)
        cluster_center[k] = km.cluster_centers_
        cluster_labels[k] = km.labels_

        # Tính khoảng cách Euclidean từ mỗi điểm dữ liệu đến tâm cụm gần nhất của nó
        # cdist(datas, km.cluster_centers_, 'euclidean'): Ma trận khoảng cách [N, k]
        # np.min(..., axis=1): Khoảng cách tới tâm cụm gần nhất của từng điểm
        # sum(...) / datas.shape[0]: Khoảng cách trung bình của toàn bộ tập dữ liệu
        meanDispersions.append(sum(
            np.min(cdist(datas, km.cluster_centers_, 'euclidean'), axis=1)) / datas.shape[0])

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Times New Roman', 'SimHei']
    # plt.figure(facecolor='lightyellow')

    # Vẽ biểu đồ đường thể hiện mối quan hệ giữa số cụm K và tổng sai số bình phương / độ lệch trung bình
    plt.plot(Ks, meanDispersions, 'bo-', mfc='r')
    plt.xlabel('Number of cluster centers k', fontsize=14)
    plt.ylabel('Sum of the Squared Errors', fontsize=14)
    plt.title('The Optimal Number of Clusters in K-means', fontsize=14)
    plt.tick_params(labelsize=13)

    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()
    return cluster_center, cluster_labels


def cluster_center_plot(datas, center):
    """
    Vẽ biểu đồ phân bố không gian 2D của các điểm nhìn cùng các tâm phân đoạn.
    
    Tham số:
        datas (np.ndarray): Dữ liệu điểm nhìn thô [x, y].
        center (np.ndarray): Tọa độ các tâm phân đoạn đã được sắp xếp theo trình tự thời gian.
    """
    x = datas[:, 0]  # Tọa độ X chuẩn hóa
    y = datas[:, 1]  # Tọa độ Y chuẩn hóa
    plt.figure(2)
    label = ["Section center", "Gaze Point"]
    # Vẽ các điểm nhìn dưới dạng các chấm tròn màu xanh lam
    plt.scatter(x, y, c='b', marker='.', s=20)
    # Vẽ đường nối các tâm phân đoạn theo trình tự thời gian (đường đứt nét màu đỏ)
    plt.plot(center[:, 0], center[:, 1], 'ro--')
    plt.legend(label, loc=0, ncol=1)
    plt.xlabel('X-axis(Normalized coordinate)')
    plt.ylabel('Y-axis(Normalized coordinate)')
    plt.title('Number of eye movement sequence segments N = {}'.format(len(center)))
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


def Remove_redundant_data(labels, centers):
    """
    Sắp xếp lại các tâm cụm theo trình tự thời gian xuất hiện đầu tiên của từng nhãn cụm trong chuỗi dữ liệu.
    
    Tham số:
        labels (np.ndarray): Mảng nhãn phân cụm của các điểm dữ liệu.
        centers (np.ndarray): Tọa độ các tâm cụm K-Means.
        
    Trả về:
        center_order (np.ndarray): Tọa độ các tâm cụm theo đúng trình tự chuỗi thời gian.
    """
    # Lấy danh sách các nhãn duy nhất theo thứ tự xuất hiện từ đầu đến cuối
    temp_list = func(labels)
    n = len(temp_list)
    center_order = np.empty(shape=[n, 2])
    # Sắp xếp lại tọa độ tâm cụm tương ứng với thứ tự nhãn đã tìm được
    for i in range(0, n):
        j = temp_list[i]
        center_order[i, :] = centers[j, :]
    return center_order


def func(labels):
    """
    Hàm phụ trợ: Trích xuất các phần tử duy nhất trong danh sách nhãn nhưng vẫn bảo toàn thứ tự xuất hiện ban đầu.
    
    Tham số:
        labels (iterable): Mảng hoặc danh sách nhãn phân cụm.
        
    Trả về:
        temp_list (list): Danh sách các nhãn không trùng lặp theo thứ tự xuất hiện.
    """
    temp_list = []
    for i in labels:
        if i not in temp_list:
            temp_list.append(i)
    return temp_list


if __name__ == '__main__':
    # Đọc dữ liệu chuyển động mắt từ file văn bản
    datas = sample_create.txt2matrix('dataset/testdataset/tester01_1.txt')
    rcParams.update(config)
    
    # Chỉ lấy tọa độ 2D [x, y] để phân cụm vị trí không gian
    datas = datas[:, 0:2]
    
    # Khảo sát số lượng cụm K từ 1 đến 9 để tìm điểm gãy (Elbow)
    Ks = range(1, 10)
    center, labels = km_deviation(Ks, datas)
    
    # Chọn cấu hình với k = 4 (index 4 trong dict tương ứng k=4 cụm)
    # Sắp xếp các tâm cụm theo trình tự xuất hiện thời gian
    center_order = Remove_redundant_data(labels[4], center[4])
    
    # Vẽ biểu đồ 2D trực quan hóa các điểm nhìn và đường nối các tâm phân đoạn
    cluster_center_plot(datas, center_order)
    print("------------------end---------------------")
