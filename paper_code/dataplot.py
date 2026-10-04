# -*- coding: utf-8 -*-
"""
Module: dataplot.py
Chức năng:
- Trực quan hóa dữ liệu chuyển động mắt trong không gian 3 chiều (X: tọa độ x, Y: tọa độ y, Z: vận tốc v).
- Vẽ biểu đồ dữ liệu thô ban đầu (Raw Data Plot).
- Vẽ biểu đồ các phân đoạn đường nhìn sau Vòng phân loại 1 (Path Classification Plot).
- Vẽ biểu đồ phân loại điểm nhìn sau Vòng phân loại 2 (Gaze Point Classification Plot).
- Tự động ánh xạ trạng thái sang 3 dạng chuyển động mắt chính (Fixation, Smooth Pursuit, Saccade)
  dựa trên mức độ vận tốc và trực quan hóa kết quả phân loại 3 lớp cuối cùng.
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def path_classification_plot(states_info):
    """
    Vẽ biểu đồ 3D biểu diễn các phân đoạn đường nhìn sau vòng phân loại thứ nhất.
    Mỗi phân đoạn (subpath/section) được hiển thị bằng một màu sắc riêng biệt trên không gian (X, Y, V).
    
    Tham số:
        states_info (dict): Từ điển chứa dữ liệu [x, y, v] của từng phân đoạn đường nhìn.
    """
    x = {}
    y = {}
    z = {}
    fig = plt.figure(2)
    ax = Axes3D(fig, auto_add_to_figure=False)
    
    # Trích xuất tọa độ X, Y và vận tốc V cho từng phân đoạn
    for key in states_info:
        position3d = np.array(list(states_info[key]))
        x[key] = position3d[:, 0]
        y[key] = position3d[:, 1]
        z[key] = position3d[:, 2]
        
        # Gán màu sắc đặc trưng cho từng phân đoạn (tối đa 6 phân đoạn)
        if key == 0:
            ax.scatter(np.array(list(x[0])), np.array(list(y[0])), np.array(list(z[0])), zdir="z", c="#00DDAA", marker="o", s=40)
        elif key == 1:
            ax.scatter(np.array(list(x[1])), np.array(list(y[1])), np.array(list(z[1])), zdir="z", c="#FF5511", marker="o", s=40)
        elif key == 2:
            ax.scatter(np.array(list(x[2])), np.array(list(y[2])), np.array(list(z[2])), zdir="z", c="#0000CD", marker="o", s=40)
        elif key == 3:
            ax.scatter(np.array(list(x[3])), np.array(list(y[3])), np.array(list(z[3])), zdir="z", c="#FFA500", marker="o", s=40)
        elif key == 4:
            ax.scatter(np.array(list(x[4])), np.array(list(y[4])), np.array(list(z[4])), zdir="z", c="#C0C0C0", marker="o", s=40)
        elif key == 5:
            ax.scatter(np.array(list(x[5])), np.array(list(y[5])), np.array(list(z[5])), zdir="z", c="#FFD700", marker="o", s=40)

    # Thiết lập nhãn các trục tọa độ và tiêu đề biểu đồ
    ax.set(xlabel="X(0~1)", ylabel="Y(0~1)", zlabel="V")
    ax.set_title(label="after first classification")
    fig.add_axes(ax)
    plt.show()


def gaze_trace_xyv_plot(datas):
    """
    Vẽ biểu đồ phân tán 3D (Scatter Plot) của chuỗi dữ liệu điểm nhìn thô ban đầu (Raw Gaze Data).
    
    Tham số:
        datas (np.ndarray): Ma trận dữ liệu gốc gồm 3 cột [tọa độ X chuẩn hóa, tọa độ Y chuẩn hóa, Vận tốc V].
    """
    x = datas[:, 0]  # Tọa độ X chuẩn hóa (0 ~ 1)
    y = datas[:, 1]  # Tọa độ Y chuẩn hóa (0 ~ 1)
    z = datas[:, 2]  # Vận tốc chuyển động mắt (Velocity)
    fig = plt.figure(1)
    ax = Axes3D(fig, auto_add_to_figure=False)
    
    # Vẽ các điểm quan sát dạng chấm tròn màu đen
    ax.scatter(x, y, z, zdir="z", c="#000000", marker="o", s=40)
    fig.add_axes(ax)
    ax.set(xlabel="X(0~1)", ylabel="Y(0~1)", zlabel="V")
    ax.set_title(label="raw datas", fontsize=20)
    plt.show()


def gaze_point_classification_plot(states_info, gaze_states_info, num):
    """
    Vẽ biểu đồ 3D biểu diễn kết quả phân loại điểm nhìn trong từng phân đoạn sau Vòng 2.
    - Màu sắc: Đại diện cho phân đoạn (Section).
    - Dạng điểm (Marker): Đại diện cho nhãn trạng thái ('x' cho trạng thái 0, '^' cho trạng thái 1, '.' cho trạng thái 2).
    
    Tham số:
        states_info (dict): Dữ liệu tọa độ [x, y, v] của từng phân đoạn.
        gaze_states_info (dict): Nhãn trạng thái phân loại của các điểm trong từng phân đoạn.
        num (int): Tham số mở rộng (số lượng nhóm/loại).
    """
    fig = plt.figure(3)
    ax = Axes3D(fig, auto_add_to_figure=False)
    x = {}
    y = {}
    z = {}
    for key in states_info:
        position3d = np.array(list(states_info[key]))
        x[key] = position3d[:, 0]
        y[key] = position3d[:, 1]
        z[key] = position3d[:, 2]
        
    for key in x:
        x1 = np.array(list(x[key]))
        y1 = np.array(list(y[key]))
        z1 = np.array(list(z[key]))
        index_0 = []
        index_1 = []
        index_2 = []
        
        # Lọc các chỉ số điểm theo từng trạng thái giải mã (0, 1, 2)
        for index, i in enumerate(np.array(list(gaze_states_info[key]))):
            if i == 0:
                index_0.append(index)
            elif i == 1:
                index_1.append(index)
            elif i == 2:
                index_2.append(index)
            x_0 = x1[index_0]
            y_0 = y1[index_0]
            z_0 = z1[index_0]
            x_1 = x1[index_1]
            y_1 = y1[index_1]
            z_1 = z1[index_1]
            x_2 = x1[index_2]
            y_2 = y1[index_2]
            z_2 = z1[index_2]
            
        # Vẽ các điểm tương ứng với từng trạng thái theo màu sắc của phân đoạn
        if key == 0:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#00DDAA", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#00DDAA", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#00DDAA", marker=".", s=40)
        elif key == 1:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#FF5511", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#FF5511", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#FF5511", marker=".", s=40)
        elif key == 2:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#0000CD", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#0000CD", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#0000CD", marker=".", s=40)
        elif key == 3:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#FFA500", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#FFA500", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#FFA500", marker=".", s=40)
        elif key == 4:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#C0C0C0", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#C0C0C0", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#C0C0C0", marker=".", s=40)
        elif key == 5:
            ax.scatter(x_0, y_0, z_0, zdir="z", c="#FFD700", marker="x", s=40)
            ax.scatter(x_1, y_1, z_1, zdir="z", c="#FFD700", marker="^", s=40)
            ax.scatter(x_2, y_2, z_2, zdir="z", c="#FFD700", marker=".", s=40)
            
    fig.add_axes(ax)
    ax.set(xlabel="X(0~1)", ylabel="Y(0~1)", zlabel="V")
    plt.show()


def gazepoint_classification_final_results(states_info, gaze_states_info):
    """
    Xác định chính xác loại chuyển động mắt dựa trên so sánh độ lớn vận tốc (V):
    - Fixation (index_fx = 0): Chuyển động cố định (Vận tốc nhỏ nhất).
    - Smooth Pursuit (index_sp = 1): Chuyển động bám đuổi (Vận tốc trung bình).
    - Saccade (index_sc = 2): Chuyển động nhảy mắt (Vận tốc lớn nhất).
    
    Sau đó vẽ biểu đồ 3D chuẩn hóa màu sắc/marker thống nhất cho toàn bộ hệ thống:
    - Fixation: Màu xanh lục ngọc (#00DDAA), marker 'x'
    - Smooth Pursuit: Màu cam đỏ (#FF5511), marker '^'
    - Saccade: Màu xanh dương (#0000CD), marker '.'
    
    Tham số:
        states_info (dict): Tọa độ [x, y, v] của các phân đoạn.
        gaze_states_info (dict): Nhãn phân loại thô từ Vòng 2.
        
    Trả về:
        result (dict): Từ điển chứa chuỗi nhãn phân loại 3 lớp chuẩn hóa cho từng phân đoạn.
    """
    fig = plt.figure(4)
    ax = Axes3D(fig, auto_add_to_figure=False)
    x = {}
    y = {}
    z = {}
    result = {}
    
    for key in states_info:
        position3d = np.array(list(states_info[key]))
        x[key] = position3d[:, 0]
        y[key] = position3d[:, 1]
        z[key] = position3d[:, 2]
        
    for key in x:
        x1 = np.array(list(x[key]))
        y1 = np.array(list(y[key]))
        z1 = np.array(list(z[key]))
        # Thu thập các chỉ số thuộc từng trạng thái ban đầu
        index_0 = []
        index_1 = []
        index_2 = []
        index_ternary = np.zeros(len(x1))
        
        for index, i in enumerate(np.array(list(gaze_states_info[key]))):
            if i == 0:
                index_0.append(index)
            elif i == 1:
                index_1.append(index)
            elif i == 2:
                index_2.append(index)
                
        index_fx = []  # Fixation: Vận tốc thấp nhất
        index_sp = []  # Smooth Pursuit: Vận tốc trung bình
        index_sc = []  # Saccade: Vận tốc cao nhất
        
        # GMM-HMM có thể không sinh đủ cả 3 trạng thái, nhất là với section
        # ngắn. Sắp xếp chỉ các trạng thái thực sự xuất hiện theo vận tốc đại
        # diện, từ thấp đến cao.
        state_indices = {
            0: index_0,
            1: index_1,
            2: index_2,
        }
        observed_states = [
            state for state, indices in state_indices.items() if indices
        ]
        if not observed_states:
            raise ValueError(
                f"Section {key} không có nhãn vi mô hợp lệ từ GMM-HMM."
            )

        observed_states.sort(
            key=lambda state: float(np.median(z1[state_indices[state]]))
        )
        physical_indices = [index_fx, index_sp, index_sc]
        for physical_index, state in zip(physical_indices, observed_states):
            physical_index.extend(state_indices[state])
                    
        x_0 = x1[index_fx]
        y_0 = y1[index_fx]
        z_0 = z1[index_fx]
        x_1 = x1[index_sp]
        y_1 = y1[index_sp]
        z_1 = z1[index_sp]
        x_2 = x1[index_sc]
        y_2 = y1[index_sc]
        z_2 = z1[index_sc]
        
        # Gán nhãn phân loại 3 lớp chuẩn hóa:
        # 0: Fixation (Cố định)
        # 1: Smooth Pursuit (Bám đuổi)
        # 2: Saccade (Nhảy mắt)
        index_ternary[index_fx] = 0
        index_ternary[index_sp] = 1
        index_ternary[index_sc] = 2
        result[key] = list(index_ternary)
        
        # Vẽ biểu đồ 3D với màu sắc và marker phân biệt cho từng loại chuyển động mắt
        ax.scatter(x_0, y_0, z_0, zdir="z", c="#00DDAA", marker="x", s=40, label="Fixation" if key==0 else "")
        ax.scatter(x_1, y_1, z_1, zdir="z", c="#FF5511", marker="^", s=40, label="Smooth Pursuit" if key==0 else "")
        ax.scatter(x_2, y_2, z_2, zdir="z", c="#0000CD", marker=".", s=40, label="Saccade" if key==0 else "")
        
    fig.add_axes(ax)
    ax.set(xlabel="X(0~1)", ylabel="Y(0~1)", zlabel="V")
    plt.show()
    return result
