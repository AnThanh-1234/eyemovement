# -*- coding: utf-8 -*-
"""
Chương trình chính (Main Pipeline) thực hiện phân loại chuyển động mắt (Eye Movement Classification)
Sử dụng mô hình GMM-HMM kết hợp phương pháp phân loại 2 vòng (Two-round Classification):
- Vòng 1: Phân đoạn toàn bộ chuỗi chuyển động mắt thành các đoạn con (sub-sequences/paths).
- Vòng 2: Phân loại chi tiết các điểm nhìn trong từng đoạn con thành 3 loại chuyển động mắt:
  + Fixation (Nhìn cố định - vận tốc thấp)
  + Smooth Pursuit (Nhìn bám đuổi - vận tốc trung bình)
  + Saccade (Chuyển động mắt nhanh/nhảy mắt - vận tốc cao)
"""

import dataplot
from hmmlearn.hmm import MultinomialHMM, GMMHMM
import sample_create


if __name__ == '__main__':
    # =========================================================================
    # Task 0: Tải và tiền xử lý dữ liệu thô (Load & Preprocess Data)
    # =========================================================================
    # Đọc file dữ liệu huấn luyện, trả về ma trận numpy chứa các cột [x, y, v]
    # x, y: tọa độ không gian chuẩn hóa của điểm nhìn (0 ~ 1)
    # v: vận tốc chuyển động mắt (Velocity)
    datas = sample_create.txt2matrix('collect_clasi_train.txt')
    # Lấy 3 cột đầu tiên: [tọa độ x, tọa độ y, vận tốc v]
    datas = datas[:, 0:3]
    # datas = datas.reshape(-1, 1)
    # print(datas)

    # =========================================================================
    # Task 1: Khởi tạo và huấn luyện mô hình GMM-HMM cho vòng 1
    # =========================================================================
    # n_components: Số trạng thái ẩn (ở đây đại diện cho số đoạn kích thích/vùng nhìn mục tiêu)
    n_components = 3
    # Khởi tạo mô hình Gaussian Mixture Model - Hidden Markov Model (GMM-HMM):
    # - n_components: Số lượng trạng thái ẩn (Hidden States) = 3
    # - n_mix: Số lượng thành phần Gauss trong mỗi trạng thái (Gaussian Mixtures) = 2
    # - covariance_type='full': Ma trận hiệp phương sai đầy đủ (tính tương quan giữa các chiều)
    # - n_iter=50: Số vòng lặp tối đa của thuật toán EM (Baum-Welch)
    # - tol=0.00001: Ngưỡng hội tụ của hàm log-likelihood
    # - verbose=True: In thông tin quá trình huấn luyện
    m_GMMHMM = GMMHMM(n_components=n_components, n_mix=2, covariance_type='full', n_iter=50, tol=0.00001, verbose=True)
    
    # Huấn luyện mô hình GMM-HMM trên tập dữ liệu đặc trưng (x, y, v)
    m_GMMHMM.fit(datas)

    # =========================================================================
    # Task 2: Phân đoạn chuỗi chuyển động mắt (Step 1) & Trực quan hóa
    # =========================================================================
    # Bước 1: Vẽ đồ thị không gian 3D (x, y, v) của dữ liệu thô ban đầu
    dataplot.gaze_trace_xyv_plot(datas)
    
    # Sử dụng thuật toán Viterbi giải mã chuỗi trạng thái ẩn tối ưu nhất cho dữ liệu
    # states[0]: Log-likelihood, states[1]: Chuỗi nhãn trạng thái ẩn
    states = m_GMMHMM.decode(datas)
    # print(states)
    
    # Phân loại vòng 1: Chia chuỗi dữ liệu thành các đoạn con (sections) dựa trên trạng thái
    # Lưu từng đoạn con thành các file section0.txt, section1.txt,...
    states_info = sample_create.classification_round_1(datas, n_components, states)
    # print(states_info)
    
    # Vẽ biểu đồ 3D biểu diễn các đoạn con sau khi phân loại vòng 1 với các màu khác nhau
    dataplot.path_classification_plot(states_info)

    # =========================================================================
    # Task 3: Phân loại chi tiết điểm nhìn (Step 2 - Second Round Classification)
    # =========================================================================
    # Phân loại vòng 2: Phân loại các điểm nhìn trong từng đoạn con
    # mode = 0: Phân loại dựa trên đặc trưng vận tốc (Velocity - v)
    # mode = 1: Phân loại dựa trên cả 3 đặc trưng (x, y, v)
    gaze_states_info = sample_create.classification_round_2(states_info, 0)
    print("Kết quả phân loại vòng 2 theo từng phân đoạn:", gaze_states_info)
    
    # Vẽ biểu đồ 3D hiển thị phân loại điểm nhìn với các dạng điểm đánh dấu khác nhau (marker: x, ^, .)
    dataplot.gaze_point_classification_plot(states_info, gaze_states_info, 2)
    
    # Ánh xạ các nhãn trạng thái sang 3 loại chuyển động mắt cụ thể (0: Fixation, 1: Smooth Pursuit, 2: Saccade)
    # Dựa trên mức độ vận tốc v trong từng cụm
    result = dataplot.gazepoint_classification_final_results(states_info, gaze_states_info)
    
    # Lưu kết quả phân loại 3 lớp cuối cùng ra file CSV 'ternary_classification_results.csv'
    sample_create.classsification_result(datas, result)
    
    print("-----------------stop----------------------")
    # sample_create.classification_xyv(states, datas)
