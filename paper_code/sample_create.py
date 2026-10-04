# -*- coding: utf-8 -*-
"""
Module: sample_create.py
Chức năng:
- Đọc, xử lý và chuyển đổi dữ liệu chuỗi chuyển động mắt thô từ file text sang ma trận NumPy.
- Cung cấp các hàm phân đoạn chuỗi điểm nhìn ở Vòng 1 (Round 1 Classification - Path Segmentation).
- Cung cấp các hàm phân loại điểm nhìn chi tiết ở Vòng 2 (Round 2 Classification - Gaze Point Classification).
- Xuất kết quả phân loại ra các file định dạng CSV và text (section0.txt, section1.txt,...).
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM


def txt2matrix(filename):
    """
    Đọc file văn bản chứa dữ liệu điểm nhìn và chuyển thành ma trận NumPy 2D.
    Hỗ trợ linh hoạt cả file 3 cột [x, y, v] và file 4 cột [x, y, v, label].
    
    Tham số:
        filename (str): Đường dẫn đến file dữ liệu (ví dụ: 'collect_clasi_train.txt' hoặc 'tester11_1.txt').
        
    Trả về:
        datamat (np.ndarray): Ma trận kích thước [N, cols] chứa dữ liệu điểm nhìn.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if not lines:
        return np.empty((0, 3))
        
    # Xác định số cột từ dòng đầu tiên
    first_row = [float(val) for val in lines[0].split() if val != '']
    cols = len(first_row)
    
    datamat = np.zeros((len(lines), cols))
    for i, line in enumerate(lines):
        vals = [float(val) for val in line.split() if val != '']
        datamat[i, :] = vals[:cols]

    return datamat


# =============================================================================
# Các hàm tính toán lý thuyết HMM mẫu (Tham khảo)
# =============================================================================
# # Lấy xác suất quan sát mẫu o thuộc trạng thái s: P(o|s)
# def prob_O2S(model, o):
#     M_O2S = model["M_O2S"]
#     return M_O2S[:,int(o)]
#
#
# # Sinh một mẫu ngẫu nhiên từ phân phối xác suất rời rạc
# def get_one_sample_from_Prob_distribution(Prob_dis):
#     N_segment = np.shape(Prob_dis)[0]
#     prob_segment = np.zeros(N_segment)
#
#     for i in range(N_segment):
#         prob_segment[i] = prob_segment[i-1] + Prob_dis[i]
#
#     S = 0
#     data = np.random.rand()
#     for i in range(N_segment):
#         if data <= prob_segment[i]:
#             S = i
#             break
#     return S
#
#
# # Sinh chuỗi quan sát và chuỗi trạng thái từ mô hình HMM
# def get_sample_from_HMM(model, N):
#     M_O2S = model["M_O2S"]
#     datas = np.zeros(N)
#     stats = np.zeros(N)
#
#     # Khởi tạo: sinh trạng thái đầu tiên theo phân phối ban đầu pi
#     init_S = get_one_sample_from_Prob_distribution(model["pi"])
#     stats[0] = init_S
#     datas[0] = get_one_sample_from_Prob_distribution(M_O2S[int(stats[0])])
#
#     # Sinh các mẫu tiếp theo dựa trên ma trận chuyển trạng thái A và ma trận phát xạ B
#     for i in range(1, N):
#         # Dựa vào trạng thái t-1 sinh trạng thái t: A[state_{t-1}]
#         stats[i] = get_one_sample_from_Prob_distribution(model["A"][int(stats[i-1])])
#         # Dựa vào trạng thái t sinh quan sát t: M_O2S[state_t]
#         datas[i] = get_one_sample_from_Prob_distribution(M_O2S[int(stats[i])])
#     return datas, stats
#
#
# # Thuật toán Forward: Tính biến alpha_t(i) = P(o_1, ..., o_t, s_t = i | lambda)
# def calc_alpha(model, observations):
#     o = observations
#     N_samples = np.shape(o)[0]
#     N_stats = np.shape(model["pi"])[0]
#
#     alpha = np.zeros([N_samples, N_stats])
#     # Khởi tạo alpha tại t=0
#     alpha[0] = model["pi"] * model["B"](model, o[0])
#
#     # Quy nạp tính alpha tại các bước thời gian tiếp theo
#     for t in range(1, N_samples):
#         s_current = np.dot(alpha[t-1], model["A"])
#         alpha[t] = s_current * model["B"](model, o[t])
#
#     return alpha
#
#
# def forward(model, observation):
#     o = observation
#     # Tính log xác suất của chuỗi quan sát: log P(O|lambda)
#     alpha = calc_alpha(model, o)
#     prob_seq_f = np.sum(alpha[-1])
#     return np.log(prob_seq_f)
#
#
# # Thuật toán Backward: Tính biến beta_t(i) = P(o_{t+1}, ..., o_T | s_t = i, lambda)
# def calc_beta(model, observation):
#     o = observation
#     N_sample = np.shape(o)[0]
#     N_stats = np.shape(model["pi"])[0]
#
#     beta = np.zeros([N_sample, N_stats])
#     # Khởi tạo beta tại bước cuối cùng T-1
#     beta[-1] = 1
#
#     # Quy nạp lùi từ T-2 về 0
#     for t in range(N_sample-2, -1, -1):
#         s_next = beta[t+1] * model["B"](model, o[t+1])
#         beta[t] = np.dot(s_next, model["A"].T)
#
#     return beta
#
#
# def backward(model, observation):
#     o = observation
#     # Tính log xác suất lùi của chuỗi quan sát
#     beta = calc_beta(model, o)
#     s_next = beta[0] * model["B"](model, o[0])
#     prob_seq_b = np.dot(s_next, model["pi"])
#     return np.log(prob_seq_b)


def classification_xyv(states, datas_test):
    """
    Ghi kết quả phân loại vòng 1 (dựa trên x, y, v) ra file CSV.
    
    Tham số:
        states (tuple): Kết quả giải mã Viterbi từ mô hình GMMHMM (score, state_sequence).
        datas_test (np.ndarray): Dữ liệu kiểm tra gồm [x, y, Velocity].
    """
    states = np.array(list(states[1]))
    states = states[:, np.newaxis]
    datas_states = np.hstack((datas_test, states))

    df = pd.DataFrame(datas_states, columns=['x', 'y', 'Velocity', 'Classification results'])
    df.to_csv('classification_results_of_subpaths.csv', encoding="utf_8_sig")


def classification_v(states, datas):
    """
    Ghi kết quả phân loại (chỉ dựa trên vận tốc v) ra file CSV.
    
    Tham số:
        states (tuple): Kết quả giải mã Viterbi từ mô hình GMMHMM.
        datas (np.ndarray): Dữ liệu vận tốc [Velocity].
    """
    states = np.array(list(states[1]))
    states = states[:, np.newaxis]
    datas_states = np.hstack((datas, states))

    df = pd.DataFrame(datas_states, columns=['Velocity', 'Classification results'])
    df.to_csv('classification_results.csv', encoding="utf_8_sig")


def classsification_result(datas, result):
    """
    Tổng hợp kết quả phân loại 3 lớp (Ternary Classification) cho toàn bộ chuỗi dữ liệu
    và lưu ra file CSV 'ternary_classification_results.csv'.
    
    Tham số:
        datas (np.ndarray): Ma trận dữ liệu gốc [x, y, Velocity].
        result (dict): Từ điển chứa danh sách nhãn phân loại của từng phân đoạn:
                       0: Fixation (Cố định), 1: Smooth Pursuit (Bám đuổi), 2: Saccade (Nhảy mắt).
    """
    a = []
    for key in result:
        a = a + result[key]
    a = np.array(a)
    a = a[:, np.newaxis]
    datas_states = np.hstack((datas, a))
    df = pd.DataFrame(datas_states, columns=['x', 'y', 'Velocity', 'Classification results'])
    df.to_csv('ternary_classification_results.csv', encoding="utf_8_sig")


def classification_round_1(datas, n_compoments, states):
    """
    Phân loại Vòng 1 (Round 1 Classification):
    Phân chia toàn bộ chuỗi quỹ đạo chuyển động mắt thành `n_components` đoạn con (sections/sub-sequences).
    Điểm chia ranh giới giữa các đoạn được xác định bằng vị trí xuất hiện đầu tiên của từng trạng thái ẩn.
    
    Tham số:
        datas (np.ndarray): Ma trận dữ liệu đầu vào [x, y, v].
        n_compoments (int): Số lượng trạng thái ẩn / số lượng đoạn con cần chia.
        states (tuple): Đầu ra từ thuật toán giải mã Viterbi (score, chuỗi trạng thái).
        
    Trả về:
        states_info (dict): Từ điển chứa dữ liệu của từng đoạn con {0: section_0, 1: section_1, ...}.
                            Đồng thời lưu từng đoạn ra các file text 'section0.txt', 'section1.txt',...
    """
    classification_round_1 = list(states[1])
    first_index_of_states = np.zeros(n_compoments + 1)
    
    # Tìm vị trí xuất hiện đầu tiên của mỗi trạng thái trong chuỗi giải mã
    for i in range(0, n_compoments):
        first_index_of_states[i] = classification_round_1.index(i)

    length = len(classification_round_1)
    first_index_of_states[n_compoments] = length
    # Sắp xếp các chỉ số mốc thời gian theo thứ tự tăng dần
    first_index_of_states.sort()
    
    states_info = {}
    for i in range(0, n_compoments):
        filename = "section" + str(i) + ".txt"
        start = (int)(first_index_of_states[i])
        end = (int)(first_index_of_states[i + 1])
        section = datas[start:end, :]
        states_info[i] = section
        # Lưu phân đoạn ra file văn bản
        df = pd.DataFrame(section)
        df.to_csv(filename)

    return states_info


def section_classification(datas):
    """
    Huấn luyện một mô hình GMMHMM cục bộ trên dữ liệu của một phân đoạn (section)
    để phân loại chi tiết các điểm nhìn trong phân đoạn đó.
    
    Tham số:
        datas (np.ndarray): Dữ liệu đặc trưng của phân đoạn (có thể là cột vận tốc v hoặc [x, y, v]).
        
    Trả về:
        x[1] (np.ndarray): Mảng nhãn trạng thái ẩn giải mã được cho các điểm nhìn trong phân đoạn.
    """
    # Khởi tạo GMMHMM với 3 trạng thái ẩn (ứng với 3 loại chuyển động mắt) và 2 thành phần Gauss mỗi trạng thái
    m_GMMHMM1 = GMMHMM(n_components=3, n_mix=2, covariance_type='full', n_iter=50, tol=0.00001, verbose=True)
    m_GMMHMM1.fit(datas)
    # Giải mã Viterbi để tìm chuỗi trạng thái ẩn tối ưu nhất
    # x[0]: log-likelihood score, x[1]: chuỗi nhãn trạng thái giải mã
    x = m_GMMHMM1.decode(datas)
    return x[1]


def classification_round_2(states_info, mode):
    """
    Phân loại Vòng 2 (Round 2 Classification):
    Tiến hành phân loại vi mô các điểm nhìn trong từng phân đoạn đã được chia ở Vòng 1.
    
    Tham số:
        states_info (dict): Từ điển chứa dữ liệu các phân đoạn từ Vòng 1.
        mode (int): Chế độ phân loại:
                    - mode = 0: Phân loại chỉ dựa trên đặc trưng vận tốc (Velocity - v).
                    - mode = 1: Phân loại dựa trên toàn bộ 3 đặc trưng không gian và vận tốc (x, y, v).
                    
    Trả về:
        gaze_state_info (dict): Từ điển chứa chuỗi nhãn trạng thái phân loại cho từng phân đoạn.
    """
    gaze_state_info = {}
    if mode == 0:
        # Chế độ 0: Trích xuất riêng cột vận tốc v (cột index 2) để phân loại
        for key in states_info:
            temp = states_info[key]
            temp = temp[:, 2]
            temp = temp.reshape(-1, 1)
            gaze_state_info[key] = section_classification(temp)
    elif mode == 1:
        # Chế độ 1: Sử dụng toàn bộ ma trận [x, y, v] của phân đoạn để phân loại
        for key in states_info:
            gaze_state_info[key] = section_classification(states_info[key])

    return gaze_state_info