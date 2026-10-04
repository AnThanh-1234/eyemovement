# -*- coding: utf-8 -*-
"""
Module: GMM_hmm.py
Chức năng:
- Cài đặt thủ công đầy đủ thuật toán GMM-HMM (Gaussian Mixture Model - Hidden Markov Model) từ đầu.
- Cung cấp các thuật toán nền tảng của Mô hình Markov Ẩn (HMM):
  + Sinh dữ liệu mẫu (Sample Generation từ GMM và GMM-HMM).
  + Thuật toán Forward & Backward (Tính xác suất chuỗi quan sát P(O|lambda)).
  + Thuật toán Viterbi (Giải mã chuỗi trạng thái ẩn tối ưu nhất).
  + Thuật toán Baum-Welch / EM (Ước lượng và cập nhật tham số: pi, A, và các thành phần Gauss GMM).
  + Tính hàm mật độ xác suất PDF Gauss đa biến (Multivariate Gaussian PDF).
"""

import numpy as np


def gen_one_sample_from_Prob_list(Prob_list):
    """
    Sinh một mẫu ngẫu nhiên (chỉ số trạng thái/thành phần) từ một phân phối xác suất rời rạc.
    
    Tham số:
        Prob_list (list / np.ndarray): Mảng phân phối xác suất (tổng bằng 1.0).
        
    Trả về:
        S (int): Chỉ số phần tử được chọn ngẫu nhiên theo phân phối xác suất.
    """
    N_segment = np.shape(Prob_list)[0]
    
    # Chia khoảng [0, 1] thành N_segment đoạn tích lũy (Cumulative Distribution)
    # Ví dụ: Prob_list = [0.3, 0.3, 0.4] -> prob_segment = [0.3, 0.6, 1.0]
    prob_segment = np.zeros(N_segment)
    for i in range(N_segment):
        prob_segment[i] = prob_segment[i-1] + Prob_list[i]
    
    S = 0
    # Sinh số ngẫu nhiên phân bố đều trong khoảng [0, 1)
    data = np.random.rand()
    # Kiểm tra số ngẫu nhiên rơi vào khoảng tích lũy nào
    for i in range(N_segment):
        if data <= prob_segment[i]:
            S = i
            break
    return S
    
    
def gen_one_sample_from_GMM(gmm):
    """
    Sinh một vector quan sát ngẫu nhiên từ một mô hình hỗn hợp Gauss (Gaussian Mixture Model - GMM).
    
    Tham số:
        gmm (dict): Chứa thông tin GMM gồm:
                    - 'ws': Trọng số các thành phần Gauss (Mixture weights)
                    - 'mus': Vector kỳ vọng trung bình (Means)
                    - 'sigmas': Ma trận hiệp phương sai (Covariance matrices)
                    
    Trả về:
        Vector quan sát D chiều được lấy mẫu từ phân phối chuẩn đa biến đã chọn.
    """
    # Bước 1: Dựa vào trọng số ws chọn ra 1 thành phần Gauss k
    k = gen_one_sample_from_Prob_list(gmm["ws"])
    
    # Bước 2: Lấy vector kỳ vọng và ma trận hiệp phương sai của thành phần k
    mu = gmm["mus"][k]
    sigma = gmm["sigmas"][k]
    
    # Bước 3: Sinh mẫu từ phân phối chuẩn đa biến N(mu, sigma)
    return np.random.multivariate_normal(mu, sigma)
    
    
def gen_samples_from_GMM_HMM(model, N):
    """
    Sinh chuỗi quan sát gồm N mẫu và chuỗi trạng thái ẩn tương ứng từ mô hình GMM-HMM.
    
    Tham số:
        model (dict): Mô hình GMM-HMM đầy đủ ('pi', 'A', 'S', 'B').
        N (int): Độ dài chuỗi quan sát cần sinh.
        
    Trả về:
        datas (np.ndarray): Ma trận chuỗi quan sát [N, D].
        stats (np.ndarray): Mảng chuỗi trạng thái ẩn tương ứng [N].
    """
    # K: số thành phần Gauss, D: số chiều của vector đặc trưng
    K, D = np.shape(model["S"][0]["mus"])
    datas = np.zeros([N, D])
    stats = np.zeros(N)
    
    # Khởi tạo: Lấy mẫu trạng thái ban đầu theo phân phối pi
    init_S = gen_one_sample_from_Prob_list(model["pi"])
    stats[0] = init_S
    
    # Dựa vào trạng thái ban đầu, sinh điểm dữ liệu quan sát đầu tiên từ GMM của trạng thái đó
    datas[0] = gen_one_sample_from_GMM(model["S"][int(stats[0])]) 
    
    # Sinh tuần tự các mẫu tiếp theo trong chuỗi thời gian
    for i in range(1, N):
        # Dựa vào trạng thái tại thời điểm i-1, chuyển sang trạng thái tại thời điểm i theo ma trận A
        stats[i] = gen_one_sample_from_Prob_list(model["A"][int(stats[i-1])])
        # Dựa vào trạng thái mới, sinh điểm quan sát từ GMM tương ứng
        datas[i] = gen_one_sample_from_GMM(model["S"][int(stats[i])])
    return datas, stats


# =============================================================================
# Thuật toán Forward - Backward (Thuật toán Tiến - Lùi)
# =============================================================================

def calc_alpha(model, observations):
    """
    Thuật toán Forward: Tính toán biến tiến alpha_t(i)
    alpha_t(i) = P(o_1, o_2, ..., o_t, q_t = S_i | model)
    Xác suất quan sát được chuỗi từ 1 đến t và trạng thái tại thời điểm t là S_i.
    
    Tham số:
        model (dict): Mô hình GMM-HMM.
        observations (np.ndarray): Chuỗi quan sát [N_samples, D].
        
    Trả về:
        alpha (np.ndarray): Ma trận alpha kích thước [N_samples, N_stats].
    """
    o = observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]
    
    # Khởi tạo ma trận alpha
    alpha = np.zeros([N_samples, N_stats])
    
    # Bước khởi tạo (t = 0): alpha_0(i) = pi_i * b_i(o_0)
    alpha[0] = model["pi"] * model["B"](model, o[0])
   
    # Bước quy nạp (t = 1 -> T-1): alpha_t(j) = [sum_i (alpha_{t-1}(i) * a_{ij})] * b_j(o_t)
    for t in range(1, N_samples):
        s_current = np.dot(alpha[t-1], model["A"])
        alpha[t] = s_current * model["B"](model, o[t])
   
    return alpha
    

def calc_beta(model, observations):
    """
    Thuật toán Backward: Tính toán biến lùi beta_t(i)
    beta_t(i) = P(o_{t+1}, o_{t+2}, ..., o_T | q_t = S_i, model)
    Xác suất quan sát được chuỗi từ t+1 đến T khi biết trạng thái tại thời điểm t là S_i.
    
    Tham số:
        model (dict): Mô hình GMM-HMM.
        observations (np.ndarray): Chuỗi quan sát [N_samples, D].
        
    Trả về:
        beta (np.ndarray): Ma trận beta kích thước [N_samples, N_stats].
    """
    o = observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]
    
    beta = np.zeros([N_samples, N_stats])
    
    # Bước khởi tạo (t = T-1): beta_{T-1}(i) = 1 với mọi i
    beta[-1] = 1
    
    # Bước quy nạp lùi (t = T-2 -> 0): beta_t(i) = sum_j [a_{ij} * b_j(o_{t+1}) * beta_{t+1}(j)]
    for t in range(N_samples-2, -1, -1):
        s_next = beta[t+1] * model["B"](model, o[t+1])
        beta[t] = np.dot(s_next, model["A"].T)
    return beta    
        

def forward(model, observations):
    """
    Tính log xác suất log P(O|model) của chuỗi quan sát bằng thuật toán Forward.
    
    Tham số:
        model (dict): Mô hình GMM-HMM.
        observations (np.ndarray): Chuỗi quan sát.
        
    Trả về:
        log P(O|model) (float): Log-likelihood của chuỗi quan sát.
    """
    o = observations
    alpha = calc_alpha(model, o)
    # Tổng xác suất tại bước cuối cùng T: P(O|lambda) = sum_i alpha_{T-1}(i)
    prob_seq_f = np.sum(alpha[-1])
    return np.log(prob_seq_f)
    

def backward(model, observations):
    """
    Tính log xác suất log P(O|model) của chuỗi quan sát bằng thuật toán Backward.
    
    Tham số:
        model (dict): Mô hình GMM-HMM.
        observations (np.ndarray): Chuỗi quan sát.
        
    Trả về:
        log P(O|model) (float): Log-likelihood của chuỗi quan sát.
    """
    o = observations
    beta = calc_beta(model, o)
    # Tổng xác suất tại bước đầu: P(O|lambda) = sum_i [pi_i * b_i(o_0) * beta_0(i)]
    s_next = beta[0] * model["B"](model, o[0]) 
    prob_seq_b = np.dot(s_next, model["pi"])
    return np.log(prob_seq_b)


# =============================================================================
# Thuật toán Viterbi (Viterbi Decoding)
# =============================================================================

def decoder(model, observations):
    """
    Thuật toán Viterbi: Tìm chuỗi trạng thái ẩn tối ưu nhất (có xác suất xảy ra cao nhất)
    đã sinh ra chuỗi quan sát cho trước.
    
    Tham số:
        model (dict): Mô hình GMM-HMM.
        observations (np.ndarray): Chuỗi quan sát [N_samples, D].
        
    Trả về:
        prob_max (float): Xác suất lớn nhất của đường đi tối ưu.
        path (np.ndarray): Chuỗi nhãn trạng thái ẩn tối ưu nhất [N_samples].
    """
    o = observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]
    
    # psi[t, i]: Lưu lại trạng thái tối ưu ở thời điểm t-1 đã chuyển đến trạng thái i ở thời điểm t
    psi = np.zeros([N_samples, N_stats])
    
    # delta[t, i]: Xác suất lớn nhất của đường đi trạng thái kết thúc tại trạng thái i ở thời điểm t
    delta = np.zeros([N_samples, N_stats])
    
    # Bước khởi tạo (t = 0): delta_0(i) = pi_i * b_i(o_0)
    delta[0] = model["pi"] * model["B"](model, o[0])
    psi[0] = 0
    
    # Bước quy nạp tiến (t = 1 -> T-1):
    for t in range(1, N_samples):
        for i in range(N_stats):
            states_prev2current = delta[t-1] * model["A"][:, i]
            delta[t][i] = np.max(states_prev2current)
            psi[t][i] = np.argmax(states_prev2current)
        
        delta[t] = delta[t] * model["B"](model, o[t])
            
    # Bước kết thúc và truy vết ngược tìm đường đi tối ưu (Backtracking):
    path = np.zeros(N_samples)
    path[-1] = np.argmax(delta[-1])
    prob_max = np.max(delta[-1])
    
    for t in range(N_samples-2, -1, -1):
        path[t] = psi[t+1][int(path[t+1])]
    
    return prob_max, path 
        

# =============================================================================
# Thuật toán Baum-Welch / EM (Expectation - Maximization)
# =============================================================================

def calcxi(model, observations, alpha, beta):
    """
    Tính biến xi_t(i, j) = P(q_t = S_i, q_{t+1} = S_j | O, model)
    Xác suất tại thời điểm t ở trạng thái S_i và tại thời điểm t+1 ở trạng thái S_j khi biết chuỗi quan sát O.
    
    Tham số:
        model (dict): Mô hình GMM-HMM.
        observations (np.ndarray): Chuỗi quan sát.
        alpha (np.ndarray): Ma trận alpha từ hàm calc_alpha.
        beta (np.ndarray): Ma trận beta từ hàm calc_beta.
        
    Trả về:
        xi (np.ndarray): Mảng 3 chiều [N_samples-1, N_stats, N_stats].
    """
    o = observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]

    xi = np.zeros([N_samples-1, N_stats, N_stats])

    for t in range(N_samples-1):
        temp = np.zeros([N_stats, N_stats])
        
        t_alpha = np.tile(np.expand_dims(alpha[t, :], axis=1), (1, N_stats))
        t_beta = np.tile(beta[t+1, :], (N_stats, 1))
        t_b = np.tile(model["B"](model, o[t+1]), (N_stats, 1))
        
        # xi_t(i, j) = alpha_t(i) * a_{ij} * b_j(o_{t+1}) * beta_{t+1}(j) / P(O|model)
        temp = t_alpha * model["A"] * t_beta * t_b
        temp = temp / np.sum(temp)  # Chuẩn hóa tổng bằng 1
 
        xi[t] = temp
    return xi


def calcgamma(alpha, beta):
    """
    Tính biến gamma_t(i) = P(q_t = S_i | O, model)
    Xác suất tại thời điểm t ở trạng thái S_i khi biết toàn bộ chuỗi quan sát O.
    
    Tham số:
        alpha (np.ndarray): Ma trận alpha.
        beta (np.ndarray): Ma trận beta.
        
    Trả về:
        gamma (np.ndarray): Ma trận gamma [N_samples, N_stats].
    """
    gamma = alpha * beta
    # Chuẩn hóa theo từng hàng (tổng xác suất trên các trạng thái tại mỗi thời điểm = 1)
    gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
    return gamma    
    

def update_pi(collect_gamma):
    """
    Cập nhật phân phối xác suất trạng thái ban đầu pi.
    pi_i = trung bình gamma_0(i) trên toàn bộ các chuỗi huấn luyện.
    
    Tham số:
        collect_gamma (list): Danh sách ma trận gamma của tất cả các chuỗi dữ liệu huấn luyện.
        
    Trả về:
        pi (np.ndarray): Vector phân phối xác suất ban đầu mới [N_stats].
    """
    N_datas = len(collect_gamma)
    _, N_stats = np.shape(collect_gamma[0])
    
    sum_gamma_1 = np.zeros(N_stats)
    for gamma in collect_gamma:
        sum_gamma_1 = sum_gamma_1 + gamma[0]
    
    pi = sum_gamma_1 / N_datas
    return pi
        

def update_A(collect_gamma, collect_xi):
    """
    Cập nhật ma trận xác suất chuyển trạng thái A.
    a_{ij} = (tổng xi_t(i, j)) / (tổng gamma_t(i))
    
    Tham số:
        collect_gamma (list): Danh sách ma trận gamma của các chuỗi dữ liệu.
        collect_xi (list): Danh sách mảng xi của các chuỗi dữ liệu.
        
    Trả về:
        A (np.ndarray): Ma trận chuyển trạng thái mới [N_stats, N_stats].
    """
    _, N_stats = np.shape(collect_gamma[0])
    sum_xi = np.zeros([N_stats, N_stats])
    sum_gamma = np.zeros(N_stats)
    
    for xi in collect_xi:
        sum_xi = sum_xi + np.sum(xi, axis=0)
    
    sum_gamma = np.sum(sum_xi, axis=1, keepdims=True)
    # for gamma in collect_gamma:
    #     sum_gamma = sum_gamma + np.sum(gamma[:-1], axis=0)
    # sum_gamma = np.tile(np.expand_dims(sum_gamma, axis=1), (1, N_stats))

    A = sum_xi / sum_gamma
    return A
    

def update_GMM_in_States(train_datas, model, collect_gamma):
    """
    Cập nhật các tham số của GMM (trọng số w, vector kỳ vọng mu, ma trận hiệp phương sai sigma)
    trong từng trạng thái ẩn theo thuật toán EM.
    
    Tham số:
        train_datas (list): Danh sách các chuỗi quan sát huấn luyện.
        model (dict): Mô hình GMM-HMM hiện tại.
        collect_gamma (list): Danh sách ma trận gamma tương ứng.
        
    Trả về:
        new_states (list): Danh sách các GMM mới cho từng trạng thái ẩn.
    """
    # Gộp tất cả dữ liệu và gamma thành một mảng liên tục
    train_datas = np.concatenate(train_datas, axis=0)
    collect_gamma = np.concatenate(collect_gamma, axis=0)
    
    T, D = np.shape(train_datas)
    N_mix = np.shape(model["S"][0]["ws"])[0]
    N_state = len(model["S"])
    
    # Tính xác suất hậu nghiệm của từng mẫu tại từng trạng thái và từng thành phần mixture: gamma_mix[t, s, m]
    gamma_mix = np.zeros([T, N_state, N_mix])
    
    for t in range(T):
        for s in range(N_state):
            p_mix = np.zeros(N_mix)
            for m in range(N_mix):
                p_mix[m] = getPdf(train_datas[t], model["S"][s]["mus"][m], model["S"][s]["sigmas"][m])
                p_mix[m] = p_mix[m] * model["S"][s]["ws"][m]
            p_mix = p_mix / np.sum(p_mix)
            gamma_mix[t, s, :] = p_mix * collect_gamma[t][s]
    
    # Khởi tạo danh sách GMM mới
    new_states = []
    for s in range(N_state):        
        gmm = dict()
        gmm["ws"] = np.zeros(N_mix)
        gmm["mus"] = np.zeros([N_mix, D])
        gmm["sigmas"] = np.zeros([N_mix, D, D])
        new_states.append(gmm)
    
    # Cập nhật tham số cho từng thành phần Gauss m của trạng thái s
    for s in range(N_state):
        for m in range(N_mix):
            r_k = gamma_mix[:, s, m]
            N_k = np.sum(r_k)
            r_k = r_k[:, np.newaxis]  # [T, 1]

            # Cập nhật vector kỳ vọng mu
            mu = np.sum(train_datas * r_k, axis=0) / N_k
            
            # Cập nhật ma trận hiệp phương sai sigma
            dx = train_datas - model["S"][s]["mus"][m]
            sigma = np.zeros([D, D])
            for t in range(T):
                sigma = sigma + r_k[t, 0] * np.outer(dx[t], dx[t])
            sigma = sigma / N_k
            
            # Thêm một lượng nhỏ vào đường chéo chính để tránh suy biến ma trận nghịch đảo
            sigma = sigma + np.eye(D) * 0.001
            
            # Cập nhật trọng số mixture w
            w = N_k / T
            
            new_states[s]["mus"][m] = mu
            new_states[s]["sigmas"][m] = sigma
            new_states[s]["ws"][m] = w
            
        # Chuẩn hóa trọng số các thành phần mixture để tổng bằng 1
        new_states[s]["ws"] = new_states[s]["ws"] / np.sum(new_states[s]["ws"])
        
    return new_states
            

def train_step_GMM_HMM(train_datas, model):    
    """
    Thực hiện 1 bước lặp huấn luyện Baum-Welch (E-step và M-step) trên tập dữ liệu.
    
    Tham số:
        train_datas (list): Danh sách các chuỗi quan sát.
        model (dict): Mô hình GMM-HMM hiện tại.
        
    Trả về:
        new_A (np.ndarray): Ma trận chuyển trạng thái mới.
        new_pi (np.ndarray): Phân phối xác suất ban đầu mới.
        new_states (list): Danh sách GMM mới cho các trạng thái.
    """
    collect_xi = []
    collect_gamma = []
    
    # Bước E (Expectation step): Tính alpha, beta, xi, gamma cho từng chuỗi
    for datas in train_datas:
        alpha = calc_alpha(model, datas)
        beta = calc_beta(model, datas)
        xi = calcxi(model, datas, alpha, beta)
        gamma = calcgamma(alpha, beta)
        
        collect_gamma.append(gamma)
        collect_xi.append(xi)
    
    # Bước M (Maximization step): Cập nhật các tham số mới A, pi, GMM states
    new_A = update_A(collect_gamma, collect_xi)
    new_pi = update_pi(collect_gamma)
    new_states = update_GMM_in_States(train_datas, model, collect_gamma)

    return new_A, new_pi, new_states
    
    
def compute_prob_for_datas(model, datas):
    """
    Tính tổng log-likelihood của toàn bộ tập chuỗi quan sát đối với mô hình: sum log P(O_k | model).
    """
    results = 0
    for o in datas:
        results += forward(model, o)
    return results
    

def train_GMM_HMM(train_datas, model, n_iteration):
    """
    Vòng lặp huấn luyện hoàn chỉnh mô hình GMM-HMM qua n_iteration lần lặp hoặc khi log-likelihood không tăng.
    
    Tham số:
        train_datas (list): Tập dữ liệu huấn luyện.
        model (dict): Mô hình GMM-HMM khởi tạo ban đầu.
        n_iteration (int): Số vòng lặp tối đa.
        
    Trả về:
        model (dict): Mô hình GMM-HMM đã được huấn luyện tối ưu.
    """
    new_model = model.copy()
    
    prob_old = compute_prob_for_datas(new_model, train_datas)
    print("Prob_first:", prob_old)
    
    for i in range(n_iteration):
        new_A, new_pi, new_states = train_step_GMM_HMM(train_datas, new_model)
        new_model["A"] = new_A
        new_model["pi"] = new_pi
        new_model["S"] = new_states
        new_model["B"] = Pdf_O2GMMs
        
        prob_new = compute_prob_for_datas(new_model, train_datas)
        print("it %d prob %f" % (i, prob_new))
      
        if prob_new > prob_old:
            prob_old = prob_new
            model = new_model
        else: 
            break
            
    return model  
    

def creat_GMM(mus, sigmas, ws):
    """
    Đóng gói các tham số thành một đối tượng GMM dạng từ điển.
    """
    gmm = dict()
    gmm['mus'] = mus
    gmm['sigmas'] = sigmas
    gmm['ws'] = ws
    return gmm


# =============================================================================
# Các hàm tính toán hàm mật độ xác suất (PDF)
# =============================================================================

def getPdf(x, mu, sigma):
    """
    Tính giá trị hàm mật độ xác suất của phân phối Gauss đa biến (Multivariate Normal PDF) tại điểm x:
    f(x) = (1 / ((2*pi)^(D/2) * |Sigma|^0.5)) * exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu))
    
    Tham số:
        x (np.ndarray): Vector điểm dữ liệu quan sát [D].
        mu (np.ndarray): Vector kỳ vọng trung bình [D].
        sigma (np.ndarray): Ma trận hiệp phương sai [D, D].
        
    Trả về:
        pdfval (float): Giá trị mật độ xác suất.
    """
    sigma = np.matrix(sigma)
    D = np.shape(x)[0]
    covar_det = np.linalg.det(sigma)
        
    c = (1 / ((2.0 * np.pi)**(float(D / 2.0)) * (covar_det)**(0.5)))
    pdfval = c * np.exp(-0.5 * np.dot(np.dot((x - mu), sigma.I), (x - mu)))
    return pdfval


def getPdfFromeGMM(x, gmm):
    """
    Tính giá trị hàm mật độ xác suất tổng hợp của mô hình GMM tại điểm x:
    f_GMM(x) = sum_{k=1}^K [ w_k * N(x | mu_k, sigma_k) ]
    
    Tham số:
        x (np.ndarray): Vector quan sát.
        gmm (dict): Cấu trúc GMM.
        
    Trả về:
        temp (float): Giá trị mật độ xác suất GMM.
    """
    K, D = np.shape(gmm["mus"])
    temp = 0
    for k in range(K):
        temp += getPdf(x, gmm["mus"][k], gmm["sigmas"][k]) * gmm["ws"][k]
    return temp


def Pdf_O2GMMs(model, o):
    """
    Tính vector xác suất phát xạ của mẫu quan sát o trên tất cả các trạng thái ẩn (mỗi trạng thái là 1 GMM).
    
    Tham số:
        model (dict): Mô hình GMM-HMM.
        o (np.ndarray): Mẫu quan sát tại một thời điểm.
        
    Trả về:
        pdfs (np.ndarray): Vector xác suất phát xạ tương ứng với từng trạng thái ẩn [N_states].
    """
    states = model["S"]
    N_states = len(states)
    pdfs = np.zeros(N_states)
    
    for i, gmm in enumerate(states):
        pdfs[i] = getPdfFromeGMM(o, gmm)
    return pdfs
        

# =============================================================================
# Kiểm thử và minh họa hoạt động của mô hình GMM-HMM
# =============================================================================
if __name__ == "__main__":
    
    # 1. Khởi tạo mô hình GMM-HMM số 1 (model_GMM_hmm1)
    model_GMM_hmm1 = dict()
    model_GMM_hmm1["pi"] = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
    model_GMM_hmm1["A"] = np.array([
        [0.8, 0.1, 0.1],
        [0.8, 0.1, 0.1],
        [0.8, 0.1, 0.1],
    ])
    states = []

    for i in range(3):
        # 3 thành phần Gauss (K=3), số chiều đặc trưng là 2 (D=2)
        mus = 0.6 * np.random.random_sample([3, 2]) - 0.3  # [K, D]
        sigmas = np.array([np.eye(2, 2) for i in range(3)])  # [K, D, D]
        ws = np.array([0.3, 0.5, 0.2])  # [K]
        gmm = creat_GMM(mus, sigmas, ws)
        states.append(gmm)

    model_GMM_hmm1["S"] = states
    model_GMM_hmm1["B"] = Pdf_O2GMMs

    # Task 0: Sinh dữ liệu mẫu từ model_GMM_hmm1
    datas, states = gen_samples_from_GMM_HMM(model_GMM_hmm1, 50)
    print("Dữ liệu quan sát sinh ra:\n", datas)
    print("Chuỗi trạng thái ẩn sinh ra:\n", states)

    # 2. Khởi tạo mô hình GMM-HMM số 2 (model_GMM_hmm2)
    model_GMM_hmm2 = dict()
    model_GMM_hmm2["pi"] = np.array([0.6, 0.2, 0.2])
    model_GMM_hmm2["A"] = np.array([
        [0.1, 0.1, 0.8],
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
    ])
    states = []

    for i in range(3):
        mus = 0.1 * np.random.random_sample([3, 2]) - 0.5  # [K, D]
        sigmas = np.array([np.eye(2, 2) for i in range(3)])  # [K, D, D]
        ws = np.array([0.1, 0.1, 0.9])  # [K]
        gmm = creat_GMM(mus, sigmas, ws)
        states.append(gmm)

    model_GMM_hmm2["S"] = states
    model_GMM_hmm2["B"] = Pdf_O2GMMs

    # Sinh tập dữ liệu huấn luyện và kiểm tra từ cả 2 mô hình
    train_datas_hmm1 = [gen_samples_from_GMM_HMM(model_GMM_hmm1, 50)[0] for _ in range(90)]
    test_datas_hmm1 = [gen_samples_from_GMM_HMM(model_GMM_hmm1, 50)[0] for _ in range(10)]
    
    train_datas_hmm2 = [gen_samples_from_GMM_HMM(model_GMM_hmm2, 50)[0] for _ in range(90)]
    test_datas_hmm2 = [gen_samples_from_GMM_HMM(model_GMM_hmm2, 50)[0] for _ in range(10)]

    # Khởi tạo một mô hình GMM-HMM ngẫu nhiên để làm điểm bắt đầu huấn luyện
    init_hmm = dict()
    N_state = 3
    N_mix = 3
    D = 2
    
    pi = np.random.random_sample(N_state)
    pi = pi / sum(pi)
    init_hmm["pi"] = pi
    
    a = np.random.random_sample((N_state, N_state))
    row_sums = a.sum(axis=1)
    a = a / row_sums[:, np.newaxis]  
    init_hmm["A"] = a
    
    states = []
    for i in range(N_state):
        mus = np.random.random_sample([N_mix, D])
        sigmas = np.array([np.eye(D, D) for _ in range(3)])
        ws = np.ones(N_mix) * (1.0 / N_mix)
        gmm = creat_GMM(mus, sigmas, ws)
        states.append(gmm)
        
    init_hmm["S"] = states
    init_hmm["B"] = Pdf_O2GMMs

    # Huấn luyện 2 mô hình GMM-HMM riêng biệt trên dữ liệu tương ứng
    new_gmm_hmm1 = train_GMM_HMM(train_datas_hmm1, init_hmm, 1)
    new_gmm_hmm2 = train_GMM_HMM(train_datas_hmm2, init_hmm, 1)
    
    # Đánh giá và kiểm tra phân loại chuỗi
    N_test1 = len(test_datas_hmm1)
    N_test2 = len(test_datas_hmm2)
    labs = [0 for _ in range(N_test1)] + [1 for _ in range(N_test2)]
    test_datas = test_datas_hmm1 + test_datas_hmm2
    
    for i, test_data in enumerate(test_datas):
        score1 = forward(new_gmm_hmm1, test_data)
        score2 = forward(new_gmm_hmm2, test_data)
        det_lab = np.argmax([score1, score2])
        print("Score1: %f, Score2: %f -> Dự đoán: %d, Thực tế: %d" % (score1, score2, det_lab, labs[i]))