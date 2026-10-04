# Eye Movement Classification with GMM-HMM

Project phân loại chuyển động mắt thành ba nhãn:

- **Fixation** (`0`)
- **Smooth Pursuit** (`1`)
- **Saccade** (`2`)

Pipeline chính sử dụng mô hình **GMM-HMM phân cấp**:

1. Tìm số segment tối ưu bằng K-means và phương pháp elbow.
2. Huấn luyện GMM-HMM ở tầng 1 trên các đặc trưng vị trí (`x`, `y`) hoặc (`x`, `y`, `v`).
3. Giải mã trạng thái bằng Viterbi.
4. Huấn luyện GMM-HMM tầng 2 trên vận tốc `v` trong từng segment.
5. Ánh xạ trạng thái sang ba nhãn chuyển động mắt dựa trên vận tốc trung bình.
6. Đánh giá bằng Accuracy, Precision, Recall và F1-score.

## Cấu trúc thư mục

```text
eyemovement/
├── dataset/
│   ├── testdataset/              # 87 file dữ liệu đầu vào .txt
│   └── results/                  # Nhãn thủ công tương ứng *_results.csv
├── paper_code/                   # Mã tham khảo và các notebook gốc
├── hyperparametter.ipynb         # Grid search 108 cấu hình
├── compare_xy_with_paper.ipynb   # So sánh cấu hình ['x', 'y'] với Paper
├── gmm_hmm_eye.ipynb             # Notebook thử nghiệm pipeline
├── evaluate_all_dataset.py       # Đánh giá trên toàn bộ dataset
├── benchmark_hyperparameter.py   # Đo thời gian chạy và ước lượng grid search
├── compare_xy_with_paper.py      # So sánh bằng script
├── grid_search_results.csv       # Kết quả grid search
├── xy_vs_paper_comparison.csv    # So sánh metric tổng thể
├── xy_vs_paper_per_label.csv     # So sánh metric theo từng nhãn
└── xy_vs_jemr_comparison.png     # Biểu đồ kết quả đã xuất
```

## Yêu cầu môi trường

- Python 3.10 trở lên
- Jupyter Notebook hoặc JupyterLab
- Khuyến nghị máy có tối thiểu 16 GB RAM khi chạy toàn bộ grid search

Các thư viện chính:

```text
numpy
pandas
scikit-learn
hmmlearn
scipy
matplotlib
joblib
jupyter
```

Cài đặt bằng PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn hmmlearn scipy matplotlib joblib jupyter
```

Nếu PowerShell chặn việc kích hoạt môi trường ảo, có thể chạy trực tiếp Python trong `.venv\Scripts\python.exe`.

## Định dạng dữ liệu

### File dữ liệu đầu vào

Các file trong `dataset/testdataset/` là file văn bản có tối thiểu ba cột:

```text
x y v
```

Trong đó:

- `x`: tọa độ theo trục x.
- `y`: tọa độ theo trục y.
- `v`: vận tốc đã tính.

### File nhãn

Mỗi file dữ liệu phải có một file nhãn tương ứng trong `dataset/results/`:

```text
dataset/testdataset/tester01_1.txt
dataset/results/tester01_1_results.csv
```

CSV nhãn phải có cột:

```text
manually results
```

Giá trị nhãn hợp lệ là `0`, `1` và `2`.

## Chạy đánh giá trên toàn bộ dataset

Từ thư mục project, chạy:

```powershell
python evaluate_all_dataset.py
```

Script sẽ đọc các file trong `dataset/testdataset/`, ghép với nhãn trong `dataset/results/`, sau đó in:

- Accuracy trung bình và độ lệch chuẩn.
- Precision, Recall và F1 macro.
- Precision, Recall và F1 của từng nhãn.

## Chạy benchmark tốc độ

Để đo thời gian chạy một cấu hình và ước lượng thời gian cho toàn bộ grid:

```powershell
python benchmark_hyperparameter.py
```

Benchmark sử dụng đa tiến trình. Số worker mặc định là `8` và các thư viện BLAS được giới hạn một thread trong mỗi process để tránh oversubscription.

## Chạy grid search

Mở [hyperparametter.ipynb](./hyperparametter.ipynb), sau đó chọn:

```text
Restart Kernel → Run All
```

Notebook chạy trên toàn bộ 87 file và tìm kiếm 108 tổ hợp:

| Tham số | Giá trị |
|---|---|
| `L1_FEATURES` | `['x', 'y']`, `['x', 'y', 'v']` |
| `COV_TYPE` | `full`, `diag`, `spherical` |
| `N_MIX` | `1`, `2`, `3` |
| `MIN_SEG_LEN` | `10`, `15`, `20` |
| `STANDARDIZE` | `True`, `False` |

Notebook có tối ưu tốc độ nhưng không thay đổi pipeline:

- Đọc và cache dữ liệu một lần.
- Cache số segment `k` theo từng file.
- Dùng `joblib` với backend `loky` để chạy song song.
- Giới hạn số thread BLAS trong worker.

Kết quả được lưu vào:

```text
grid_search_results.csv
```

CSV bao gồm metric tổng thể và metric theo từng nhãn, ví dụ:

```text
Precision_Fixation
Recall_Fixation
F1_Fixation
Precision_Smooth_Pursuit
Recall_Smooth_Pursuit
F1_Smooth_Pursuit
Precision_Saccade
Recall_Saccade
F1_Saccade
```

## So sánh với Paper

Sau khi có `grid_search_results.csv`, mở [compare_xy_with_paper.ipynb](./compare_xy_with_paper.ipynb) và chạy:

```text
Restart Kernel → Run All
```

Notebook sẽ:

1. Lọc các cấu hình dùng tầng 1 `['x', 'y']`.
2. Chọn cấu hình có `Mean_Accuracy` cao nhất.
3. So sánh Accuracy, Precision, Recall và F1 tổng thể với Paper.
4. So sánh F1 của Fixation, Smooth Pursuit và Saccade với Paper.
5. Hiển thị hai biểu đồ có cùng quy ước màu:
   - Grid Search: xanh.
   - Paper: cam.
6. Ghi giá trị phần trăm trên đầu mỗi cột.

Các kết quả được xuất ra:

```text
xy_vs_paper_comparison.csv
xy_vs_paper_per_label.csv
```

Có thể chạy phiên bản script bằng:

```powershell
python compare_xy_with_paper.py
```

## Kết quả tham khảo hiện tại

Với cấu hình `['x', 'y']` tốt nhất đã được ghi trong kết quả hiện tại, các metric tổng thể được báo cáo xấp xỉ:

| Metric | Grid Search | Paper |
|---|---:|---:|
| Accuracy | 87.07% | 94.39% |
| Precision | 85.06% | 95.31% |
| Recall | 84.22% | 94.98% |
| F1 | 82.96% | 94.92% |

Các giá trị trên chỉ là kết quả của file CSV hiện có. Nếu chạy lại grid search, hãy chạy lại notebook so sánh để cập nhật bảng và biểu đồ.

## Lưu ý

- Không dùng đường dẫn tuyệt đối kiểu `D:\...`; các script chính sử dụng đường dẫn tương đối theo vị trí file.
- Cần giữ đúng quy tắc tên giữa file `.txt` và file nhãn `*_results.csv`.
- Nếu thiếu nhãn hoặc số dòng giữa dữ liệu và nhãn không khớp, quá trình đánh giá sẽ báo lỗi.
- Kết quả của project là trung bình trên dataset hiện tại; kết quả trong Paper dùng thiết lập thực nghiệm của tác giả nên chỉ nên xem là mốc tham khảo.
- File PDF Paper được giữ trong project để đối chiếu số liệu, không phải dữ liệu huấn luyện.
