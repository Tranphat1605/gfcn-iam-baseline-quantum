# GFCN trên IAM: baseline và bộ lọc 4 qubit

Repository gồm hai mô hình nhận dạng chữ viết tay theo dòng, kèm checkpoint đã train, lịch sử huấn luyện và kết quả test thực tế. Bộ lọc quantum được mô phỏng bằng PyTorch, không cần máy tính lượng tử hay PennyLane để chạy.

| Mô hình | Epoch đã train | Checkpoint tốt nhất | CER validation | CER test | WER test |
|---|---:|---|---:|---:|---:|
| GFCN baseline | 129 | `best_126.pt` (epoch 127) | 5,84% | 8,77% | 31,12% |
| GFCN + bộ lọc 4 qubit cố định | 98 | `best_90.pt` (epoch 91) | 6,96% | 9,87% | 35,87% |

Test ngày 28/11/2026 trên cùng 2.915 dòng, 123.090 ký tự; batch size 2, greedy CTC, không dùng mô hình ngôn ngữ. Checkpoint được chọn bằng CER validation. Quantum hiện có CER test cao hơn baseline khoảng 1,10 điểm phần trăm; hai bản chưa có cùng số epoch huấn luyện. Chưa có bằng chứng bộ lọc này cải thiện kết quả.

## Các thư mục

- `baseline/source/`: code GFCN; `quantum/source/`: code GFCN với bộ lọc 4 qubit.
- `*/source/outputs/*/checkpoints/`: mỗi bản có checkpoint `best` để đánh giá và `last` để tiếp tục train, bao gồm trạng thái Adam.
- `*/Datasets/iam_lines/iam_lines_64H_NKR/params.pkl`: bảng ký tự và thống kê train dùng cho inference.
- `results/`: CER/WER, lịch sử train, SHA-256 của checkpoint. Chỉ checkpoint cuối và tốt nhất còn được giữ; các checkpoint trung gian trong lịch sử không đi kèm.
- `predict.py`: nhận dạng một ảnh dòng chữ bằng một trong hai mô hình.
- `run.py`: chạy toàn bộ test hoặc tiếp tục train với dataset của bạn.

Không kèm ảnh IAM, dữ liệu HDF5, môi trường Python, log lớn hoặc các bản backup. Code gốc và kết quả trên máy tác giả được giữ riêng.

## Cài đặt

Dùng **Python 3.11**, PyTorch 2.0.1 và NumPy 1.24.4 như môi trường đã kiểm tra. Tại thư mục repository trên Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_artifacts.py
```

Trên Linux, dùng `python3.11 -m venv .venv` và thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`. Có thể chạy bằng CPU; CUDA cần GPU NVIDIA phù hợp với bản PyTorch cu117.

## Chạy thử ảnh của bạn

Ảnh phải được cắt thành **một dòng chữ**, ưu tiên chữ đen nền trắng. Mô hình học bảng ký tự tiếng Anh của IAM. Không cần tải toàn bộ IAM cho bước này.

```powershell
.\.venv\Scripts\python.exe predict.py --model baseline --image "D:\anh\dong_chu.png" --device cpu
.\.venv\Scripts\python.exe predict.py --model quantum --image "D:\anh\dong_chu.png" --device cpu
```

Đầu ra JSON gồm tên mô hình, checkpoint và chuỗi `prediction`. Tiền xử lý khớp lúc train: grayscale, chiều cao 64, giữ nguyên chiều rộng gốc, chuẩn hóa theo thống kê train. Bộ lọc quantum lấy patch 2×2, stride 1 và xuất bốn kênh kỳ vọng Z; mạch cố định gồm hai lớp và seed 0. Phần GFCN được huấn luyện bằng CTC.

## Đánh giá tập test

Chuẩn bị bộ IAM đã định dạng có `train/` (6.482 file), `valid/` (976), `test/` (2.915), `params.pkl` và `params.txt`. Có thể sao chép thư mục `Datasets/iam_lines/iam_lines_64H_NKR` từ bản local đã chuẩn bị, rồi chỉ định cùng một đường dẫn cho cả hai mô hình:

```powershell
.\.venv\Scripts\python.exe run.py --model baseline --mode test --dataset-path "D:\IAM\iam_lines_64H_NKR" --device cuda
.\.venv\Scripts\python.exe run.py --model quantum --mode test --dataset-path "D:\IAM\iam_lines_64H_NKR" --device cuda
```

Kết quả được ghi dưới `baseline/source/outputs/.../results/` và `quantum/source/outputs/.../results/`. Đây là kết quả mới của máy chạy; kết quả đã công bố trong repo nằm riêng ở `results/`.

Nếu chuẩn bị từ ảnh IAM gốc, lấy bộ split XML từ [source GFCN gốc](https://github.com/FactoDeepLearning/LinePytorchOCR), đặt trong `baseline/preformat iam/xml/`, rồi dùng `preformat_iam.py` và `source/format_data.py`. Không đổi split, bảng ký tự hay thống kê train nếu muốn so sánh với các checkpoint này. Dataset [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) là một cách đóng gói khác, không thay trực tiếp cho dữ liệu HDF5 của repo.

## Tiếp tục train

```powershell
.\.venv\Scripts\python.exe run.py --model baseline --mode train --dataset-path "D:\IAM\iam_lines_64H_NKR" --device cuda
.\.venv\Scripts\python.exe run.py --model quantum --mode train --dataset-path "D:\IAM\iam_lines_64H_NKR" --device cuda
```

Chạy lần lượt khi dùng một GPU. Lệnh train nạp checkpoint `last` cùng Adam, tiếp tục lần lượt từ epoch 130 và 99. Mục tiêu mặc định của code là tổng 1000 epoch; nên chủ động theo dõi validation và dừng an toàn sau khi lưu epoch. Không chọn checkpoint dựa trên test CER.

## Nguồn và ghi nhận

- Coquenet, Chatelain, Paquet, *Recurrence-free unconstrained handwritten text recognition using gated fully convolutional network*: [bài báo](https://arxiv.org/abs/2012.04961), [source GFCN](https://github.com/FactoDeepLearning/LinePytorchOCR), commit nền `d2d7541675191a2b8adf27c1d47b6da58a6df027`.
- Henderson et al., *Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits*: [bài báo](https://arxiv.org/abs/1904.04767).
- [PennyLane Quanvolution demo](https://pennylane.ai/demos/tutorial_quanvolution): tham khảo patch 2×2, mã hóa góc và đo Z. Stride 1, padding và cách ghép với GFCN là lựa chọn của thí nghiệm này.
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database): dữ liệu nghiên cứu gốc. Tự chuẩn bị dataset theo điều kiện cung cấp của nguồn dữ liệu.

Giữ thông báo bản quyền của tác giả GFCN trong source. Xem `LICENSE.md` (CeCILL-C) cho code kế thừa. Các file checkpoint là sản phẩm huấn luyện của dự án này; dữ liệu IAM không được đóng gói trong repository.
