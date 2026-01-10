### Hệ thống Phát hiện Gian lận Giao dịch Thời gian thực (Hybrid Model)
1. Tổng quan dự án
Dự án xây dựng hệ thống phát hiện gian lận giao dịch tài chính theo mô hình End-to-End. Hệ thống kết hợp sức mạnh của LightGBM (dữ liệu bảng) và Graph Neural Networks - GNN (dữ liệu đồ thị) để khai thác các mối quan hệ phức tạp giữa các thực thể như Người dùng (User), Thẻ (Card), và Cửa hàng (Merchant).

Điểm nổi bật của dự án:

Xoay quanh kiến trúc Streaming thay vì xử lý Batch truyền thống.

Sử dụng Feature Store (Redis) để tính toán đặc trưng thời gian thực.

Áp dụng Hybrid Model để tăng độ chính xác và giảm tỷ lệ báo động giả (False Positives).

2. Kiến trúc hệ thống
Hệ thống bao gồm các thành phần chính:

Data Simulator: Script mô phỏng luồng giao dịch liên tục từ tập dữ liệu (PaySim/IEEE-CIS).

API Service (FastAPI): Tiếp nhận yêu cầu, điều phối dữ liệu và trả về kết quả dự đoán.

Feature Store (Redis): Lưu trữ và cập nhật các đặc trưng cửa sổ thời gian (Window features).

Graph Engine: Trích xuất các đặc trưng quan hệ (Embeddings) từ đồ thị giao dịch.

Inference Engine: Mô hình Hybrid thực hiện dự đoán thời gian thực với độ trễ thấp.

3. Danh mục công nghệ
Ngôn ngữ: Python 3.9+

Học máy: LightGBM, DGL (Deep Graph Library) hoặc PyTorch Geometric.

Backend: FastAPI, Uvicorn.

Cơ sở dữ liệu: Redis (Feature Store), PostgreSQL (Transaction Logs).

DevOps: Docker, Docker Compose.

4. Cấu trúc thư mục
Plaintext

.
├── data/                   # Thư mục chứa dữ liệu (đã được cấu hình gitignore)
├── notebooks/              # Phân tích dữ liệu (EDA) và thử nghiệm mô hình
├── src/
│   ├── api/                # Code xử lý FastAPI và các Endpoints
│   ├── features/           # Logic tính toán Window & Graph features
│   ├── models/             # Cấu trúc mô hình Hybrid và luồng huấn luyện
│   ├── simulator/          # Script giả lập luồng dữ liệu streaming
│   └── utils/              # Các hàm hỗ trợ (Kết nối DB, Logging)
├── docker-compose.yml      # Cấu hình triển khai hệ thống bằng Docker
├── requirements.txt        # Danh sách thư viện cần thiết
└── README.md
5. Hướng dẫn cài đặt và Triển khai
1. Sao chép dự án (Clone)
Bash

git clone https://github.com/yourusername/fraud-detection-e2e.git
cd fraud-detection-e2e
2. Thiết lập môi trường
Khuyến khích sử dụng Docker để triển khai nhanh chóng và đồng nhất:

Bash

docker-compose up --build
Hoặc cài đặt thủ công trên môi trường ảo (venv):

Bash

pip install -r requirements.txt
3. Chạy trình giả lập dữ liệu
Sau khi API đã khởi động thành công, chạy script sau để bắt đầu mô phỏng luồng giao dịch:

Bash

python src/simulator/stream_data.py
6. Tiêu chí đánh giá
Do đặc thù dữ liệu gian lận mất cân bằng nghiêm trọng, dự án tập trung tối ưu hóa:

Precision-Recall AUC (PR-AUC): Thay vì chỉ dùng Accuracy.

F1-Score: Cân bằng giữa việc bắt đúng gian lận và hạn chế chặn nhầm khách hàng.

Inference Latency: Mục tiêu xử lý dưới 200ms cho mỗi giao dịch.