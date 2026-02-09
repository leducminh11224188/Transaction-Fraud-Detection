# Transaction Fraud Detection System / Hệ thống Phát hiện Gian lận Giao dịch

[English](#english) | [Tiếng Việt](#tiếng-việt)

---

<a name="english"></a>
## 🇬🇧 English

### 1. Overview
This project implements a robust fraud detection system based on the IEEE-CIS Fraud Detection dataset. It leverages a **LightGBM** model for high-performance tabular data classification and provides a **FastAPI** service for real-time inference.

The goal is to identify fraudulent transactions accurately while maintaining low latency for real-time applications.

### 2. Key Features
- **Data Pipeline**: Analysis, preprocessing, and standardizing of transaction data (IEEE-CIS).
- **Feature Engineering**: Custom logic to extracting meaningful patterns from transaction history (`src/features.py`).
- **Model**: LightGBM Classifier, optimized for memory efficiency and speed (`src/train.py`).
- **Real-time API**: REST Endpoint built with FastAPI to serve predictions (`src/api/`).
- **Monitoring**: Integration with Prometheus for tracking API usage and performance.
- **Docker Support**: Containerized environment for reproducible deployment.

### 3. Project Structure
The project is organized as follows:
```text
├── config/              # Configuration files
├── data/                # Data storage (IEEE-CIS dataset)
│   └── IEEE-CIS/
├── models/              # Saved models and artifacts (LightGBM model, feature maps)
├── notebooks/           # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── src/                 # Source code
│   ├── api/             # FastAPI application and schemas
│   ├── simulator/       # Transaction simulation modules
│   ├── utils/           # Utility functions (logging, etc.)
│   ├── features.py      # Feature engineering logic
│   ├── preprocessing.py # Data cleaning and transformation
│   └── train.py         # Model training pipeline
├── tests/               # Unit tests
├── Dockerfile           # Docker image configuration
└── requirements.txt     # Python project dependencies
```

### 4. Setup & Installation

#### Prerequisites
- Python 3.9+
- Docker (optional)

#### 1. Clone the repository
```bash
git clone <repo-url>
cd "Transaction Fraud Detection"
```

#### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

#### 3. Data Preparation
Download the IEEE-CIS Fraud Detection dataset and place it in the `data/IEEE-CIS/` directory. The required files are:
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

### 5. Usage

#### Training the Model
To execute the training pipeline (load data, process features, train LightGBM), run:
```bash
python src/train.py
```
*Artifacts (model files, feature names) will be saved to the `models/` directory.*

#### Running the API Server
Start the FastAPI application:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### 6. Docker Deployment
You can build and run the entire application using Docker.

```bash
# Build the image
docker build -t fraud-detection .

# Run the container
docker run -p 8000:8000 fraud-detection
```

---

<a name="tiếng-việt"></a>
## 🇻🇳 Tiếng Việt

### 1. Tổng quan
Dự án này xây dựng một hệ thống phát hiện gian lận giao dịch dựa trên bộ dữ liệu IEEE-CIS Fraud Detection. Hệ thống sử dụng mô hình **LightGBM** để phân loại với hiệu năng cao và cung cấp dịch vụ **FastAPI** cho việc dự đoán thời gian thực.

Mục tiêu là phát hiện chính xác các giao dịch gian lận trong khi vẫn đảm bảo độ trễ thấp cho các ứng dụng thực tế.

### 2. Tính năng chính
- **Xử lý dữ liệu**: Phân tích, tiền xử lý và chuẩn hóa dữ liệu giao dịch (IEEE-CIS).
- **Kỹ thuật đặc trưng (Feature Engineering)**: Logic tùy chỉnh để trích xuất các mẫu quan trọng từ lịch sử giao dịch (`src/features.py`).
- **Mô hình**: LightGBM Classifier, được tối ưu hóa về tốc độ và bộ nhớ (`src/train.py`).
- **API thời gian thực**: REST Endpoint được xây dựng với FastAPI để phục vụ dự đoán (`src/api/`).
- **Giám sát**: Tích hợp Prometheus để theo dõi hiệu suất và lưu lượng API.
- **Hỗ trợ Docker**: Môi trường container hóa giúp triển khai dễ dàng và đồng nhất.

### 3. Cấu trúc dự án
Cấu trúc thư mục của dự án như sau:
```text
├── config/              # Các file cấu hình
├── data/                # Nơi lưu trữ dữ liệu (IEEE-CIS dataset)
│   └── IEEE-CIS/
├── models/              # Nơi lưu model đã huấn luyện và các artifact
├── notebooks/           # Jupyter notebooks phân tích dữ liệu (EDA)
├── src/                 # Mã nguồn chính
│   ├── api/             # Ứng dụng FastAPI và schemas
│   ├── simulator/       # Module giả lập giao dịch
│   ├── utils/           # Các hàm tiện ích (logging, v.v.)
│   ├── features.py      # Logic tính toán đặc trưng (Feature Engineering)
│   ├── preprocessing.py # Làm sạch và chuyển đổi dữ liệu
│   └── train.py         # Quy trình huấn luyện mô hình
├── tests/               # Unit tests
├── Dockerfile           # Cấu hình Docker image
└── requirements.txt     # Danh sách các thư viện Python cần thiết
```

### 4. Cài đặt

#### Yêu cầu hệ thống
- Python 3.9 trở lên
- Docker (tùy chọn)

#### 1. Tải dự án
```bash
git clone <repo-url>
cd "Transaction Fraud Detection"
```

#### 2. Cài đặt thư viện
Khuyên dùng môi trường ảo (virtual environment).
```bash
pip install -r requirements.txt
```

#### 3. Chuẩn bị dữ liệu
Tải bộ dữ liệu IEEE-CIS Fraud Detection và đặt vào thư mục `data/IEEE-CIS/`. Các file cần thiết bao gồm:
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

### 5. Hướng dẫn sử dụng

#### Huấn luyện mô hình
Để chạy quy trình huấn luyện (tải dữ liệu, xử lý đặc trưng, train LightGBM), chạy lệnh:
```bash
python src/train.py
```
*Các file model và danh sách đặc trưng sẽ được lưu vào thư mục `models/`.*

#### Chạy API Server
Khởi động ứng dụng FastAPI:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- **Tài liệu API**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Kiểm tra trạng thái (Health Check)**: [http://localhost:8000/health](http://localhost:8000/health)

### 6. Triển khai với Docker
Bạn có thể xây dựng và chạy toàn bộ ứng dụng bằng Docker.

```bash
# Xây dựng image
docker build -t fraud-detection .

# Chạy container
docker run -p 8000:8000 fraud-detection
```
