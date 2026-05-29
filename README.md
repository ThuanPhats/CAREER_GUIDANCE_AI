# 📂 CAREER_GUIDANCE_AI

## 👥 Thành viên nhóm và Phân công công việc
- **La Thuận Phát**: Triển khai ứng dụng, xử lý các quy trình nghiệp vụ ứng dụng và viết báo cáo.
- **Hồ Chí Dũng**: Kiểm định mô hình, thay đổi thông số, train mô hình và làm slide báo cáo.
- **Nguyễn Thị Thùy Dương**: Thu thập dữ liệu, tiền xử lý dữ liệu, tìm hiểu và build cấu trúc mô hình.
- **Nguyễn Phú Trọng**: Tham gia triển khai ứng dụng.

> **Lưu ý về Dữ liệu (Data/Models):** Đối với các file dữ liệu và mô hình có kích thước lớn, nhóm đã lưu trữ và upload lên **Google Drive**. Bạn cần tải dữ liệu từ Drive về và đặt vào đúng các thư mục tương ứng (`data/`, `models/`, `outputs/`...) trước khi chạy dự án.

## 📁 Cấu trúc thư mục (Root Folder)

📁 data/ (Chứa các file dữ liệu)

raw_jobs.csv: Dữ liệu thô vừa cào về (Phát phụ trách).

dataset_model_c_ready.csv: Dữ liệu đã làm sạch cơ bản (Phát).

FINAL_MINING_DATASET.csv: Dữ liệu đã qua gom cụm và làm giàu (Dương & Dũng).

📁 notebooks/ (Chứa các file chạy thử nghiệm .ipynb nếu có)

eda_exploration.ipynb: File vẽ biểu đồ, WordCloud.

clustering_experiment.ipynb: Thử nghiệm số cụm KMeans.

📁 src/ (Chứa mã nguồn chính - Source code)

cleaning_pipeline.py: Code làm sạch dữ liệu (No5.py).

mining_engine.py: Code chạy KMeans và Apriori.

utils.py: Các hàm bổ trợ (xử lý regex, lọc từ khóa).

📁 models/ (Lưu trữ các model đã huấn luyện)

kmeans_model.pkl: File model KMeans đã lưu để dùng lại.

tfidf_vectorizer.pkl: File lưu bộ chuyển đổi văn bản.

📁 web_app/ (Chứa code giao diện tư vấn)

app.py: File chạy chính (Flask hoặc Streamlit).

📁 templates/: Chứa các file HTML.

📁 static/: Chứa file CSS, hình ảnh giao diện.

📁 reports/ (Chứa kết quả đầu ra cho đồ án)

MINING_REPORT.txt: File báo cáo chỉ số.

📁 figures/: Chứa các ảnh biểu đồ (WordCloud, Heatmap) để dán vào Word.

requirements.txt: Danh sách các thư viện cần cài (pandas, scikit-learn, mlxtend...).

README.md: Hướng dẫn cách chạy dự án.
