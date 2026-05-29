📂 CAREER_GUIDANCE_AI (Root Folder)

📁 data/ (Chứa các file dữ liệu)

raw_jobs.csv: Dữ liệu thô vừa cào về.

dataset_model_c_ready.csv: Dữ liệu đã làm sạch cơ bản.

FINAL_MINING_DATASET.csv: Dữ liệu đã qua gom cụm và làm giàu.

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
