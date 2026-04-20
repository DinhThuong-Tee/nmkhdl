Dự báo Môi trường Nuôi trồng Cá giò và Hàu - Quảng Ninh (HUST IT4930)

Dự án ứng dụng Học máy để dự báo 21 chỉ số chất lượng nước tại 99 trạm quan trắc
khu vực biển Quảng Ninh, phục vụ đánh giá mức độ phù hợp sinh cảnh (HSI) cho
nuôi trồng cá giò và hàu.

## 👥 Phân công nhiệm vụ thực hiện

- **Hồ Lương An:** Xây dựng thuật toán tiền xử lý dữ liệu và kỹ thuật tạo biến trễ thời gian.
- **Phạm Quang Khánh:** Triển khai thuật toán XGBoost và cơ chế dự báo đa đầu ra (Multi-output).
- **Đinh Văn Thượng:** Thiết lập mạng nơ-ron LSTM và phối hợp xây dựng mô hình lai LSTM-XGBoost.
- **Nguyễn Phương Linh:** Áp dụng kỹ thuật Học chuyển giao để tinh chỉnh mô hình trên dữ liệu địa phương.
- **Chu Phúc Anh:** Lập trình thuật toán tính chỉ số phù hợp sinh học HSI cho các đối tượng nuôi trồng.
- **Vũ Phương Ly:** Phát triển thuật toán xác định bán kính ảnh hưởng không gian dựa trên biến động sinh thái.

🌟 Tính năng nổi bật

- Dự báo đa mục tiêu (Multi-output Forecasting): Dự báo đồng thời 21 thông số
  môi trường (pH, Oxy hòa tan, kim loại nặng...).
- Học chuyển giao (Transfer Learning): Kế thừa tri thức từ bộ dữ liệu lớn của
  Hồng Kông (HK Dataset) để tối ưu hóa dự báo cho dữ liệu nhỏ tại Quảng Ninh.
- Chỉ số HSI (Habitat Suitability Index): Tự động đánh giá mức độ phù hợp nuôi
  trồng dựa trên ngưỡng sinh học QCVN.
- Bản đồ tương tác: Hiển thị 99 trạm quan trắc với mã màu cảnh báo và bán kính
  ảnh hưởng sinh thái.
- Mô hình lai LSTM-XGBoost: (R&D) Tối ưu hóa dự báo chuỗi thời gian cho các
  biến có độ nhiễu cao.

🏗 Kiến trúc hệ thống

Dự án được xây dựng dựa trên luồng xử lý:

1.  Tiền xử lý: Làm sạch dữ liệu, xử lý giá trị khuyết, tạo biến trễ (Lag
    features).
2.  Base Model: Huấn luyện XGBoost trên dữ liệu Hồng Kông.
3.  Fine-tuning: Tinh chỉnh mô hình trên dữ liệu thực tế Quảng Ninh (2021-2024).
4.  Application: Tính toán HSI và hiển thị lên giao diện Dashboard.

🛠 Công nghệ sử dụng

- Ngôn ngữ: Python
- Thư viện Data Science: pandas, NumPy, scikit-learn, XGBoost, PyTorch (cho mô
  hình LSTM).
- Giao diện: Streamlit, Folium (Bản đồ), Plotly (Biểu đồ tương tác).

🚀 Cài đặt và Chạy thử

1. Clone repository

git clone https://github.com/DinhThuong-Tee/nmkhdl.git

2. Cài đặt thư viện

pip install -r requirements.txt

3. Chạy Dashboard
   - python model/basemodel.py
   - python model/finetune_cobia.py
   - python model/finetune_oyster.py
   - python model/metal.py
   - streamlit run interface/main.py

📁 Cấu trúc thư mục

- /data: Chứa dữ liệu đã xử lý (Quảng Ninh & HK).
- /models: Các mô hình đã huấn luyện dưới dạng .pkl và .pt.
- /notebooks: File Jupyter Notebook phân tích và huấn luyện.
- app.py: Mã nguồn chính của giao diện Streamlit.
- utils.py: Các hàm bổ trợ tính toán HSI và xử lý bản đồ.

📈 Kết quả thực nghiệm

Mô hình sau khi áp dụng Transfer Learning đạt được những cải thiện đáng kể so
với mô hình cơ sở:

- Chỉ số pH: Giảm sai số RMSE tới 50.3%.
- Oxy hòa tan (DO): Giảm sai số RMSE 23.2%.
- Độ mặn: Cải thiện độ chính xác 25.9%.

📄 Giấy phép

Dự án được thực hiện cho mục đích học tập tại Đại học Bách Khoa Hà Nội.

Mẹo cho bạn:

1.  Requirements.txt: Hãy nhớ tạo file này bằng cách chạy lệnh pip freeze >
    requirements.txt trong môi trường ảo của bạn.
2.  Screenshots: Hãy chụp 1-2 ảnh đẹp về giao diện Dashboard và lưu vào thư mục
    /assets, sau đó chèn vào README để nhìn chuyên nghiệp hơn. (Ví dụ:
    ![Dashboard Demo](assets/demo.png)).
3.  About section: Trên GitHub, hãy thêm các tag như #machine-learning,
    #aquaculture, #hust, #streamlit để project dễ được tìm thấy.
