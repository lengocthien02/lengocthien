# 📊 Dự báo giá vàng bằng SARIMA 📈

Dự án này sử dụng mô hình **SARIMA** và **biến đổi Box-Cox** để phân tích và dự báo giá vàng.  
Quy trình bao gồm kiểm tra tính dừng của dữ liệu, lựa chọn mô hình tối ưu bằng **Grid Search**, và dự báo giá vàng trong 24 tháng tới.

---

## 📂 Tổng quan dự án

🔹 **Nguồn dữ liệu:** XAU/USD daily price (`XAU_1d_data.csv`)  
🔹 **Công cụ sử dụng:** Python, Pandas, Matplotlib, Statsmodels, Scipy, Seaborn  
🔹 **Mô hình dự báo:** SARIMA với tối ưu hóa Grid Search  

---

## 📖 Giới thiệu về thuật toán SARIMA

### 1️⃣ **SARIMA là gì?**
SARIMA (**Seasonal Autoregressive Integrated Moving Average**) là một mở rộng của mô hình **ARIMA**, giúp mô hình hóa dữ liệu chuỗi thời gian có yếu tố mùa vụ.
SARIMA gồm các thành phần chính:

- **AR (Auto-Regressive - Tự hồi quy):** Phụ thuộc vào giá trị của chính nó trong quá khứ.
- **I (Integrated - Tích hợp):** Khác biệt hóa để làm cho chuỗi trở nên dừng.
- **MA (Moving Average - Trung bình trượt):** Dựa vào nhiễu của mô hình trong quá khứ.
- **S (Seasonality - Mùa vụ):** Dự báo dựa trên chu kỳ mùa vụ.

Mô hình SARIMA được viết dưới dạng **SARIMA(p, d, q) x (P, D, Q, m)**, trong đó:
- **(p, d, q):** Tham số ARIMA cho dữ liệu không có mùa vụ.
- **(P, D, Q, m):** Tham số mùa vụ:
  - **P:** Số bậc của thành phần tự hồi quy mùa vụ.
  - **D:** Số lần khác biệt hóa mùa vụ.
  - **Q:** Số bậc của thành phần trung bình trượt mùa vụ.
  - **m:** Chu kỳ mùa vụ (ví dụ: m = 12 nếu dữ liệu theo tháng).

### 2️⃣ **Lý do chọn SARIMA cho dự báo giá vàng**
- Dữ liệu giá vàng có yếu tố **chu kỳ hàng tháng** (m = 12).
- Mô hình SARIMA có thể dự báo chính xác hơn khi dữ liệu có xu hướng và mùa vụ.
- Có khả năng xử lý dữ liệu **không dừng** bằng cách kết hợp **khác biệt hóa (differencing)**.
- Dễ dàng điều chỉnh thông số để phù hợp với từng bộ dữ liệu cụ thể.

---

## 🛠 Các tính năng chính

### 1️⃣ **Xử lý dữ liệu**
- Chuyển đổi cột `Date` sang **định dạng thời gian**.
- Resample dữ liệu sang **giá trung bình theo tháng**.
- Loại bỏ dữ liệu bị thiếu, giữ lại các cột số để phân tích.

### 2️⃣ **Trực quan hóa dữ liệu**
- Vẽ biểu đồ **xu hướng giá vàng theo tháng**.
- Phân tích **thành phần xu hướng, mùa vụ** bằng phương pháp Seasonal Decomposition.

### 3️⃣ **Kiểm tra tính dừng của chuỗi**
- Sử dụng **Dickey-Fuller test** để xác định tính dừng.
- Áp dụng **biến đổi Box-Cox** để xử lý phương sai không ổn định.
- Dùng **khác biệt hóa (differencing)** để làm dừng chuỗi.

### 4️⃣ **Lựa chọn mô hình SARIMA tối ưu**
- Dùng **Grid Search** để thử nghiệm nhiều bộ tham số khác nhau.
- Chọn mô hình có **giá trị AIC nhỏ nhất**.

### 5️⃣ **Dự báo giá vàng**
- Dự báo **24 tháng tiếp theo** dựa trên mô hình SARIMA tối ưu.
- So sánh giá trị thực tế và giá trị dự báo trên biểu đồ.

---

## 📊 Trực quan hóa dữ liệu

### 📈 Xu hướng giá vàng theo tháng
Biểu đồ dưới đây thể hiện biến động giá vàng qua thời gian:  
![Xu hướng giá vàng](https://github.com/lengocthien02/lengocthien/blob/main/Figure_1.png?raw=true)

### 🔮 Dự báo giá vàng với SARIMA
Mô hình dự báo giá vàng trong 24 tháng tiếp theo dựa trên SARIMA:  
![Dự báo giá vàng](https://github.com/lengocthien02/lengocthien/blob/main/Figure_2.png?raw=true)

---

## 🚀 Cách sử dụng

### 📥 1. Clone repository về máy
```sh
git clone https://github.com/lengocthien02/gold-price-forecasting.git
cd gold-price-forecasting
```

### 📦 2. Cài đặt các thư viện cần thiết
```sh
pip install -r requirements.txt
```

### 🏃‍♂️ 3. Chạy script để dự báo
```sh
python forecast_gold_prices.py
```

---

## 📝 Đoạn mã quan trọng

```python
# Huấn luyện mô hình SARIMA với tham số tối ưu
model = sm.tsa.statespace.SARIMAX(df_month['Close_box'].dropna(),
                                  order=(1, 1, 1),
                                  seasonal_order=(1, 1, 1, 12)).fit()

# Dự báo 24 tháng tiếp theo
df_month2['forecast'] = invboxcox(model.predict(start=len(df_month), end=len(df_month2)-1), lmbda)
```

---

## 📬 Liên hệ
🔹 **GitHub:** [@lengocthien02](https://github.com/lengocthien02)  
🔹 **LinkedIn:** [Your LinkedIn](https://linkedin.com/in/yourprofile)  
🔹 **Email:** your-email@gmail.com  


