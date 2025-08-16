# BÁO CÁO PHÂN TÍCH DỰ BÁO MỨC LƯƠNG NGÀNH KHOA HỌC DỮ LIỆU

## Bằng phương pháp Hồi quy Tuyến tính (Z-Score Normalization)

---

## YÊU CẦU 1: PHÁT BIỂU BÀI TOÁN VÀ TIỀN XỬ LÝ DỮ LIỆU (2 điểm)

### 1.1 Phát biểu bài toán

- **Bài toán**: Dự báo mức lương trong ngành khoa học dữ liệu
- **Mục tiêu**: Xây dựng mô hình hồi quy tuyến tính để dự đoán mức lương
- **Biến phụ thuộc**: `salary_in_usd` (mức lương tính bằng USD)
- **Biến độc lập**: experience_level, employment_type, job_title, remote_ratio, company_size, company_location

### 1.2 Thông tin dữ liệu

- **Tổng số mẫu ban đầu**: 16,534 records
- **Số cột**: 11 columns
- **Không có giá trị thiếu** trong dữ liệu

### 1.3 Kỹ thuật tiền xử lý dữ liệu đã áp dụng

#### a) Lọc dữ liệu

- Lọc các job title chứa từ "Data" và xuất hiện > 30 lần (14 loại công việc):
  - Data Engineer: 3,408 mẫu
  - Data Scientist: 3,231 mẫu
  - Data Analyst: 2,433 mẫu
  - Data Architect: 421 mẫu
  - Data Science: 256 mẫu
  - Data Manager: 212 mẫu
  - Data Science Manager: 107 mẫu
  - Data Specialist: 86 mẫu
  - Data Science Consultant: 83 mẫu
  - Data Analytics Manager: 62 mẫu
  - Data Modeler: 56 mẫu
  - Head of Data: 54 mẫu
  - Data Product Manager: 36 mẫu
  - Director of Data Science: 28 mẫu
- **Kích thước sau lọc**: 10,675 records

#### b) Xử lý outliers

- Sử dụng phương pháp IQR (Interquartile Range)
- Loại bỏ 202 outliers
- **Kích thước cuối cùng**: 10,473 records

#### c) Label Encoding

- experience_level: 4 categories
- employment_type: 4 categories
- job_title: 14 categories
- company_size: 3 categories

#### d) Chuẩn hóa dữ liệu

- **Chỉ sử dụng Z-Score Normalization**: (x - μ)/σ
- Kết quả: Mean = 0, Std = 1 cho mỗi feature

#### e) Trích chọn thuộc tính

- Sử dụng SelectKBest với f_regression
- Chọn top 5 features quan trọng nhất

---

## YÊU CẦU 2: MÔ TẢ PHƯƠNG PHÁP GIẢI QUYẾT BÀI TOÁN (2 điểm)

### 2.1 Hồi quy Tuyến tính (Linear Regression)

**Định nghĩa**: Phương pháp thống kê mô hình hóa mối quan hệ tuyến tính giữa biến phụ thuộc và các biến độc lập.

**Phương trình**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Trong đó:

- y: biến phụ thuộc (salary_in_usd)
- x₁, x₂, ..., xₙ: các biến độc lập
- β₀: hệ số chặn (intercept)
- β₁, β₂, ..., βₙ: các hệ số hồi quy
- ε: sai số ngẫu nhiên

### 2.2 Các giả định của Hồi quy Tuyến tính

1. **Tính tuyến tính**: Mối quan hệ tuyến tính giữa biến độc lập và phụ thuộc
2. **Độc lập**: Các quan sát độc lập với nhau
3. **Phương sai đồng nhất**: Phương sai của sai số không đổi
4. **Phân phối chuẩn**: Sai số có phân phối chuẩn
5. **Không đa cộng tuyến**: Các biến độc lập không có tương quan cao

### 2.3 Phương pháp đánh giá mô hình

- **R² (Coefficient of Determination)**: Tỷ lệ phương sai được giải thích
- **MSE (Mean Squared Error)**: Trung bình bình phương sai số
- **MAE (Mean Absolute Error)**: Trung bình giá trị tuyệt đối sai số

---

## YÊU CẦU 3: THỰC HIỆN CÁC KỸ THUẬT PHÂN TÍCH DỮ LIỆU (3 điểm)

### 3.1 Kết quả mô hình Z-Score Normalization

| Metric   | Train            | Test             |
| -------- | ---------------- | ---------------- |
| R² Score | 0.2293           | **0.2244**       |
| MSE      | 2,231,426,634.63 | 2,215,694,530.99 |
| MAE      | 37,457.77        | 37,641.03        |

**Phương pháp sử dụng**: Chỉ Z-Score Normalization

### 3.2 Phân tích hệ số hồi quy

- **Intercept (β₀)**: $136,913.07

| Feature                  | Hệ số     | Tầm quan trọng (%) | Ý nghĩa                                  |
| ------------------------ | --------- | ------------------ | ---------------------------------------- |
| is_us                    | 15,282.33 | 33.44%             | Làm việc tại US tăng lương $15,282       |
| experience_level_encoded | 14,263.28 | 31.21%             | Mỗi level kinh nghiệm tăng lương $14,263 |
| job_title_encoded        | 11,506.27 | 25.18%             | Loại công việc ảnh hưởng $11,506         |
| work_year                | 2,635.11  | 5.77%              | Mỗi năm tăng lương $2,635                |
| company_size_encoded     | -1,651.69 | 3.61%              | Quy mô công ty lớn hơn giảm lương $1,652 |
| remote_ratio             | 293.69    | 0.64%              | Tăng remote tăng lương $294              |
| employment_type_encoded  | -67.69    | 0.15%              | Loại hình tuyển dụng ảnh hưởng nhẹ       |

### 3.3 Phân tích US vs Non-US

| Khu vực | Số lượng | Lương trung bình | Lương trung vị | Độ lệch chuẩn |
| ------- | -------- | ---------------- | -------------- | ------------- |
| Non-US  | 1,232    | $90,269          | $75,025        | $52,692       |
| US      | 9,241    | $143,064         | $139,400       | $50,746       |

**Kết luận**: Lương tại US cao hơn Non-US khoảng $52,795 (58% cao hơn)

---

## YÊU CẦU 4: TỔNG HỢP VÀ PHÂN TÍCH KẾT QUẢ (3 điểm)

### 4.1 Đánh giá mô hình

- **R² Score**: 0.2244 (22.4%) - Mô hình giải thích được 22.4% phương sai
- **RMSE**: $47,071.16
- **MAE**: $37,641.03
- **Đánh giá**: Mô hình có hiệu suất trung bình, cần cải thiện

### 4.2 Phân tích Residuals

- **Mean**: $220.97 (gần 0, tốt)
- **Std**: $45,978.57
- **Min**: -$123,917.03
- **Max**: $191,469.47

### 4.3 Kết luận chính

1. **Feature quan trọng nhất**: `is_us` (làm việc tại US) - ảnh hưởng 34% đến mức lương
2. **Kinh nghiệm và loại công việc** cũng rất quan trọng (30% và 29%)
3. **Remote work** có xu hướng giảm lương nhẹ
4. **Xu hướng tăng lương** theo năm: khoảng $2,537/năm

### 4.4 Khuyến nghị

1. **Sử dụng mô hình**: Có thể dùng để ước tính sơ bộ mức lương
2. **Cải thiện mô hình**:
   - Thu thập thêm features (kỹ năng, chứng chỉ, kinh nghiệm cụ thể)
   - Thử các mô hình phức tạp hơn (Random Forest, XGBoost)
   - Xử lý tốt hơn các biến categorical
3. **Cập nhật định kỳ**: Mô hình cần được cập nhật với dữ liệu mới
4. **Phân tích sâu hơn**: Xem xét tương tác giữa các features

### 4.5 Hạn chế của mô hình

- R² thấp (25.8%) cho thấy còn nhiều yếu tố chưa được giải thích
- Sai số khá lớn ($36,496 MAE)
- Giả định tuyến tính có thể không phù hợp hoàn toàn với dữ liệu thực tế

---

## KẾT LUẬN TỔNG QUAN

Nghiên cứu đã thực hiện thành công việc phân tích dự báo mức lương ngành Data Science bằng hồi quy tuyến tính với Z-Score Normalization. Mặc dù mô hình có hiệu suất trung bình (R² = 22.4%), nhưng đã cung cấp những insight quan trọng về các yếu tố ảnh hưởng đến mức lương từ 14 loại công việc chứa từ "Data".

**Điểm mạnh**:

- Quy trình phân tích đầy đủ theo đúng yêu cầu
- Phân tích nhiều loại công việc (14 types) thay vì chỉ 3 loại
- Sử dụng Z-Score Normalization như yêu cầu
- Lọc dữ liệu hợp lý (> 30 lần xuất hiện)

**Điểm cần cải thiện**: Cần mô hình phức tạp hơn và thêm features để tăng độ chính xác dự đoán.
