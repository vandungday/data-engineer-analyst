2. THÔNG TIN CƠ BẢN VỀ DỮ LIỆU:

---

Kích thước dữ liệu: (16534, 11)
Số dòng: 16534
Số cột: 11

3. TÓM LƯỢC DỮ LIỆU:

---

Thông tin các cột:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 16534 entries, 0 to 16533
Data columns (total 11 columns):

# Column Non-Null Count Dtype

---

0 work_year 16534 non-null int64
1 experience_level 16534 non-null object
2 employment_type 16534 non-null object
3 job_title 16534 non-null object
4 salary 16534 non-null int64
5 salary_currency 16534 non-null object
6 salary_in_usd 16534 non-null int64
7 employee_residence 16534 non-null object
8 remote_ratio 16534 non-null int64
9 company_location 16534 non-null object
10 company_size 16534 non-null object
dtypes: int64(4), object(7)
memory usage: 1.4+ MB
None

Thống kê mô tả:
work_year salary salary_in_usd remote_ratio
count 16534.000000 1.653400e+04 16534.000000 16534.000000
mean 2023.226866 1.637270e+05 149686.777973 32.003750
std 0.713558 3.402057e+05 68505.293156 46.245158
min 2020.000000 1.400000e+04 15000.000000 0.000000
25% 2023.000000 1.017630e+05 101125.000000 0.000000
50% 2023.000000 1.422000e+05 141300.000000 0.000000
75% 2024.000000 1.872000e+05 185900.000000 100.000000
max 2024.000000 3.040000e+07 800000.000000 100.000000

Kiểm tra giá trị thiếu:
Series([], dtype: int64)
Không có giá trị thiếu trong dữ liệu

4. PHÂN TÍCH JOB TITLE CHỨA TỪ 'DATA':

---

Top 20 job title chứa 'Data':
job_title
Data Engineer 3464
Data Scientist 3314
Data Analyst 2440
Data Architect 435
Data Science 271
Data Manager 212
Data Science Manager 122
Data Specialist 86
Data Science Consultant 83
Data Analytics Manager 62
Head of Data 61
Data Modeler 56
Data Product Manager 36
Director of Data Science 33
Data Developer 30
Data Science Engineer 29
Data Science Lead 26
Data Lead 26
Data Strategist 24
Data Operations Analyst 24
Name: count, dtype: int64

Các job xuất hiện > 30 lần (14 loại):
job_title
Data Engineer 3464
Data Scientist 3314
Data Analyst 2440
Data Architect 435
Data Science 271
Data Manager 212
Data Science Manager 122
Data Specialist 86
Data Science Consultant 83
Data Analytics Manager 62
Head of Data 61
Data Modeler 56
Data Product Manager 36
Director of Data Science 33
Name: count, dtype: int64

6. LÀM SẠCH DỮ LIỆU:

---

Các job chứa 'Data' xuất hiện > 30 lần: ['Data Engineer', 'Data Scientist', 'Data Analyst', 'Data Architect', 'Data Science', 'Data Manager', 'Data Science Manager', 'Data Specialist', 'Data Science Consultant', 'Data Analytics Manager', 'Head of Data', 'Data Modeler', 'Data Product Manager', 'Director of Data Science']
Kích thước dữ liệu sau khi lọc job titles: (10675, 11)
Min = 15000
Max = 774000
Median = 135000
Mean = 140563
Std = 60510
Giới hạn outliers: [-16420, 289852]
Đã loại bỏ 202 outliers
Kích thước dữ liệu cuối cùng: (10473, 11)

7. TIỀN XỬ LÝ DỮ LIỆU:

---

6. LÀM SẠCH DỮ LIỆU:

---

Các job chứa 'Data' xuất hiện > 30 lần: ['Data Engineer', 'Data Scientist', 'Data Analyst', 'Data Architect', 'Data Science', 'Data Manager', 'Data Science Manager', 'Data Specialist', 'Data Science Consultant', 'Data Analytics Manager', 'Head of Data', 'Data Modeler', 'Data Product Manager', 'Director of Data Science']
Kích thước dữ liệu sau khi lọc job titles: (10675, 11)
Min = 15000
Max = 774000
Median = 135000
Mean = 140563
Std = 60510
Giới hạn outliers: [-16420, 289852]
Đã loại bỏ 202 outliers
Kích thước dữ liệu cuối cùng: (10473, 11)
Thực hiện Label Encoding cho các biến categorical:

- experience_level: 4 categories
- employment_type: 4 categories
- job_title: 14 categories
- company_size: 3 categories

Số lượng features: 7
Features: ['work_year', 'experience_level_encoded', 'employment_type_encoded', 'job_title_encoded', 'remote_ratio', 'company_size_encoded', 'is_us']

Kích thước tập train: (8378, 7)
Kích thước tập test: (2095, 7)

8. CHUẨN HÓA DỮ LIỆU:

---

Sẽ thực hiện chuẩn hóa sau khi feature selection để đảm bảo consistency

9. TRÍCH CHỌN THUỘC TÍNH:

---

Top 5 features được chọn:

- work_year: 19.15
- experience_level_encoded: 986.34
- employment_type_encoded: 8.42
- job_title_encoded: 504.50
- is_us: 931.06

1. HUẤN LUYỆN MÔ HÌNH HỒI QUY TUYẾN TÍNH:

---

Sử dụng Z-Score Normalization với 5 features được chọn
Đã chuẩn hóa 5 features được chọn

## KẾT QUẢ MÔ HÌNH Z-SCORE NORMALIZATION

## Metric Train Test

R² Score 0.2284 0.2279  
MSE 2,234,139,149.60 2,205,724,096.38
RMSE 47,266.68 46,965.14  
MAE 37,489.36 37,530.46

---

2. PHÂN TÍCH CHI TIẾT MÔ HÌNH

---

Phương pháp chuẩn hóa: Z-Score Normalization
Intercept (β₀): $136,913.07

## HỆ SỐ HỒI QUY (βᵢ) - 5 FEATURES ĐƯỢC CHỌN

work_year : 2,302.03
experience_level_encoded : 14,195.57
employment_type_encoded : -64.46
job_title_encoded : 11,564.17
is_us : 15,215.21

## TẦM QUAN TRỌNG CỦA 5 FEATURES ĐƯỢC CHỌN

is_us : 35.11% ███████
experience_level_encoded : 32.75% ██████
job_title_encoded : 26.68% █████
work_year : 5.31% █
employment_type_encoded : 0.15%

## ĐÁNH GIÁ MÔ HÌNH

• R² Score (0.2279): Yếu - Mô hình cần cải thiện đáng kể
• Mô hình giải thích được 22.8% biến thiên của mức lương
• RMSE: $46,965.14 (34.4% của mức lương trung bình)

================================================================================
📊 DASHBOARD TỔNG KẾT KẾT QUẢ PHÂN TÍCH  
================================================================================

## 📋 THÔNG TIN DỮ LIỆU

• Tổng số mẫu ban đầu: 16,534
• Số mẫu sau xử lý: 10,473
• Số loại công việc: 14
• Khoảng lương: $15,000.00 - $288,400.00

## 🎯 HIỆU SUẤT MÔ HÌNH

• Độ chính xác (R²): 22.8%
• Sai số trung bình (MAE): $37,530.46
• Sai số bình phương trung bình (MSE): $2,205,724,096.38
• Sai số bình phương gốc (RMSE): $46,965.14

## 🔍 TOP 5 YẾU TỐ QUAN TRỌNG NHẤT (ĐÃ ĐƯỢC CHỌN)

1. is_us: 35.1%
2. experience_level_encoded: 32.8%
3. job_title_encoded: 26.7%
4. work_year: 5.3%
5. employment_type_encoded: 0.1%

================================================================================
🎉 HOÀN THÀNH PHÂN TÍCH!  
================================================================================
