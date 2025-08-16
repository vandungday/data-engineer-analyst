# PHÂN TÍCH CHI TIẾT CÁC BIỂU ĐỒ

## Dự án: Phân tích và Dự báo Mức lương Ngành Data Science

---

## 📊 TỔNG QUAN

Dự án tạo ra **11 biểu đồ chất lượng cao** (300 DPI) phân tích toàn diện mức lương trong ngành Data Science từ dataset 16,534 records, sau khi làm sạch còn 10,473 records chất lượng cao.

---

## 1️⃣ **01a_data_distribution.png** - Phân phối Dữ liệu Cơ bản

### 📈 **Mô tả:**

Biểu đồ gồm 3 subplot ngang hiển thị phân phối cơ bản của dữ liệu đã được làm sạch.

### 🔍 **Chi tiết từng subplot:**

**Subplot 1: Phân phối mức lương**

- **Loại:** Histogram với 50 bins
- **Đặc điểm:** Phân phối lệch phải (right-skewed)
- **Tập trung:** Phần lớn mức lương trong khoảng $75K-$200K
- **Đỉnh:** Khoảng $125K-$150K
- **Format:** Trục Y hiển thị tần suất, trục X format $XXXk

**Subplot 2: Mức lương theo Experience Level**

- **Loại:** Box plot
- **Các nhóm:** EN (Entry), MI (Mid), SE (Senior), EX (Executive)
- **Xu hướng:** Tăng dần từ Entry đến Executive
- **Outliers:** Có một số outliers ở tất cả các level
- **Median:** EN ~$80K, MI ~$120K, SE ~$160K, EX ~$200K

**Subplot 3: Mức lương theo Company Size**

- **Loại:** Box plot
- **Các nhóm:** S (Small), M (Medium), L (Large)
- **Xu hướng:** Medium companies trả lương cao nhất
- **Thứ tự:** M > L > S
- **Đặc điểm:** Large companies có phạm vi lương rộng nhất

---

## 2️⃣ **01b_data_analysis.png** - Phân tích Chi tiết

### 📈 **Mô tả:**

Biểu đồ gồm 3 subplot ngang cung cấp phân tích sâu hơn về xu hướng và so sánh.

### 🔍 **Chi tiết từng subplot:**

**Subplot 1: Top 10 Job Titles theo lương TB**

- **Loại:** Horizontal bar chart
- **Thứ tự:** Từ thấp đến cao (ascending)
- **Top 3:** Data Science Manager, Data Architect, Director of Data Science
- **Đặc điểm:** Management roles có lương cao hơn technical roles
- **Format:** Hiển thị giá trị $XXXk bên cạnh mỗi bar

**Subplot 2: Xu hướng lương theo năm** ⭐ **ĐÃ ĐƯỢC CẢI THIỆN**

- **Loại:** Line plot với markers lớn
- **Xu hướng:** Tăng mạnh từ 2020-2022, ổn định 2022-2024
- **Giá trị:** 2020: $77K → 2024: $141K
- **Đặc điểm:** Tăng trưởng 83% trong 4 năm
- **Cải thiện:** Trục X hiển thị năm rõ ràng, không còn số thập phân

**Subplot 3: So sánh US vs Non-US**

- **Loại:** Bar chart với 2 cột
- **Chênh lệch:** US cao hơn Non-US đáng kể
- **Giá trị:** US: $143K, Non-US: $90K
- **Tỷ lệ:** US cao hơn 59%
- **Highlight:** Có text box hiển thị chênh lệch và phần trăm

---

## 3️⃣ **02_model_results.png** - Kết quả Mô hình

### 📈 **Mô tả:**

Biểu đồ 2x2 (4 subplots) hiển thị kết quả và đánh giá mô hình Linear Regression.

### 🔍 **Chi tiết từng subplot:**

**Subplot 1: Actual vs Predicted (Train)**

- **Loại:** Scatter plot với đường diagonal
- **Đánh giá:** R² = 0.2293
- **Đặc điểm:** Điểm phân tán quanh đường y=x
- **Nhận xét:** Mô hình có độ chính xác khiêm tốn

**Subplot 2: Actual vs Predicted (Test)**

- **Loại:** Scatter plot với đường diagonal
- **Đánh giá:** R² = 0.2244
- **Đặc điểm:** Tương tự train set, không overfitting
- **Nhận xét:** Generalization tốt

**Subplot 3: Residuals vs Predicted**

- **Loại:** Scatter plot
- **Đánh giá:** Kiểm tra homoscedasticity
- **Đặc điểm:** Residuals phân tán ngẫu nhiên
- **Nhận xét:** Không có pattern rõ ràng (tốt)

**Subplot 4: Residuals Distribution**

- **Loại:** Histogram
- **Đánh giá:** Kiểm tra tính chuẩn của residuals
- **Đặc điểm:** Gần như phân phối chuẩn
- **Nhận xét:** Đáp ứng giả định của Linear Regression

---

## 4️⃣ **03_correlation_matrix.png** - Ma trận Tương quan

### 📈 **Mô tả:**

Heatmap hiển thị mối tương quan giữa các features số.

### 🔍 **Chi tiết:**

- **Màu sắc:** Từ xanh dương (âm) đến đỏ (dương)
- **Giá trị:** Hiển thị hệ số tương quan (-1 đến 1)
- **Đường chéo:** Giá trị 1.0 (tự tương quan)
- **Insights:**
  - `is_us` có tương quan mạnh với `salary_in_usd`
  - `experience_level_encoded` tương quan dương với lương
  - Không có multicollinearity nghiêm trọng

---

## 5️⃣ **04_salary_by_job.png** - Lương theo Job Title

### 📈 **Mô tả:**

Biểu đồ 2 subplots hiển thị phân phối lương theo từng job title.

### 🔍 **Chi tiết:**

**Subplot 1: Box plots theo Job Title**

- **Loại:** Multiple box plots
- **Đặc điểm:** 14 job titles được phân tích
- **Xu hướng:** Management roles có median cao hơn
- **Outliers:** Mỗi job đều có outliers

**Subplot 2: Violin plots theo Job Title**

- **Loại:** Violin plots (density + box plot)
- **Đặc điểm:** Hiển thị phân phối density của từng job
- **Insights:** Một số jobs có phân phối bimodal
- **Ưu điểm:** Thấy được shape của distribution

---

## 6️⃣ **05_experience_impact.png** - Tác động Kinh nghiệm

### 📈 **Mô tả:**

Biểu đồ 2 subplots ngang phân tích tác động của experience level.

### 🔍 **Chi tiết:**

**Subplot 1: Count plot theo Experience Level**

- **Loại:** Bar chart
- **Thứ tự:** SE > MI > EN > EX
- **Insights:** Senior level chiếm đa số trong dataset
- **Đặc điểm:** Executive level ít nhất (scarcity premium)

**Subplot 2: Salary distribution theo Experience Level**

- **Loại:** Box plot với màu sắc khác nhau
- **Xu hướng:** Tăng dần EN → MI → SE → EX
- **Chênh lệch:** EX cao hơn EN khoảng 150%
- **Đặc điểm:** EX có variance cao nhất

---

## 7️⃣ **06_company_size_salary.png** - Company Size vs Salary

### 📈 **Mô tả:**

Box plot đơn giản hiển thị mức lương theo quy mô công ty.

### 🔍 **Chi tiết:**

- **Loại:** Box plot với 3 categories
- **Thứ tự lương:** M (Medium) > L (Large) > S (Small)
- **Insights:** Medium companies trả lương cao nhất
- **Giải thích:** Medium có flexibility + resources tốt
- **Variance:** Large companies có phạm vi lương rộng nhất

---

## 8️⃣ **07_remote_impact.png** - Tác động Remote Work

### 📈 **Mô tả:**

Biểu đồ 2 subplots ngang phân tích tác động của remote work. ⭐ **ĐÃ ĐƯỢC CẢI THIỆN**

### 🔍 **Chi tiết:**

**Subplot 1: Box plot theo Remote Groups**

- **Loại:** Box plot với 3 nhóm
- **Nhóm:** On-site (0-25%), Hybrid (25-75%), Remote (75-100%)
- **Xu hướng:** Remote > Hybrid > On-site
- **Insights:** Full remote có lương cao nhất

**Subplot 2: Scatter plot Remote Ratio vs Salary**

- **Loại:** Scatter plot với grid
- **Cải thiện:** Trục X hiển thị % rõ ràng (0%, 25%, 50%, 75%, 100%)
- **Pattern:** Xu hướng tăng nhẹ theo remote ratio
- **Đặc điểm:** Có grid giúp đọc dễ hơn

---

## 9️⃣ **08_residuals_analysis.png** - Phân tích Residuals

### 📈 **Mô tả:**

Biểu đồ 2x2 (4 subplots) phân tích chi tiết residuals của mô hình.

### 🔍 **Chi tiết từng subplot:**

**Subplot 1: Residuals vs Fitted Values**

- **Loại:** Scatter plot
- **Mục đích:** Kiểm tra homoscedasticity
- **Kết quả:** Residuals phân tán ngẫu nhiên
- **Đánh giá:** Đạt yêu cầu (không có pattern)

**Subplot 2: Q-Q Plot**

- **Loại:** Quantile-Quantile plot
- **Mục đích:** Kiểm tra tính chuẩn của residuals
- **Kết quả:** Gần như thẳng hàng với đường diagonal
- **Đánh giá:** Residuals có phân phối gần chuẩn

**Subplot 3: Scale-Location Plot**

- **Loại:** Sqrt(|Residuals|) vs Fitted
- **Mục đích:** Kiểm tra homoscedasticity
- **Kết quả:** Đường trend tương đối phẳng
- **Đánh giá:** Variance ổn định

**Subplot 4: Residuals vs Leverage**

- **Loại:** Scatter plot với Cook's distance
- **Mục đích:** Tìm influential points
- **Kết quả:** Một số điểm có leverage cao
- **Đánh giá:** Không có outliers nghiêm trọng

---

## 🔟 **09_feature_importance.png** - Tầm quan trọng Features

### 📈 **Mô tả:**

Biểu đồ 2 subplots hiển thị tầm quan trọng của các features.

### 🔍 **Chi tiết:**

**Subplot 1: Feature Coefficients**

- **Loại:** Horizontal bar chart
- **Thứ tự:** Theo giá trị tuyệt đối của coefficients
- **Top 3:** is_us, experience_level_encoded, job_title_encoded
- **Insights:** Location (US) là factor quan trọng nhất

**Subplot 2: Feature Importance Percentage**

- **Loại:** Pie chart
- **Phân bố:** is_us (33.4%), experience_level (31.2%), job_title (25.2%)
- **Insights:** 3 features này chiếm gần 90% tầm quan trọng
- **Màu sắc:** Mỗi feature có màu riêng biệt

---

## 1️⃣1️⃣ **10_salary_predictions.png** - Dự báo Salary

### 📈 **Mô tả:**

Biểu đồ 2x2 (4 subplots) hiển thị kết quả dự báo cho các scenarios.

### 🔍 **Chi tiết từng subplot:**

**Subplot 1: Prediction Scenarios**

- **Loại:** Bar chart với error bars
- **Scenarios:** 5 kịch bản khác nhau
- **Error bars:** Confidence intervals 95%
- **Insights:** US positions có lương cao hơn đáng kể

**Subplot 2: Salary Range by Experience**

- **Loại:** Bar chart với ranges
- **Đặc điểm:** Hiển thị min-max range cho mỗi level
- **Xu hướng:** Range tăng theo experience level
- **Insights:** Executive có variance cao nhất

**Subplot 3: Geographic Comparison**

- **Loại:** Grouped bar chart
- **So sánh:** US vs Non-US cho từng job type
- **Pattern:** US luôn cao hơn Non-US
- **Chênh lệch:** Từ 40-70% tùy job type

**Subplot 4: Prediction Confidence**

- **Loại:** Error bar plot
- **Mục đích:** Hiển thị độ tin cậy của predictions
- **Đặc điểm:** Confidence intervals rộng hơn cho extreme values
- **Insights:** Model ít confident với outlier cases

---

## 📋 **TỔNG KẾT PHÂN TÍCH**

### 🎯 **Insights Chính:**

1. **Location Impact:** US positions trả lương cao hơn 59% so với Non-US
2. **Experience Premium:** Executive level cao hơn Entry level 150%
3. **Company Size:** Medium companies trả lương cao nhất
4. **Remote Advantage:** Full remote có lương cao hơn on-site
5. **Job Type:** Management roles > Technical roles
6. **Time Trend:** Lương tăng 83% từ 2020-2024

### 📊 **Chất lượng Biểu đồ:**

- **Resolution:** 300 DPI (print quality)
- **Format:** PNG với nền trắng
- **Typography:** Font rõ ràng, size phù hợp
- **Color Scheme:** Consistent và professional
- **Layout:** Balanced và easy-to-read

### 🔧 **Cải thiện Đã thực hiện:**

- **Xu hướng năm:** Trục X hiển thị năm rõ ràng
- **Remote impact:** Grid và % labels
- **Format số:** $XXXk thay vì số dài
- **Confidence intervals:** Hiển thị độ tin cậy predictions

### 💡 **Khuyến nghị Sử dụng:**

1. **Báo cáo:** Suitable cho presentations và reports
2. **Decision making:** Support salary negotiations
3. **Career planning:** Guide career development
4. **HR Analytics:** Benchmark compensation packages
