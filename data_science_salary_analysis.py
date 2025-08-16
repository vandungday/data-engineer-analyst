import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import os
from scipy import stats
warnings.filterwarnings('ignore')

# Thiết lập font hỗ trợ tiếng Việt và style cho biểu đồ
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

print("Sử dụng font: DejaVu Sans")

# Thêm cấu hình để hỗ trợ tiếng Việt tốt hơn
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.formatter.use_locale'] = True

# Sử dụng style an toàn hơn
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")

# Thiết lập backend an toàn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

class DataScienceSalaryAnalysis:
    def __init__(self, data_path):
        """
        Khởi tạo với đường dẫn dữ liệu
        
        Args:
            data_path (str): Đường dẫn đến file dữ liệu
            df (DataFrame): Dữ liệu gốc
            df_processed (DataFrame): Dữ liệu đã xử lý
            scaler (StandardScaler): Đối tượng chuẩn hóa dữ liệu
            label_encoders (dict): Dictionary chứa các LabelEncoder cho các cột categorical
            model (LinearRegression): Đối tượng mô hình hồi quy tuyến tính
            X_train, X_test, y_train, y_test (DataFrame/Series): Dữ liệu huấn luyện và kiểm tra
        """
        self.data_path = data_path 
        self.df = None 
        self.df_processed = None 
        self.scaler = None 
        self.label_encoders = {}
        self.model = None 
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"\n2. THÔNG TIN CƠ BẢN VỀ DỮ LIỆU:")
        print("-" * 30)
        print(f"Kích thước dữ liệu: {self.df.shape}")
        print(f"Số dòng: {self.df.shape[0]}")
        print(f"Số cột: {self.df.shape[1]}")
        
        return self.df
    
    def explore_data(self):
        """
        Khám phá và tóm lược dữ liệu
        """
        print("\n3. TÓM LƯỢC DỮ LIỆU:")
        print("-" * 30)
        
        # Hiển thị thông tin cơ bản
        print("Thông tin các cột:")
        print(self.df.info())
        
        print("\nThống kê mô tả:")
        print(self.df.describe())
        
        print("\nKiểm tra giá trị thiếu:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        if missing_values.sum() == 0:
            print("Không có giá trị thiếu trong dữ liệu")
        
        # Phân tích các job title chứa từ "data"
        print("\n4. PHÂN TÍCH JOB TITLE CHỨA TỪ 'DATA':")
        print("-" * 30)
        data_jobs = self.df[self.df['job_title'].str.contains('Data', case=False, na=False)]
        job_counts = data_jobs['job_title'].value_counts()

        print("Top 20 job title chứa 'Data':")
        print(job_counts.head(20))

        # Lọc các job xuất hiện > 30 lần
        frequent_jobs = job_counts[job_counts > 30]
        print(f"\nCác job xuất hiện > 30 lần ({len(frequent_jobs)} loại):")
        print(frequent_jobs)
        
        return data_jobs, frequent_jobs
    
    def visualize_data(self):
        print("\n5. BIỂU ĐỒ KHÁM PHÁ DỮ LIỆU (DỮ LIỆU ĐÃ XỬ LÝ):")
        print("-" * 30)

        # Kiểm tra xem đã có dữ liệu processed chưa
        if not hasattr(self, 'df_processed') or self.df_processed is None:
            print("⚠️  Chưa có dữ liệu đã xử lý. Sử dụng dữ liệu gốc.")
            data = self.df
        else:
            print(f"✅ Sử dụng dữ liệu đã xử lý: {self.df_processed.shape[0]} records")
            data = self.df_processed

        # Hàm format số ngắn gọn
        def format_currency(x, pos):
            if x >= 1e6:
                return f'${x/1e6:.1f}M'
            elif x >= 1e3:
                return f'${x/1e3:.0f}K'
            else:
                return f'${x:.0f}'

        # PHẦN 1: Phân phối và so sánh cơ bản (3 biểu đồ)
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
        fig1.suptitle('Data Science Salary Analysis - Phần 1: Phân phối dữ liệu', fontsize=16, fontweight='bold')

        # 1. Histogram của salary
        axes1[0].hist(data['salary_in_usd'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes1[0].set_title('Phân phối mức lương')
        axes1[0].set_xlabel('Mức lương (USD)')
        axes1[0].set_ylabel('Tần suất')
        axes1[0].grid(True, alpha=0.3)
        axes1[0].xaxis.set_major_formatter(plt.FuncFormatter(format_currency))

        # 2. Box plot salary theo experience level
        sns.boxplot(data=data, x='experience_level', y='salary_in_usd', ax=axes1[1])
        axes1[1].set_title('Mức lương theo Experience Level')
        axes1[1].set_xlabel('Experience Level')
        axes1[1].set_ylabel('Mức lương (USD)')
        axes1[1].yaxis.set_major_formatter(plt.FuncFormatter(format_currency))

        # 3. Box plot salary theo company size
        sns.boxplot(data=data, x='company_size', y='salary_in_usd', ax=axes1[2])
        axes1[2].set_title('Mức lương theo Company Size')
        axes1[2].set_xlabel('Company Size')
        axes1[2].set_ylabel('Mức lương (USD)')
        axes1[2].yaxis.set_major_formatter(plt.FuncFormatter(format_currency))

        plt.tight_layout()
        plt.savefig('images/01a_data_distribution.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/01a_data_distribution.png")
        plt.close()

        # PHẦN 2: Phân tích chi tiết (3 biểu đồ)
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        fig2.suptitle('Data Science Salary Analysis - Phần 2: Phân tích chi tiết', fontsize=16, fontweight='bold')

        # 4. Top 10 job titles theo average salary
        top_jobs = data.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=True).tail(10)
        bars = axes2[0].barh(range(len(top_jobs)), top_jobs.values, color='lightcoral')
        axes2[0].set_title('Top 10 Job Titles theo lương TB')
        axes2[0].set_xlabel('Lương trung bình (USD)')
        axes2[0].set_ylabel('Job Title')
        axes2[0].set_yticks(range(len(top_jobs)))
        axes2[0].set_yticklabels(top_jobs.index)
        axes2[0].xaxis.set_major_formatter(plt.FuncFormatter(format_currency))

        # Thêm giá trị lên các cột
        for i, (bar, value) in enumerate(zip(bars, top_jobs.values)):
            axes2[0].text(value + 2000, i, f'${value/1000:.0f}K',
                         va='center', fontsize=9)

        # 5. Salary trend theo năm
        yearly_salary = data.groupby('work_year')['salary_in_usd'].mean()
        axes2[1].plot(yearly_salary.index, yearly_salary.values, marker='o', linewidth=3, markersize=10, color='green')
        axes2[1].set_title('Xu hướng lương theo năm')
        axes2[1].set_xlabel('Năm')
        axes2[1].set_ylabel('Lương trung bình (USD)')
        axes2[1].grid(True, alpha=0.3)
        axes2[1].yaxis.set_major_formatter(plt.FuncFormatter(format_currency))

        # Cải thiện hiển thị trục x
        axes2[1].set_xticks(yearly_salary.index)
        axes2[1].set_xticklabels([str(int(year)) for year in yearly_salary.index], fontsize=10)

        # Thêm giá trị lên các điểm
        for year, salary in yearly_salary.items():
            axes2[1].text(year, salary + 3000, f'${salary/1000:.0f}K',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 6. US vs Non-US salary comparison
        if 'is_us' not in data.columns:
            data = data.copy()
            data['is_us'] = data['company_location'] == 'US'
        us_comparison = data.groupby('is_us')['salary_in_usd'].mean()
        us_comparison.index = ['Non-US', 'US']

        bars = axes2[2].bar(us_comparison.index, us_comparison.values, color=['orange', 'lightblue'], alpha=0.8)
        axes2[2].set_title('So sánh lương: US vs Non-US')
        axes2[2].set_xlabel('Khu vực')
        axes2[2].set_ylabel('Lương trung bình (USD)')
        axes2[2].yaxis.set_major_formatter(plt.FuncFormatter(format_currency))

        # Thêm giá trị lên các cột
        for bar, value in zip(bars, us_comparison.values):
            axes2[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
                         f'${value/1000:.0f}K', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Thêm chênh lệch
        diff = us_comparison['US'] - us_comparison['Non-US']
        pct = (diff / us_comparison['Non-US']) * 100
        axes2[2].text(0.5, max(us_comparison.values) * 0.7,
                     f'Chênh lệch: ${diff/1000:.0f}K\n({pct:.1f}% cao hơn)',
                     ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.tight_layout()
        plt.savefig('images/01b_data_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/01b_data_analysis.png")
        plt.close()

        # In thống kê so sánh US vs Non-US
        print("\nSo sánh mức lương US vs Non-US:")
        us_stats = data.groupby('is_us')['salary_in_usd'].agg(['count', 'mean', 'median', 'std'])
        us_stats.index = ['Non-US', 'US']
        print(us_stats)

    def clean_data(self):
        """
        Làm sạch dữ liệu và lọc các job chứa từ "data" xuất hiện > 30 lần
        """
        print("\n6. LÀM SẠCH DỮ LIỆU:")
        print("-" * 30)

        # Tạo bản sao để xử lý
        self.df_processed = self.df.copy()

        # Lọc các job chứa từ "data" và xuất hiện > 30 lần
        data_jobs = self.df_processed[self.df_processed['job_title'].str.contains('Data', case=False, na=False)]
        job_counts = data_jobs['job_title'].value_counts()
        frequent_jobs = job_counts[job_counts > 30].index.tolist()

        print(f"Các job chứa 'Data' xuất hiện > 30 lần: {frequent_jobs}")

        # Lọc dữ liệu chỉ giữ lại các job frequent
        self.df_processed = self.df_processed[
            self.df_processed['job_title'].isin(frequent_jobs)
        ]

        print(f"Kích thước dữ liệu sau khi lọc job titles: {self.df_processed.shape}")

        # Loại bỏ các outliers (sử dụng IQR method)
        Q1 = self.df_processed['salary_in_usd'].quantile(0.25)
        Q3 = self.df_processed['salary_in_usd'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"Giới hạn outliers: [{lower_bound:.0f}, {upper_bound:.0f}]")

        # Loại bỏ outliers
        outliers_count = len(self.df_processed[
            (self.df_processed['salary_in_usd'] < lower_bound) |
            (self.df_processed['salary_in_usd'] > upper_bound)
        ])

        self.df_processed = self.df_processed[
            (self.df_processed['salary_in_usd'] >= lower_bound) &
            (self.df_processed['salary_in_usd'] <= upper_bound)
        ]

        print(f"Đã loại bỏ {outliers_count} outliers")
        print(f"Kích thước dữ liệu cuối cùng: {self.df_processed.shape}")

        # Thêm cột phân loại US vs Non-US
        self.df_processed['is_us'] = (self.df_processed['company_location'] == 'US').astype(int)

        # Tạo các cột encoded tạm thời để sử dụng trong visualization
        from sklearn.preprocessing import LabelEncoder
        categorical_columns = ['experience_level', 'employment_type', 'job_title', 'company_size']

        for col in categorical_columns:
            le = LabelEncoder()
            self.df_processed[col + '_encoded'] = le.fit_transform(self.df_processed[col])

        return self.df_processed

    def preprocess_data(self):
        """
        Tiền xử lý dữ liệu: Encoding, Normalization, Feature Selection
        """
        print("\n7. TIỀN XỬ LÝ DỮ LIỆU:")
        print("-" * 30)

        # Tạo bản sao để xử lý
        df_prep = self.df_processed.copy()

        # 1. Label Encoding cho các biến categorical
        categorical_columns = ['experience_level', 'employment_type', 'job_title', 'company_size']

        print("Thực hiện Label Encoding cho các biến categorical:")
        for col in categorical_columns:
            le = LabelEncoder()
            df_prep[col + '_encoded'] = le.fit_transform(df_prep[col])
            self.label_encoders[col] = le
            print(f"- {col}: {len(le.classes_)} categories")

        # 2. Chọn features cho mô hình
        feature_columns = [
            'work_year', 'experience_level_encoded', 'employment_type_encoded',
            'job_title_encoded', 'remote_ratio', 'company_size_encoded', 'is_us'
        ]

        X = df_prep[feature_columns]
        y = df_prep['salary_in_usd']

        print(f"\nSố lượng features: {len(feature_columns)}")
        print(f"Features: {feature_columns}")

        # 3. Chia dữ liệu train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nKích thước tập train: {self.X_train.shape}")
        print(f"Kích thước tập test: {self.X_test.shape}")

        # 4. Chuẩn hóa dữ liệu (chỉ sử dụng Z-Score Normalization)
        print("\n8. CHUẨN HÓA DỮ LIỆU:")
        print("-" * 30)

        # Chỉ sử dụng StandardScaler (Z-Score Normalization)
        self.scaler = StandardScaler()
        X_train_normalized = self.scaler.fit_transform(self.X_train)
        X_test_normalized = self.scaler.transform(self.X_test)

        # Lưu trữ dữ liệu đã chuẩn hóa
        self.X_train_normalized = X_train_normalized
        self.X_test_normalized = X_test_normalized

        print("Đã thực hiện chuẩn hóa dữ liệu bằng Z-Score Normalization:")
        print("- Công thức: (x - μ)/σ")
        print("- Kết quả: Mean = 0, Std = 1 cho mỗi feature")

        # 5. Feature Selection
        print("\n9. TRÍCH CHỌN THUỘC TÍNH:")
        print("-" * 30)

        # Sử dụng SelectKBest với f_regression
        selector = SelectKBest(score_func=f_regression, k=5)  # Chọn 5 features tốt nhất
        X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = selector.transform(self.X_test)

        # Lấy tên các features được chọn
        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        feature_scores = selector.scores_[selector.get_support()]

        print("Top 5 features được chọn:")
        for feature, score in zip(selected_features, feature_scores):
            print(f"- {feature}: {score:.2f}")

        self.selected_features = selected_features
        self.feature_selector = selector

        return {
            'scaler': self.scaler,
            'X_train_normalized': self.X_train_normalized,
            'X_test_normalized': self.X_test_normalized,
            'selected_features': self.selected_features
        }

    def describe_methods(self):
        """
        YÊU CẦU 2: Mô tả chi tiết các phương pháp giải quyết bài toán
        """
        print("\n" + "=" * 60)
        print("YÊU CẦU 2: MÔ TẢ PHƯƠNG PHÁP GIẢI QUYẾT BÀI TOÁN")
        print("=" * 60)

        print("\n1. HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION):")
        print("-" * 40)
        print("Định nghĩa:")
        print("- Hồi quy tuyến tính là phương pháp thống kê để mô hình hóa mối quan hệ")
        print("  tuyến tính giữa biến phụ thuộc (y) và một hoặc nhiều biến độc lập (X)")
        print("- Phương trình: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε")
        print("  Trong đó:")
        print("  + y: biến phụ thuộc (salary_in_usd)")
        print("  + x₁, x₂, ..., xₙ: các biến độc lập")
        print("  + β₀: hệ số chặn (intercept)")
        print("  + β₁, β₂, ..., βₙ: các hệ số hồi quy")
        print("  + ε: sai số ngẫu nhiên")

        print("\n2. CÁC GIẢ ĐỊNH CỦA HỒI QUY TUYẾN TÍNH:")
        print("-" * 40)
        print("- Tính tuyến tính: Mối quan hệ giữa biến độc lập và phụ thuộc là tuyến tính")
        print("- Độc lập: Các quan sát độc lập với nhau")
        print("- Phương sai đồng nhất: Phương sai của sai số không đổi")
        print("- Phân phối chuẩn: Sai số có phân phối chuẩn")
        print("- Không có đa cộng tuyến: Các biến độc lập không có tương quan cao")

        print("\n3. PHƯƠNG PHÁP ĐÁNH GIÁ MÔ HÌNH:")
        print("-" * 40)
        print("a) R² (Coefficient of Determination):")
        print("   - Đo lường tỷ lệ phương sai được giải thích bởi mô hình")
        print("   - Giá trị từ 0 đến 1, càng gần 1 càng tốt")
        print("   - Công thức: R² = 1 - (SSres/SStot)")

        print("\nb) Mean Squared Error (MSE):")
        print("   - Trung bình bình phương của sai số")
        print("   - Công thức: MSE = (1/n) * Σ(yᵢ - ŷᵢ)²")
        print("   - Giá trị càng nhỏ càng tốt")

        print("\nc) Mean Absolute Error (MAE):")
        print("   - Trung bình giá trị tuyệt đối của sai số")
        print("   - Công thức: MAE = (1/n) * Σ|yᵢ - ŷᵢ|")
        print("   - Ít nhạy cảm với outliers hơn MSE")

        print("\n4. CÁC KỸ THUẬT TIỀN XỬ LÝ ÁP DỤNG:")
        print("-" * 40)
        print("a) Chuẩn hóa dữ liệu:")
        print("   - Z-Score: (x - μ)/σ")
        print("   - Min-Max: (x - min)/(max - min)")
        print("   - Unit Vector: x/||x||₂")

        print("\nb) Trích chọn thuộc tính:")
        print("   - SelectKBest với f_regression")
        print("   - Chọn k features có điểm F-score cao nhất")

        print("\nc) Xử lý outliers:")
        print("   - Sử dụng phương pháp IQR (Interquartile Range)")
        print("   - Loại bỏ các giá trị nằm ngoài [Q1-1.5*IQR, Q3+1.5*IQR]")

    def train_models(self):
        """
        YÊU CẦU 3: Thực hiện các kỹ thuật phân tích dữ liệu
        """
        print("\n" + "=" * 60)
        print("YÊU CẦU 3: THỰC HIỆN CÁC KỸ THUẬT PHÂN TÍCH DỮ LIỆU")
        print("=" * 60)

        print("\n1. HUẤN LUYỆN MÔ HÌNH HỒI QUY TUYẾN TÍNH:")
        print("-" * 40)
        print("Sử dụng Z-Score Normalization")

        # Tạo và huấn luyện mô hình
        self.model = LinearRegression()
        self.model.fit(self.X_train_normalized, self.y_train)

        # Dự đoán
        self.y_train_pred = self.model.predict(self.X_train_normalized)
        self.y_test_pred = self.model.predict(self.X_test_normalized)

        # Tính các metrics
        self.train_r2 = r2_score(self.y_train, self.y_train_pred)
        self.test_r2 = r2_score(self.y_test, self.y_test_pred)
        self.train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        self.test_mse = mean_squared_error(self.y_test, self.y_test_pred)
        self.train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        self.test_mae = mean_absolute_error(self.y_test, self.y_test_pred)

        # Hiển thị kết quả
        print(f"\nKết quả mô hình Z-Score Normalization:")
        print(f"  Train R²: {self.train_r2:.4f}")
        print(f"  Test R²: {self.test_r2:.4f}")
        print(f"  Train MSE: {self.train_mse:.2f}")
        print(f"  Test MSE: {self.test_mse:.2f}")
        print(f"  Train MAE: {self.train_mae:.2f}")
        print(f"  Test MAE: {self.test_mae:.2f}")

        # Lưu kết quả để tương thích với code cũ
        self.model_results = {
            'zscore': {
                'model': self.model,
                'train_r2': self.train_r2,
                'test_r2': self.test_r2,
                'train_mse': self.train_mse,
                'test_mse': self.test_mse,
                'train_mae': self.train_mae,
                'test_mae': self.test_mae,
                'y_train_pred': self.y_train_pred,
                'y_test_pred': self.y_test_pred
            }
        }

        self.best_method = 'zscore'
        self.best_model = self.model

        return self.model_results

    def analyze_results(self):
        """
        Phân tích chi tiết kết quả mô hình
        """
        print("\n2. PHÂN TÍCH CHI TIẾT MÔ HÌNH:")
        print("-" * 40)

        # Hiển thị hệ số hồi quy
        print(f"Phương pháp chuẩn hóa: Z-Score Normalization")
        print(f"Intercept (β₀): {self.model.intercept_:.2f}")

        print("\nHệ số hồi quy (βᵢ):")
        feature_names = self.X_train.columns
        for i, (feature, coef) in enumerate(zip(feature_names, self.model.coef_)):
            print(f"  {feature}: {coef:.2f}")

        # Phân tích tầm quan trọng của features
        feature_importance = abs(self.model.coef_)
        feature_importance_normalized = feature_importance / feature_importance.sum() * 100

        print("\nTầm quan trọng của các features (%):")
        for feature, importance in zip(feature_names, feature_importance_normalized):
            print(f"  {feature}: {importance:.2f}%")

        return self.model_results['zscore']

    def visualize_results(self):
        """
        Trực quan hóa kết quả mô hình
        """
        print("\n3. TRỰC QUAN HÓA KẾT QUẢ:")
        print("-" * 40)

        best_result = self.model_results[self.best_method]

        # Tạo figure với nhiều subplot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Linear Regression Model Analysis Results', fontsize=16, fontweight='bold')

        # 1. Hiển thị R² Score của mô hình
        bars = axes[0, 0].bar(['Z-Score Normalization'], [self.test_r2], color='skyblue')
        axes[0, 0].set_title('R² Score của mô hình')
        axes[0, 0].set_xlabel('Phương pháp chuẩn hóa')
        axes[0, 0].set_ylabel('Test R² Score')
        axes[0, 0].set_ylim(0, max(0.5, self.test_r2 + 0.1))

        # Thêm giá trị lên cột
        axes[0, 0].text(0, self.test_r2 + 0.01, f'{self.test_r2:.3f}',
                       ha='center', va='bottom', fontweight='bold')

        # 2. Actual vs Predicted (Test set)
        axes[0, 1].scatter(self.y_test, self.y_test_pred, alpha=0.6, color='blue')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()],
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_title('Actual vs Predicted (Test Set)')
        axes[0, 1].set_xlabel('Actual Salary (USD)')
        axes[0, 1].set_ylabel('Predicted Salary (USD)')

        # 3. Residuals plot
        residuals = self.y_test - self.y_test_pred
        axes[0, 2].scatter(self.y_test_pred, residuals, alpha=0.6, color='green')
        axes[0, 2].axhline(y=0, color='r', linestyle='--')
        axes[0, 2].set_title('Residuals Plot')
        axes[0, 2].set_xlabel('Predicted Salary (USD)')
        axes[0, 2].set_ylabel('Residuals')

        # 4. Feature importance
        feature_names = self.X_train.columns
        feature_importance = abs(self.model.coef_)
        feature_importance_normalized = feature_importance / feature_importance.sum() * 100

        # Sắp xếp theo tầm quan trọng
        sorted_idx = np.argsort(feature_importance_normalized)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = feature_importance_normalized[sorted_idx]

        axes[1, 0].barh(range(len(sorted_features)), sorted_importance, color='orange')
        axes[1, 0].set_title('Tầm quan trọng của Features')
        axes[1, 0].set_xlabel('Tầm quan trọng (%)')
        axes[1, 0].set_yticks(range(len(sorted_features)))
        axes[1, 0].set_yticklabels(sorted_features)

        # 5. Distribution of residuals
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Phân phối Residuals')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Tần suất')

        # 6. US vs Non-US salary comparison với predictions
        us_actual = self.df_processed.groupby('is_us')['salary_in_usd'].mean()

        # Tạo dữ liệu để dự đoán cho US vs Non-US
        us_comparison_data = []
        feature_columns = ['work_year', 'experience_level_encoded', 'employment_type_encoded',
                          'job_title_encoded', 'remote_ratio', 'company_size_encoded', 'is_us']

        for is_us in [0, 1]:
            subset = self.df_processed[self.df_processed['is_us'] == is_us]
            if len(subset) > 0:
                # Tạo dữ liệu với các features đã encoded
                mean_data = []
                mean_data.append(subset['work_year'].mean())
                mean_data.append(subset['experience_level_encoded'].mean())
                mean_data.append(subset['employment_type_encoded'].mean())
                mean_data.append(subset['job_title_encoded'].mean())
                mean_data.append(subset['remote_ratio'].mean())
                mean_data.append(subset['company_size_encoded'].mean())
                mean_data.append(is_us)
                us_comparison_data.append(mean_data)

        if len(us_comparison_data) == 2:
            # Chuẩn hóa dữ liệu bằng Z-Score Normalization
            us_data_normalized = self.scaler.transform(us_comparison_data)
            us_predictions = self.model.predict(us_data_normalized)

            x_pos = np.arange(2)
            width = 0.35

            axes[1, 2].bar(x_pos - width/2, [us_actual[0], us_actual[1]], width,
                          label='Actual', color='lightblue', alpha=0.8)
            axes[1, 2].bar(x_pos + width/2, us_predictions, width,
                          label='Predicted', color='lightcoral', alpha=0.8)

            axes[1, 2].set_title('So sánh Actual vs Predicted: US vs Non-US')
            axes[1, 2].set_xlabel('Khu vực')
            axes[1, 2].set_ylabel('Mức lương trung bình (USD)')
            axes[1, 2].set_xticks(x_pos)
            axes[1, 2].set_xticklabels(['Non-US', 'US'])
            axes[1, 2].legend()

        plt.tight_layout()

        # Lưu ảnh với nền trắng
        plt.savefig('images/02_model_results.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu biểu đồ kết quả mô hình: images/02_model_results.png")
        plt.close()  # Đóng figure để giải phóng bộ nhớ

    def generate_additional_charts(self):
        """
        Tạo các biểu đồ bổ sung quan trọng
        """
        print("\n4. TẠO CÁC BIỂU ĐỒ BỔ SUNG:")
        print("-" * 40)

        # 1. Correlation Matrix
        print("Tạo correlation matrix...")
        plt.figure(figsize=(12, 10))

        # Chọn các cột số để tính correlation
        numeric_cols = ['work_year', 'salary_in_usd', 'remote_ratio',
                       'experience_level_encoded', 'employment_type_encoded',
                       'job_title_encoded', 'company_size_encoded', 'is_us']

        corr_data = self.df_processed[numeric_cols]
        correlation_matrix = corr_data.corr()

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, fmt='.2f')
        plt.title('Ma trận tương quan giữa các biến', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('images/03_correlation_matrix.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/03_correlation_matrix.png")
        plt.close()

        # 2. Salary Distribution by Job Title
        print("Tạo phân phối lương theo job title...")
        plt.figure(figsize=(15, 8))

        # Lấy top 10 job titles có nhiều mẫu nhất
        top_jobs = self.df_processed['job_title'].value_counts().head(10).index
        data_subset = self.df_processed[self.df_processed['job_title'].isin(top_jobs)]

        sns.boxplot(data=data_subset, x='job_title', y='salary_in_usd')
        plt.xticks(rotation=45, ha='right')
        plt.title('Phân phối mức lương theo Job Title (Top 10)', fontsize=16, fontweight='bold')
        plt.xlabel('Job Title')
        plt.ylabel('Mức lương (USD)')
        plt.tight_layout()
        plt.savefig('images/04_salary_by_job.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/04_salary_by_job.png")
        plt.close()

        # 3. Experience Level Impact
        print("Tạo biểu đồ tác động của experience level...")
        plt.figure(figsize=(12, 6))

        exp_mapping = {0: 'Entry', 1: 'Mid', 2: 'Senior', 3: 'Executive'}
        self.df_processed['exp_level_name'] = self.df_processed['experience_level_encoded'].map(exp_mapping)

        exp_stats = self.df_processed.groupby('exp_level_name')['salary_in_usd'].agg(['mean', 'count'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Mean salary by experience
        exp_stats['mean'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Mức lương trung bình theo Experience Level')
        ax1.set_ylabel('Mức lương trung bình (USD)')
        ax1.tick_params(axis='x', rotation=45)

        # Count by experience
        exp_stats['count'].plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Số lượng mẫu theo Experience Level')
        ax2.set_ylabel('Số lượng')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('images/05_experience_impact.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/05_experience_impact.png")
        plt.close()

        # 4. Company Size vs Salary
        print("Tạo biểu đồ company size vs salary...")
        plt.figure(figsize=(10, 6))

        size_mapping = {0: 'Small', 1: 'Medium', 2: 'Large'}
        self.df_processed['company_size_name'] = self.df_processed['company_size_encoded'].map(size_mapping)

        sns.violinplot(data=self.df_processed, x='company_size_name', y='salary_in_usd')
        plt.title('Phân phối mức lương theo quy mô công ty', fontsize=16, fontweight='bold')
        plt.xlabel('Quy mô công ty')
        plt.ylabel('Mức lương (USD)')
        plt.tight_layout()
        plt.savefig('images/06_company_size_salary.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/06_company_size_salary.png")
        plt.close()

        # 5. Remote Ratio Impact
        print("Tạo biểu đồ tác động của remote ratio...")
        plt.figure(figsize=(12, 6))

        # Chia remote ratio thành các nhóm
        self.df_processed['remote_group'] = pd.cut(self.df_processed['remote_ratio'],
                                                  bins=[0, 25, 75, 100],
                                                  labels=['On-site (0-25%)', 'Hybrid (25-75%)', 'Remote (75-100%)'],
                                                  include_lowest=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot
        sns.boxplot(data=self.df_processed, x='remote_group', y='salary_in_usd', ax=ax1)
        ax1.set_title('Mức lương theo mức độ Remote')
        ax1.set_xlabel('Mức độ Remote')
        ax1.set_ylabel('Mức lương (USD)')
        ax1.tick_params(axis='x', rotation=45)

        # Scatter plot với cải thiện
        ax2.scatter(self.df_processed['remote_ratio'], self.df_processed['salary_in_usd'],
                   alpha=0.6, s=20, color='blue')
        ax2.set_title('Mối quan hệ Remote Ratio vs Salary')
        ax2.set_xlabel('Remote Ratio (%)')
        ax2.set_ylabel('Mức lương (USD)')
        ax2.set_xticks([0, 25, 50, 75, 100])
        ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('images/07_remote_impact.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/07_remote_impact.png")
        plt.close()

        return True

    def generate_residuals_analysis(self):
        """
        Tạo biểu đồ phân tích residuals chi tiết
        """
        print("Tạo phân tích residuals chi tiết...")

        residuals = self.y_test - self.y_test_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phân tích Residuals chi tiết', fontsize=16, fontweight='bold')

        # 1. Histogram of residuals
        axes[0, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Phân phối Residuals')
        axes[0, 0].set_xlabel('Residuals')
        axes[0, 0].set_ylabel('Tần suất')
        axes[0, 0].axvline(residuals.mean(), color='red', linestyle='--',
                          label=f'Mean: {residuals.mean():.2f}')
        axes[0, 0].legend()

        # 2. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[0, 1].grid(True)

        # 3. Residuals vs Fitted
        axes[1, 0].scatter(self.y_test_pred, residuals, alpha=0.6, color='green')
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Residuals vs Fitted Values')
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('Residuals')

        # 4. Scale-Location plot
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[1, 1].scatter(self.y_test_pred, sqrt_abs_residuals, alpha=0.6, color='orange')
        axes[1, 1].set_title('Scale-Location Plot')
        axes[1, 1].set_xlabel('Fitted Values')
        axes[1, 1].set_ylabel('√|Residuals|')

        plt.tight_layout()
        plt.savefig('images/08_residuals_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/08_residuals_analysis.png")
        plt.close()

        return True

    def generate_feature_importance_chart(self):
        """
        Tạo biểu đồ tầm quan trọng của features
        """
        print("Tạo biểu đồ tầm quan trọng features...")

        feature_names = self.X_train.columns
        feature_importance = abs(self.model.coef_)
        feature_importance_normalized = feature_importance / feature_importance.sum() * 100

        # Tạo DataFrame để dễ xử lý
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance_normalized,
            'Coefficient': self.model.coef_
        }).sort_values('Importance', ascending=True)

        plt.figure(figsize=(12, 8))

        # Tạo màu sắc dựa trên dấu của coefficient
        colors = ['red' if coef < 0 else 'green' for coef in importance_df['Coefficient']]

        bars = plt.barh(range(len(importance_df)), importance_df['Importance'], color=colors, alpha=0.7)
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Tầm quan trọng (%)')
        plt.title('Tầm quan trọng của các Features trong mô hình\n(Xanh: Tác động tích cực, Đỏ: Tác động tiêu cực)',
                 fontsize=14, fontweight='bold')

        # Thêm giá trị lên các thanh
        for i, (bar, importance, coef) in enumerate(zip(bars, importance_df['Importance'], importance_df['Coefficient'])):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{importance:.1f}%\n({coef:.0f})',
                    ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('images/09_feature_importance.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/09_feature_importance.png")
        plt.close()

        return True

    def predict_salary(self, input_data):
        """
        Dự báo mức lương cho dữ liệu đầu vào mới

        Args:
            input_data (dict): Dictionary chứa thông tin để dự báo
                - work_year: năm (int)
                - experience_level: 'EN', 'MI', 'SE', 'EX' (str)
                - employment_type: 'FT', 'PT', 'CT', 'FL' (str)
                - job_title: tên công việc (str)
                - remote_ratio: tỷ lệ remote 0-100 (int)
                - company_size: 'S', 'M', 'L' (str)
                - company_location: mã quốc gia như 'US', 'VN', etc. (str)

        Returns:
            dict: Kết quả dự báo với confidence interval
        """
        try:
            # Kiểm tra mô hình đã được train chưa
            if not hasattr(self, 'model') or self.model is None:
                raise ValueError("Mô hình chưa được huấn luyện. Vui lòng chạy train_models() trước.")

            # Tạo DataFrame từ input
            input_df = pd.DataFrame([input_data])

            # Xử lý các biến categorical giống như trong training
            # Experience level encoding
            exp_mapping = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
            input_df['experience_level_encoded'] = input_df['experience_level'].map(exp_mapping)

            # Employment type encoding
            emp_mapping = {'FT': 0, 'PT': 1, 'CT': 2, 'FL': 3}
            input_df['employment_type_encoded'] = input_df['employment_type'].map(emp_mapping)

            # Job title encoding (sử dụng label encoder đã fit)
            if input_data['job_title'] in self.label_encoders['job_title'].classes_:
                input_df['job_title_encoded'] = self.label_encoders['job_title'].transform([input_data['job_title']])[0]
            else:
                # Nếu job title mới, sử dụng giá trị trung bình
                input_df['job_title_encoded'] = self.X_train['job_title_encoded'].mean()
                print(f"Cảnh báo: Job title '{input_data['job_title']}' không có trong dữ liệu training. Sử dụng giá trị trung bình.")

            # Company size encoding
            size_mapping = {'S': 0, 'M': 1, 'L': 2}
            input_df['company_size_encoded'] = input_df['company_size'].map(size_mapping)

            # US vs Non-US
            input_df['is_us'] = (input_df['company_location'] == 'US').astype(int)

            # Chọn features giống như trong training
            feature_columns = ['work_year', 'experience_level_encoded', 'employment_type_encoded',
                             'job_title_encoded', 'remote_ratio', 'company_size_encoded', 'is_us']

            X_input = input_df[feature_columns]

            # Chuẩn hóa dữ liệu
            X_input_normalized = self.scaler.transform(X_input)

            # Dự báo
            predicted_salary = self.model.predict(X_input_normalized)[0]

            # Tính confidence interval dựa trên MAE
            confidence_interval = self.test_mae * 1.96  # 95% confidence interval
            lower_bound = predicted_salary - confidence_interval
            upper_bound = predicted_salary + confidence_interval

            # Đảm bảo lower bound không âm
            lower_bound = max(0, lower_bound)

            result = {
                'predicted_salary': predicted_salary,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence_interval': confidence_interval,
                'input_processed': X_input.iloc[0].to_dict()
            }

            return result

        except Exception as e:
            raise ValueError(f"Lỗi trong quá trình dự báo: {str(e)}")

    def predict_multiple_scenarios(self):
        """
        Dự báo cho nhiều kịch bản khác nhau
        """
        print("\n" + "=" * 60)
        print("DỰ BÁO MỨC LƯƠNG CHO CÁC KỊCH BẢN KHÁC NHAU")
        print("=" * 60)

        # Định nghĩa các kịch bản
        scenarios = [
            {
                'name': 'Data Scientist - Senior - US - Remote',
                'data': {
                    'work_year': 2024,
                    'experience_level': 'SE',
                    'employment_type': 'FT',
                    'job_title': 'Data Scientist',
                    'remote_ratio': 100,
                    'company_size': 'L',
                    'company_location': 'US'
                }
            },
            {
                'name': 'Data Engineer - Mid-level - US - Hybrid',
                'data': {
                    'work_year': 2024,
                    'experience_level': 'MI',
                    'employment_type': 'FT',
                    'job_title': 'Data Engineer',
                    'remote_ratio': 50,
                    'company_size': 'M',
                    'company_location': 'US'
                }
            },
            {
                'name': 'Data Analyst - Entry - Non-US - On-site',
                'data': {
                    'work_year': 2024,
                    'experience_level': 'EN',
                    'employment_type': 'FT',
                    'job_title': 'Data Analyst',
                    'remote_ratio': 0,
                    'company_size': 'S',
                    'company_location': 'VN'
                }
            },
            {
                'name': 'Data Scientist - Executive - US - On-site',
                'data': {
                    'work_year': 2024,
                    'experience_level': 'EX',
                    'employment_type': 'FT',
                    'job_title': 'Data Scientist',
                    'remote_ratio': 0,
                    'company_size': 'L',
                    'company_location': 'US'
                }
            },
            {
                'name': 'Data Engineer - Senior - Non-US - Remote',
                'data': {
                    'work_year': 2024,
                    'experience_level': 'SE',
                    'employment_type': 'FT',
                    'job_title': 'Data Engineer',
                    'remote_ratio': 100,
                    'company_size': 'M',
                    'company_location': 'DE'
                }
            }
        ]

        results = []

        for scenario in scenarios:
            try:
                prediction = self.predict_salary(scenario['data'])
                results.append({
                    'scenario': scenario['name'],
                    'prediction': prediction
                })

                print(f"\n{scenario['name']}:")
                print(f"  Dự báo lương: ${prediction['predicted_salary']:,.0f}")
                print(f"  Khoảng tin cậy 95%: ${prediction['lower_bound']:,.0f} - ${prediction['upper_bound']:,.0f}")

            except Exception as e:
                print(f"\nLỗi dự báo cho {scenario['name']}: {str(e)}")

        return results

    def create_prediction_interface(self):
        """
        Tạo giao diện dự báo tương tác
        """
        print("\n" + "=" * 60)
        print("GIAO DIỆN DỰ BÁO MỨC LƯƠNG TƯƠNG TÁC")
        print("=" * 60)

        print("\nHướng dẫn nhập thông tin:")
        print("- Work Year: Năm (ví dụ: 2024)")
        print("- Experience Level: EN (Entry), MI (Mid), SE (Senior), EX (Executive)")
        print("- Employment Type: FT (Full-time), PT (Part-time), CT (Contract), FL (Freelance)")
        print("- Job Title: Tên công việc (ví dụ: Data Scientist, Data Engineer, Data Analyst)")
        print("- Remote Ratio: Tỷ lệ remote 0-100 (0: on-site, 100: full remote)")
        print("- Company Size: S (Small), M (Medium), L (Large)")
        print("- Company Location: Mã quốc gia (US, VN, DE, UK, etc.)")

        # Ví dụ sử dụng
        example_input = {
            'work_year': 2024,
            'experience_level': 'SE',
            'employment_type': 'FT',
            'job_title': 'Data Scientist',
            'remote_ratio': 50,
            'company_size': 'M',
            'company_location': 'US'
        }

        print(f"\nVí dụ input:")
        for key, value in example_input.items():
            print(f"  {key}: {value}")

        try:
            result = self.predict_salary(example_input)
            print(f"\nKết quả dự báo cho ví dụ:")
            print(f"  Mức lương dự báo: ${result['predicted_salary']:,.0f}")
            print(f"  Khoảng tin cậy 95%: ${result['lower_bound']:,.0f} - ${result['upper_bound']:,.0f}")

        except Exception as e:
            print(f"Lỗi trong ví dụ dự báo: {str(e)}")

        print(f"\nĐể sử dụng, gọi hàm predict_salary() với dictionary chứa thông tin cần dự báo.")

    def visualize_predictions(self):
        """
        Tạo biểu đồ so sánh dự báo cho các kịch bản khác nhau
        """
        print("Tạo biểu đồ dự báo...")

        # Định nghĩa các kịch bản để so sánh
        scenarios = [
            ('Data Scientist\n(US, Senior, Remote)', {
                'work_year': 2024, 'experience_level': 'SE', 'employment_type': 'FT',
                'job_title': 'Data Scientist', 'remote_ratio': 100, 'company_size': 'L', 'company_location': 'US'
            }),
            ('Data Engineer\n(US, Mid, Hybrid)', {
                'work_year': 2024, 'experience_level': 'MI', 'employment_type': 'FT',
                'job_title': 'Data Engineer', 'remote_ratio': 50, 'company_size': 'M', 'company_location': 'US'
            }),
            ('Data Analyst\n(Non-US, Entry, On-site)', {
                'work_year': 2024, 'experience_level': 'EN', 'employment_type': 'FT',
                'job_title': 'Data Analyst', 'remote_ratio': 0, 'company_size': 'S', 'company_location': 'VN'
            }),
            ('Data Scientist\n(Non-US, Senior, Remote)', {
                'work_year': 2024, 'experience_level': 'SE', 'employment_type': 'FT',
                'job_title': 'Data Scientist', 'remote_ratio': 100, 'company_size': 'M', 'company_location': 'DE'
            }),
            ('Data Engineer\n(US, Executive, On-site)', {
                'work_year': 2024, 'experience_level': 'EX', 'employment_type': 'FT',
                'job_title': 'Data Engineer', 'remote_ratio': 0, 'company_size': 'L', 'company_location': 'US'
            })
        ]

        # Tính dự báo cho từng kịch bản
        predictions = []
        labels = []
        lower_bounds = []
        upper_bounds = []

        for label, scenario in scenarios:
            try:
                result = self.predict_salary(scenario)
                predictions.append(result['predicted_salary'])
                lower_bounds.append(result['lower_bound'])
                upper_bounds.append(result['upper_bound'])
                labels.append(label)
            except Exception as e:
                print(f"Lỗi dự báo cho {label}: {e}")

        # Tạo biểu đồ
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dự báo mức lương cho các kịch bản khác nhau', fontsize=16, fontweight='bold')

        # 1. Bar chart với confidence intervals
        x_pos = range(len(labels))
        bars = axes[0, 0].bar(x_pos, predictions, alpha=0.7, color='skyblue', edgecolor='black')

        # Thêm error bars
        errors = [[pred - lower for pred, lower in zip(predictions, lower_bounds)],
                 [upper - pred for pred, upper in zip(predictions, upper_bounds)]]
        axes[0, 0].errorbar(x_pos, predictions, yerr=errors, fmt='none', color='red', capsize=5)

        axes[0, 0].set_title('Dự báo mức lương theo kịch bản')
        axes[0, 0].set_xlabel('Kịch bản')
        axes[0, 0].set_ylabel('Mức lương (USD)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')

        # Thêm giá trị lên các cột
        for bar, pred in zip(bars, predictions):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                           f'${pred:,.0f}', ha='center', va='bottom', fontsize=9)

        # 2. So sánh tác động Experience Level
        exp_levels = ['EN', 'MI', 'SE', 'EX']
        exp_names = ['Entry', 'Mid', 'Senior', 'Executive']
        exp_salaries = []

        base_scenario = {
            'work_year': 2024, 'employment_type': 'FT', 'job_title': 'Data Scientist',
            'remote_ratio': 50, 'company_size': 'M', 'company_location': 'US'
        }

        for level in exp_levels:
            base_scenario['experience_level'] = level
            result = self.predict_salary(base_scenario)
            exp_salaries.append(result['predicted_salary'])

        axes[0, 1].plot(exp_names, exp_salaries, marker='o', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_title('Tác động của Experience Level')
        axes[0, 1].set_xlabel('Experience Level')
        axes[0, 1].set_ylabel('Mức lương dự báo (USD)')
        axes[0, 1].grid(True, alpha=0.3)

        # Thêm giá trị lên các điểm
        for i, (name, salary) in enumerate(zip(exp_names, exp_salaries)):
            axes[0, 1].text(i, salary + 3000, f'${salary:,.0f}', ha='center', va='bottom')

        # 3. So sánh US vs Non-US
        locations = ['US', 'Non-US']
        location_salaries = []
        location_errors = []

        base_scenario['experience_level'] = 'MI'  # Reset to Mid

        # US
        base_scenario['company_location'] = 'US'
        us_result = self.predict_salary(base_scenario)
        location_salaries.append(us_result['predicted_salary'])
        location_errors.append([
            us_result['predicted_salary'] - us_result['lower_bound'],
            us_result['upper_bound'] - us_result['predicted_salary']
        ])

        # Non-US (sử dụng trung bình của các quốc gia khác)
        non_us_countries = ['DE', 'UK', 'CA', 'IN', 'VN']
        non_us_salaries = []
        for country in non_us_countries:
            base_scenario['company_location'] = country
            result = self.predict_salary(base_scenario)
            non_us_salaries.append(result['predicted_salary'])

        non_us_avg = sum(non_us_salaries) / len(non_us_salaries)
        location_salaries.append(non_us_avg)
        location_errors.append([us_result['confidence_interval'], us_result['confidence_interval']])  # Sử dụng cùng CI

        bars = axes[1, 0].bar(locations, location_salaries,
                             color=['red', 'lightblue'], alpha=0.7, edgecolor='black')

        # Thêm error bars
        axes[1, 0].errorbar(range(len(locations)), location_salaries,
                           yerr=[[err[0] for err in location_errors], [err[1] for err in location_errors]],
                           fmt='none', color='black', capsize=5)

        axes[1, 0].set_title('Dự báo lương: US vs Non-US')
        axes[1, 0].set_xlabel('Khu vực')
        axes[1, 0].set_ylabel('Mức lương dự báo (USD)')

        # Thêm giá trị lên các cột
        for bar, salary in zip(bars, location_salaries):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                           f'${salary:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Thêm chênh lệch
        difference = location_salaries[0] - location_salaries[1]
        percentage = (difference / location_salaries[1]) * 100
        axes[1, 0].text(0.5, max(location_salaries) * 0.8,
                       f'Chênh lệch: ${difference:,.0f}\n({percentage:.1f}% cao hơn)',
                       ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # 4. Tác động của Remote Work
        remote_ratios = [0, 25, 50, 75, 100]
        remote_names = ['On-site', '25% Remote', 'Hybrid', '75% Remote', 'Full Remote']
        remote_salaries = []

        base_scenario['company_location'] = 'US'  # Reset to US

        for ratio in remote_ratios:
            base_scenario['remote_ratio'] = ratio
            result = self.predict_salary(base_scenario)
            remote_salaries.append(result['predicted_salary'])

        axes[1, 1].plot(remote_names, remote_salaries, marker='s', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_title('Tác động của Remote Work')
        axes[1, 1].set_xlabel('Mức độ Remote')
        axes[1, 1].set_ylabel('Mức lương dự báo (USD)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # Thêm giá trị lên các điểm
        for i, (name, salary) in enumerate(zip(remote_names, remote_salaries)):
            axes[1, 1].text(i, salary + 1000, f'${salary:,.0f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig('images/10_salary_predictions.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Đã lưu: images/10_salary_predictions.png")
        plt.close()

        return True

    def generate_report(self):
        """
        YÊU CẦU 4: Tổng hợp và phân tích kết quả và trình bày báo cáo
        """
        print("\n" + "=" * 60)
        print("YÊU CẦU 4: TỔNG HỢP VÀ PHÂN TÍCH KẾT QUẢ")
        print("=" * 60)

        print("\n1. TÓM TẮT DỮ LIỆU:")
        print("-" * 30)
        print(f"- Tổng số mẫu ban đầu: {self.df.shape[0]:,}")
        print(f"- Số mẫu sau khi lọc và làm sạch: {self.df_processed.shape[0]:,}")
        print(f"- Số features sử dụng: {len(self.X_train.columns)}")
        print(f"- Tỷ lệ train/test: 80%/20%")

        # Thống kê các job được phân tích
        job_stats = self.df_processed['job_title'].value_counts()
        print(f"\nCác job title được phân tích:")
        for job, count in job_stats.items():
            print(f"  - {job}: {count:,} mẫu")

        # Thống kê US vs Non-US
        us_stats = self.df_processed.groupby('is_us').agg({
            'salary_in_usd': ['count', 'mean', 'median', 'std']
        }).round(2)
        us_stats.columns = ['Count', 'Mean', 'Median', 'Std']
        us_stats.index = ['Non-US', 'US']

        print(f"\nThống kê US vs Non-US:")
        print(us_stats)

        print("\n2. KẾT QUẢ MÔ HÌNH:")
        print("-" * 30)

        print("Kết quả mô hình Z-Score Normalization:")
        print(f"- Test R²: {self.test_r2:.4f}")
        print(f"- Test MSE: {self.test_mse:.2f}")
        print(f"- Test MAE: {self.test_mae:.2f}")
        print(f"- Train R²: {self.train_r2:.4f}")
        print(f"- Train MSE: {self.train_mse:.2f}")
        print(f"- Train MAE: {self.train_mae:.2f}")

        print("\n3. PHÂN TÍCH HỆ SỐ HỒI QUY:")
        print("-" * 30)

        feature_names = self.X_train.columns

        print(f"Intercept (β₀): ${self.model.intercept_:,.2f}")
        print("\nHệ số hồi quy và ý nghĩa:")

        for feature, coef in zip(feature_names, self.model.coef_):
            direction = "tăng" if coef > 0 else "giảm"
            print(f"  - {feature}: {coef:,.2f}")
            print(f"    → Khi {feature} tăng 1 đơn vị, lương {direction} ${abs(coef):,.2f}")

        print("\n4. PHƯƠNG TRÌNH HỒI QUY TUYẾN TÍNH:")
        print("-" * 40)

        # Tạo phương trình hoàn chỉnh
        equation = f"salary_in_usd = {self.model.intercept_:.2f}"

        for feature, coef in zip(feature_names, self.model.coef_):
            if coef >= 0:
                equation += f" + {coef:.2f} * {feature}"
            else:
                equation += f" - {abs(coef):.2f} * {feature}"

        print(f"\nPhương trình dự báo:")
        print(f"{equation}")

        # Phiên bản rút gọn với ký hiệu toán học
        print(f"\nDạng toán học:")
        equation_math = f"ŷ = {self.model.intercept_:.2f}"

        for i, (feature, coef) in enumerate(zip(feature_names, self.model.coef_)):
            if coef >= 0:
                equation_math += f" + {coef:.2f}x₍{i+1}₎"
            else:
                equation_math += f" - {abs(coef):.2f}x₍{i+1}₎"

        print(f"{equation_math}")

        print(f"\nTrong đó:")
        print(f"  ŷ = salary_in_usd (mức lương dự báo)")
        for i, feature in enumerate(feature_names):
            print(f"  x₍{i+1}₎ = {feature}")

        # Ví dụ tính toán
        print(f"\nVí dụ tính toán cho Data Scientist Senior tại US:")
        example_values = {
            'work_year': 2024,
            'experience_level_encoded': 2,  # SE = 2
            'employment_type_encoded': 0,   # FT = 0
            'job_title_encoded': 1,         # Data Scientist
            'remote_ratio': 50,
            'company_size_encoded': 1,      # M = 1
            'is_us': 1
        }

        calculation = f"ŷ = {self.model.intercept_:.2f}"
        predicted_value = self.model.intercept_

        for feature, coef in zip(feature_names, self.model.coef_):
            value = example_values.get(feature, 0)
            contribution = coef * value
            predicted_value += contribution

            if coef >= 0:
                calculation += f" + {coef:.2f} × {value}"
            else:
                calculation += f" - {abs(coef):.2f} × {value}"

        print(f"{calculation}")
        print(f"ŷ = ${predicted_value:,.2f}")

        print(f"\nGiải thích:")
        print(f"  - Mức lương cơ sở: ${self.model.intercept_:,.2f}")
        for feature, coef in zip(feature_names, self.model.coef_):
            value = example_values.get(feature, 0)
            contribution = coef * value
            if contribution != 0:
                effect = "tăng" if contribution > 0 else "giảm"
                print(f"  - {feature} ({value}): {effect} ${abs(contribution):,.2f}")
        print(f"  - Tổng cộng: ${predicted_value:,.2f}")

        print("\n4. ĐÁNH GIÁ MÔ HÌNH:")
        print("-" * 30)

        r2_interpretation = ""
        if self.test_r2 >= 0.8:
            r2_interpretation = "Rất tốt - Mô hình giải thích được hơn 80% phương sai"
        elif self.test_r2 >= 0.6:
            r2_interpretation = "Tốt - Mô hình giải thích được 60-80% phương sai"
        elif self.test_r2 >= 0.4:
            r2_interpretation = "Trung bình - Mô hình giải thích được 40-60% phương sai"
        else:
            r2_interpretation = "Kém - Mô hình giải thích được ít hơn 40% phương sai"

        print(f"R² Score: {self.test_r2:.4f} - {r2_interpretation}")

        # Tính RMSE
        rmse = np.sqrt(self.test_mse)
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${self.test_mae:,.2f}")

        # Phân tích residuals
        residuals = self.y_test - self.y_test_pred
        print(f"\nPhân tích Residuals:")
        print(f"- Mean: ${residuals.mean():,.2f}")
        print(f"- Std: ${residuals.std():,.2f}")
        print(f"- Min: ${residuals.min():,.2f}")
        print(f"- Max: ${residuals.max():,.2f}")

        print("\n5. KẾT LUẬN VÀ KHUYẾN NGHỊ:")
        print("-" * 30)

        print("Kết luận chính:")
        print(f"1. Mô hình hồi quy tuyến tính với Z-Score Normalization")
        print(f"2. Mô hình có thể giải thích {self.test_r2*100:.1f}% phương sai của mức lương")
        print(f"3. Sai số trung bình tuyệt đối là ${self.test_mae:,.2f}")

        # Tìm feature quan trọng nhất
        feature_importance = abs(self.model.coef_)
        most_important_idx = np.argmax(feature_importance)
        most_important_feature = feature_names[most_important_idx]

        print(f"4. Feature quan trọng nhất: {most_important_feature}")

        # Hiển thị số lượng job types được phân tích
        job_types = self.df_processed['job_title'].nunique()
        print(f"5. Đã phân tích {job_types} loại công việc chứa từ 'Data' (xuất hiện > 30 lần)")

        print("\nKhuyến nghị:")
        print("1. Có thể sử dụng mô hình này để ước tính mức lương trong ngành Data Science")
        print("2. Cần thu thập thêm dữ liệu để cải thiện độ chính xác")
        print("3. Xem xét thêm các features khác như kỹ năng, chứng chỉ, kinh nghiệm cụ thể")
        print("4. Định kỳ cập nhật mô hình với dữ liệu mới")

        return {
            'method': 'Z-Score Normalization',
            'test_r2': self.test_r2,
            'test_mse': self.test_mse,
            'test_mae': self.test_mae,
            'train_r2': self.train_r2,
            'train_mse': self.train_mse,
            'train_mae': self.train_mae,
            'feature_importance': dict(zip(feature_names, self.model.coef_)),
            'job_types_analyzed': job_types
        }

    def draw_visualize(self):
        """
        Tạo tất cả các biểu đồ phân tích
        """
        print("\n" + "=" * 60)
        print("TẠO TẤT CẢ CÁC BIỂU ĐỒ PHÂN TÍCH")
        print("=" * 60)

        # 1. Biểu đồ khám phá dữ liệu (2 ảnh)
        print("\n1. Tạo biểu đồ khám phá dữ liệu...")
        self.visualize_data()

        # 2. Biểu đồ kết quả mô hình
        print("\n2. Tạo biểu đồ kết quả mô hình...")
        self.visualize_results()

        # 3. Các biểu đồ bổ sung (5 ảnh)
        print("\n3. Tạo các biểu đồ bổ sung...")
        self.generate_additional_charts()

        # 4. Phân tích residuals
        print("\n4. Tạo phân tích residuals...")
        self.generate_residuals_analysis()

        # 5. Tầm quan trọng features
        print("\n5. Tạo biểu đồ tầm quan trọng features...")
        self.generate_feature_importance_chart()

        # 6. Biểu đồ dự báo
        print("\n6. Tạo biểu đồ dự báo...")
        self.visualize_predictions()

        print("\n✅ Đã tạo tất cả biểu đồ thành công!")
        print("📊 Tổng cộng: 11 biểu đồ chất lượng cao")

    def run_complete_analysis(self):
        """
        Chạy toàn bộ quy trình phân tích - Clean version
        """
        print("PHÂN TÍCH DỰ BÁO MỨC LƯƠNG NGÀNH KHOA HỌC DỮ LIỆU")
        print("BẰNG PHƯƠNG PHÁP HỒI QUY TUYẾN TÍNH")
        print("=" * 80)

        # Bước 1: Tải và khám phá dữ liệu
        self.load_data()
        self.explore_data()

        # Bước 2: Làm sạch và tiền xử lý dữ liệu
        self.clean_data()
        self.preprocess_data()

        # Bước 3: Mô tả phương pháp
        self.describe_methods()

        # Bước 4: Huấn luyện và phân tích mô hình
        self.train_models()
        self.analyze_results()

        # Bước 5: Dự báo cho các kịch bản
        self.predict_multiple_scenarios()
        self.create_prediction_interface()

        # Bước 6: Tạo tất cả biểu đồ
        self.draw_visualize()

        # Bước 7: Tổng hợp báo cáo
        self.generate_report()

        print("\n" + "=" * 80)
        print("HOÀN THÀNH PHÂN TÍCH VÀ DỰ BÁO!")
        print("=" * 80)

        # Return kết quả cuối cùng
        return {
            'method': 'Z-Score Normalization',
            'test_r2': self.test_r2,
            'test_mse': self.test_mse,
            'test_mae': self.test_mae,
            'train_r2': self.train_r2,
            'train_mse': self.train_mse,
            'train_mae': self.train_mae,
            'job_types_analyzed': len(self.df_processed['job_title'].unique())
        }


def main():
    """
    Hàm main để chạy phân tích
    """
    # Đường dẫn đến file dữ liệu
    data_path = "data-salary.csv"

    try:
        # Tạo đối tượng phân tích
        analyzer = DataScienceSalaryAnalysis(data_path)

        # Chạy phân tích hoàn chỉnh
        results = analyzer.run_complete_analysis()

        # In kết quả cuối cùng
        print(f"\nKết quả cuối cùng:")
        print(f"- Phương pháp: {results['method']}")
        print(f"- Test R² Score: {results['test_r2']:.4f}")
        print(f"- Test MSE: {results['test_mse']:.2f}")
        print(f"- Test MAE: {results['test_mae']:.2f}")
        print(f"- Số loại công việc phân tích: {results['job_types_analyzed']}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {data_path}")
        print("Vui lòng kiểm tra đường dẫn file dữ liệu")
    except Exception as e:
        print(f"Lỗi trong quá trình phân tích: {str(e)}")


if __name__ == "__main__":
    main()
