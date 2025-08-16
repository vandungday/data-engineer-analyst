#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo chức năng dự báo mức lương
"""

from data_science_salary_analysis import DataScienceSalaryAnalysis

def demo_salary_prediction():
    """
    Demo chức năng dự báo mức lương
    """
    print("DEMO DỰ BÁO MỨC LƯƠNG NGÀNH DATA SCIENCE")
    print("=" * 60)
    
    # Đường dẫn đến file dữ liệu
    data_path = "data-salary.csv"
    
    try:
        # Tạo và huấn luyện mô hình
        print("Đang tải dữ liệu và huấn luyện mô hình...")
        analyzer = DataScienceSalaryAnalysis(data_path)
        analyzer.load_data()
        analyzer.clean_data()
        analyzer.preprocess_data()
        analyzer.train_models()
        
        print("Mô hình đã được huấn luyện thành công!")
        print(f"R² Score: {analyzer.test_r2:.4f}")
        print(f"MAE: ${analyzer.test_mae:,.0f}")
        
        # Demo dự báo cho các trường hợp cụ thể
        print("\n" + "=" * 60)
        print("DỰ BÁO CHO CÁC TRƯỜNG HỢP CỤ THỂ")
        print("=" * 60)
        
        # Trường hợp 1: Data Scientist Senior tại US
        case1 = {
            'work_year': 2024,
            'experience_level': 'SE',
            'employment_type': 'FT',
            'job_title': 'Data Scientist',
            'remote_ratio': 100,
            'company_size': 'L',
            'company_location': 'US'
        }
        
        result1 = analyzer.predict_salary(case1)
        print("\n1. Data Scientist - Senior - US - Remote - Large Company:")
        print(f"   Dự báo lương: ${result1['predicted_salary']:,.0f}")
        print(f"   Khoảng tin cậy 95%: ${result1['lower_bound']:,.0f} - ${result1['upper_bound']:,.0f}")
        
        # Trường hợp 2: Data Engineer Mid-level tại Việt Nam
        case2 = {
            'work_year': 2024,
            'experience_level': 'MI',
            'employment_type': 'FT',
            'job_title': 'Data Engineer',
            'remote_ratio': 50,
            'company_size': 'M',
            'company_location': 'VN'
        }
        
        result2 = analyzer.predict_salary(case2)
        print("\n2. Data Engineer - Mid-level - Vietnam - Hybrid - Medium Company:")
        print(f"   Dự báo lương: ${result2['predicted_salary']:,.0f}")
        print(f"   Khoảng tin cậy 95%: ${result2['lower_bound']:,.0f} - ${result2['upper_bound']:,.0f}")
        
        # Trường hợp 3: Data Analyst Entry level
        case3 = {
            'work_year': 2024,
            'experience_level': 'EN',
            'employment_type': 'FT',
            'job_title': 'Data Analyst',
            'remote_ratio': 0,
            'company_size': 'S',
            'company_location': 'IN'
        }
        
        result3 = analyzer.predict_salary(case3)
        print("\n3. Data Analyst - Entry - India - On-site - Small Company:")
        print(f"   Dự báo lương: ${result3['predicted_salary']:,.0f}")
        print(f"   Khoảng tin cậy 95%: ${result3['lower_bound']:,.0f} - ${result3['upper_bound']:,.0f}")
        
        # So sánh các trường hợp
        print("\n" + "=" * 60)
        print("SO SÁNH CÁC TRƯỜNG HỢP")
        print("=" * 60)
        
        cases = [
            ("Data Scientist (US, Senior, Remote)", result1['predicted_salary']),
            ("Data Engineer (VN, Mid, Hybrid)", result2['predicted_salary']),
            ("Data Analyst (IN, Entry, On-site)", result3['predicted_salary'])
        ]
        
        # Sắp xếp theo lương
        cases.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, salary) in enumerate(cases, 1):
            print(f"{i}. {name}: ${salary:,.0f}")
        
        # Phân tích tác động của các yếu tố
        print("\n" + "=" * 60)
        print("PHÂN TÍCH TÁC ĐỘNG CÁC YẾU TỐ")
        print("=" * 60)
        
        # Tác động của location (US vs Non-US)
        base_case = {
            'work_year': 2024,
            'experience_level': 'MI',
            'employment_type': 'FT',
            'job_title': 'Data Scientist',
            'remote_ratio': 50,
            'company_size': 'M',
            'company_location': 'US'
        }

        us_result = analyzer.predict_salary(base_case)

        # Tính trung bình cho Non-US từ nhiều quốc gia
        non_us_countries = ['DE', 'UK', 'CA', 'IN', 'VN']
        non_us_salaries = []

        for country in non_us_countries:
            base_case['company_location'] = country
            result = analyzer.predict_salary(base_case)
            non_us_salaries.append(result['predicted_salary'])

        non_us_avg = sum(non_us_salaries) / len(non_us_salaries)
        location_impact = us_result['predicted_salary'] - non_us_avg

        print(f"\nTác động của location (US vs Non-US):")
        print(f"  US: ${us_result['predicted_salary']:,.0f}")
        print(f"  Non-US (trung bình): ${non_us_avg:,.0f}")
        print(f"  Chênh lệch: ${location_impact:,.0f} ({location_impact/non_us_avg*100:.1f}%)")
        
        # Tác động của experience level
        print(f"\nTác động của experience level:")
        exp_levels = ['EN', 'MI', 'SE', 'EX']
        exp_names = ['Entry', 'Mid', 'Senior', 'Executive']
        
        base_case['company_location'] = 'US'  # Reset to US
        
        for level, name in zip(exp_levels, exp_names):
            base_case['experience_level'] = level
            result = analyzer.predict_salary(base_case)
            print(f"  {name}: ${result['predicted_salary']:,.0f}")
        
        # Tác động của remote work
        print(f"\nTác động của remote work:")
        base_case['experience_level'] = 'MI'  # Reset to Mid
        
        for remote in [0, 50, 100]:
            base_case['remote_ratio'] = remote
            result = analyzer.predict_salary(base_case)
            remote_type = "On-site" if remote == 0 else "Hybrid" if remote == 50 else "Full Remote"
            print(f"  {remote_type} ({remote}%): ${result['predicted_salary']:,.0f}")
        
        print("\n" + "=" * 60)
        print("HƯỚNG DẪN SỬ DỤNG")
        print("=" * 60)
        print("Để dự báo mức lương cho trường hợp cụ thể, sử dụng:")
        print("analyzer.predict_salary({")
        print("    'work_year': 2024,")
        print("    'experience_level': 'SE',  # EN/MI/SE/EX")
        print("    'employment_type': 'FT',   # FT/PT/CT/FL")
        print("    'job_title': 'Data Scientist',")
        print("    'remote_ratio': 100,       # 0-100")
        print("    'company_size': 'L',       # S/M/L")
        print("    'company_location': 'US'   # US/VN/DE/etc.")
        print("})")
        
    except Exception as e:
        print(f"Lỗi trong demo: {str(e)}")

if __name__ == "__main__":
    demo_salary_prediction()
