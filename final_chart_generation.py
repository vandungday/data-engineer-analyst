#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script cuối cùng để tạo tất cả biểu đồ với font đã sửa
"""

from data_science_salary_analysis import DataScienceSalaryAnalysis

def main():
    """
    Tạo tất cả biểu đồ với font DejaVu Sans
    """
    print("Tạo tất cả biểu đồ với font đã sửa...")
    
    # Đường dẫn đến file dữ liệu
    data_path = "data-salary.csv"
    
    try:
        # Tạo đối tượng phân tích
        analyzer = DataScienceSalaryAnalysis(data_path)
        
        # Chạy phân tích hoàn chỉnh
        print("Đang chạy phân tích hoàn chỉnh...")
        results = analyzer.run_complete_analysis()
        
        print("\n" + "="*60)
        print("HOÀN THÀNH TẠO TẤT CẢ BIỂU ĐỒ!")
        print("="*60)
        print("Tất cả biểu đồ đã được lưu trong folder 'images' với:")
        print("- Font: DejaVu Sans (hỗ trợ tiếng Việt)")
        print("- DPI: 300 (chất lượng cao)")
        print("- Nền trắng, không lỗi hiển thị")
        print("\nDanh sách biểu đồ:")
        print("1. 01_data_exploration.png - Khám phá dữ liệu")
        print("2. 02_model_results.png - Kết quả mô hình")
        print("3. 03_correlation_matrix.png - Ma trận tương quan")
        print("4. 04_salary_by_job.png - Lương theo job title")
        print("5. 05_experience_impact.png - Tác động kinh nghiệm")
        print("6. 06_company_size_salary.png - Lương theo quy mô công ty")
        print("7. 07_remote_impact.png - Tác động remote work")
        print("8. 08_residuals_analysis.png - Phân tích residuals")
        print("9. 09_feature_importance.png - Tầm quan trọng features")
        
        print(f"\nKết quả phân tích:")
        print(f"- Phương pháp: {results['method']}")
        print(f"- Test R²: {results['test_r2']:.4f}")
        print(f"- Test MAE: ${results['test_mae']:,.0f}")
        print(f"- Số job types: {results['job_types_analyzed']}")
        
    except Exception as e:
        print(f"Lỗi trong quá trình tạo biểu đồ: {str(e)}")

if __name__ == "__main__":
    main()
