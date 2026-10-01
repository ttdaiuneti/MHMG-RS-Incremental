import numpy as np
import pandas as pd
import time
import os
import glob
import re
from sklearn.preprocessing import MinMaxScaler

def natural_sort_key(s):
    """Hỗ trợ sắp xếp tên tệp theo thứ tự tự nhiên (1, 2, 10 thay vì 1, 10, 2)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

class MHTG_Reducer:
    """Lớp triển khai thuật toán Rút gọn thuộc tính dựa trên Hạt Tôpô Đồng nhất Cực đại"""
    
    def __init__(self):
        self.reduct = []
        self.gamma_best = 0

    def _calculate_distances(self, X):
        """Tính ma trận khoảng cách Euclidean tối ưu bằng vector hóa"""
        sum_X = np.sum(np.square(X), axis=1)
        dist_matrix = np.sqrt(np.maximum(sum_X[:, np.newaxis] + sum_X - 2 * np.dot(X, X.T), 0))
        return dist_matrix

    def _calculate_gamma(self, X, y, max_potential_energy):
        """Tính độ phụ thuộc Gamma dựa trên Log-Sum (Eq. 3)"""
        n = X.shape[0]
        if X.shape[1] == 0: return 0
        
        dist_matrix = self._calculate_distances(X)
        
        # 1. Xác định Bán kính an toàn delta_B(x_i)
        deltas = np.zeros(n)
        for i in range(n):
            enemy_mask = (y != y[i])
            enemies_dist = dist_matrix[i][enemy_mask]
            if len(enemies_dist) > 0:
                deltas[i] = np.min(enemies_dist)
            else:
                deltas[i] = np.inf # Không có kẻ thù

        # 2. Xác định kích thước hạt MHTG và tính năng lượng thu được
        # G_B(x_i) = {x_j | dist(x_i, x_j) < delta_i}
        granular_sizes = np.array([np.sum(dist_matrix[i] < deltas[i]) for i in range(n)])
        captured_energy = np.sum(np.log2(granular_sizes + 1))

        return captured_energy / max_potential_energy

    def fit_reduction(self, X, y):
        """Thực hiện thuật toán MHTG-RS với cơ chế dừng tự động (Anti-monotonicity)"""
        start_time = time.time()
        n_samples, n_features = X.shape
        
        # Chuẩn hóa dữ liệu về [0, 1] để khoảng cách Euclidean có ý nghĩa
        X = MinMaxScaler().fit_transform(X)
        
        self.reduct = []
        self.gamma_best = 0
        
        # Tính mẫu số Gamma (Năng lượng tiềm năng tối đa) - Cố định cho mỗi dataset
        unique_labels, counts = np.unique(y, return_counts=True)
        class_size_map = dict(zip(unique_labels, counts))
        max_potential_energy = np.sum([np.log2(class_size_map[label] + 1) for label in y])
        
        remaining_features = list(range(n_features))
        
        while len(remaining_features) > 0:
            gamma_local_max = -1
            best_feature = None
            
            # Forward Selection: Tìm thuộc tính làm tăng Gamma nhiều nhất
            for f in remaining_features:
                current_subset = self.reduct + [f]
                gamma_current = round(self._calculate_gamma(X[:, current_subset], y, max_potential_energy),2)
                
                if gamma_current > gamma_local_max:
                    gamma_local_max = gamma_current
                    best_feature = f
                    
            
            # KIỂM TRA TÍNH PHẢN ĐƠN ĐIỆU (CƠ CHẾ DỪNG TỰ ĐỘNG)
            # Chỉ thêm thuộc tính nếu nó thực sự làm tăng năng lượng hạt (giãn nở hạt)
            if gamma_local_max > self.gamma_best:
                self.reduct.append(best_feature)
                remaining_features.remove(best_feature)
                self.gamma_best = gamma_local_max
            else:
                # Ngừng khi gặp thuộc tính nhiễu hoặc đạt đỉnh tôpô (Shrinkage detected)
                break
                
        duration = time.time() - start_time
        return self.reduct, duration, self.gamma_best

def process_batch_MHTG(folder_path, output_file):
    """Hàm xử lý lô các bộ dữ liệu theo mẫu yêu cầu"""
    
    # Lấy danh sách tệp .csv và sắp xếp tự nhiên
    dataset_list = sorted(glob.glob(os.path.join(folder_path, "*.csv")), key=natural_sort_key)
    
    if not dataset_list:
        print(f"Không tìm thấy tệp .csv nào trong thư mục: {folder_path}")
        return

    results = []
    reducer = MHTG_Reducer()
    
    print(f"--- BẮT ĐẦU THỰC NGHIỆM MHTG (Parameter-free) ---")
    
    for file_path in dataset_list:
        dataset_name = os.path.basename(file_path).split('.')[0]
        print(f" - Đang xử lý: {dataset_name}...", end=" ", flush=True)
        
        try:
            # Đọc dữ liệu
            df = pd.read_csv(file_path)
            
            # Xử lý dữ liệu (loại bỏ cột không cần thiết nếu có, mặc định cột cuối là nhãn)
            X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
            y = df.iloc[:, -1].values
            
            # Thực hiện rút gọn
            reduct, duration, gamma = reducer.fit_reduction(X, y)
            
            results.append({
                'Dataset': dataset_name,
                '|U|': X.shape[0],
                '|C|': X.shape[1],
                '|R|': len(reduct),
                'Gamma': round(gamma, 4),
                # Chuyển chỉ số về 1-based (i+1) để khớp với danh sách thuộc tính trong bài báo
                'R': "{" + ", ".join(map(str, sorted([i + 1 for i in reduct]))) + "}",
                'Time(s)': round(duration, 4)
            })
            print(f"Xong! (|R|={len(reduct)})")
            
        except Exception as e:
            print(f"Lỗi tại {dataset_name}: {e}")

    if results:
        df_res = pd.DataFrame(results)
        
        # Tính toán hàng Average
        avg_row = {
            'Dataset': 'Average',
            '|U|': '-',
            '|C|': '-',
            '|R|': round(df_res['|R|'].mean(), 2),
            'Gamma': round(df_res['Gamma'].mean(), 4),
            'R': '-',
            'Time(s)': round(df_res['Time(s)'].mean(), 4)
        }
        
        df_res = pd.concat([df_res, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Xuất ra Excel
        df_res.to_excel(output_file, index=False)
        print(f"\n--- TẤT CẢ HOÀN THÀNH ---")
        print(f"Kết quả lưu tại: {output_file}")
        print(df_res.tail())

if __name__ == "__main__":
    # Cấu hình đường dẫn
    FOLDER_INPUT = "12-datasets" # Đảm bảo thư mục này chứa các file csv của bạn
    FILE_OUTPUT = "MHTG.xlsx"
    
    if not os.path.exists(FOLDER_INPUT):
        os.makedirs(FOLDER_INPUT)
        print(f"Vui lòng tạo thư mục '{FOLDER_INPUT}' và chép các tệp .csv vào đó.")
    else:
        process_batch_MHTG(FOLDER_INPUT, FILE_OUTPUT)