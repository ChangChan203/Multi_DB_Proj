import os
import numpy as np
from extract_features import extract

def compute_global_stats(data_folder='dataset/train'):
    all_features = []
    for file in sorted(os.listdir(data_folder)):
        if file.lower().endswith('.wav'):
            file_path = os.path.join(data_folder, file)
            vec = extract(file_path)
            if vec is not None:
                all_features.append(vec)

    all_features = np.vstack(all_features)
    global_mean = np.mean(all_features, axis=0)
    global_std = np.std(all_features, axis=0)
    np.savez("zscore_stats.npz", mean=global_mean, std=global_std)
    print("✅ Đã lưu global_mean và global_std vào zscore_stats.npz")

if __name__ == "__main__":
    compute_global_stats()