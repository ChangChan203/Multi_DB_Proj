import numpy as np
import librosa
import traceback

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def extract(file_path, sr=16000, n_mfcc=13, frame_length_ms=25, hop_length_ms=10, window='hamming', global_mean=None, global_std=None):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        y = pre_emphasis(y, coeff=0.97)

        n_fft = int(sr * frame_length_ms / 1000)
        hop_length = int(sr * hop_length_ms / 1000)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                    n_fft=n_fft, hop_length=hop_length,
                                    win_length=n_fft, window=window)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_39 = np.vstack([mfcc, delta, delta2])

        mean = np.mean(mfcc_39, axis=1)
        std = np.std(mfcc_39, axis=1)
        feature_vector = np.concatenate([mean, std])  # 78 chiều

        # ✅ Z-score normalization
        if global_mean is not None and global_std is not None:
            feature_vector = (feature_vector - global_mean) / global_std

        return feature_vector
    except Exception as e:
        print(f"[Lỗi extract()] File: {file_path}")
        print(f"Lỗi: {str(e)}")
        traceback.print_exc()
        return None
