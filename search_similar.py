
import os
import numpy as np
from extract_features import extract
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="female_voice_db")

def load_or_compute_zscore_stats(data_folder='dataset/train'):
    stats_path = "zscore_stats.npz"
    if os.path.exists(stats_path):
        zscore_data = np.load(stats_path)
        return zscore_data["mean"], zscore_data["std"]
    else:
        print("⚠️ Không tìm thấy zscore_stats.npz. Đang tính toán lại...")
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
        np.savez(stats_path, mean=global_mean, std=global_std)
        print("✅ Đã lưu global_mean và global_std vào zscore_stats.npz")
        return global_mean, global_std

global_mean, global_std = load_or_compute_zscore_stats()

def add_dataset_to_chromadb(data_folder='dataset/train'):
    added = 0
    for file in sorted(os.listdir(data_folder)):
        if file.lower().endswith('.wav'):
            file_path = os.path.join(data_folder, file)
            vec = extract(file_path, global_mean=global_mean, global_std=global_std)
            if vec is not None:
                collection.upsert(
                    ids=[file],
                    embeddings=[vec.tolist()],
                    metadatas=[{"filename": file}],
                    documents=[file]
                )
                added += 1

def search_top_k(file_path, top_k=3):
    query_vec = extract(file_path, global_mean=global_mean, global_std=global_std)
    if query_vec is None:
        return []

    results_all = collection.get(include=["embeddings", "documents"])
    if not results_all or "embeddings" not in results_all:
        return []

    db_vectors = np.array(results_all["embeddings"])
    doc_names = results_all["documents"]

    sims = cosine_similarity(query_vec.reshape(1, -1), db_vectors)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = [(doc_names[i], sims[i]) for i in top_indices]
    return results
