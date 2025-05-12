import streamlit as st
from prepare_file import prepare_uploaded_file
from search_similar import search_top_k
import os

st.set_page_config(page_title="Tìm kiếm giọng nói", layout="centered")

st.title("🔍 Tìm kiếm giọng nói phụ nữ")
st.write("Tải lên một file WAV để tìm 3 giọng giống nhất trong hệ thống.")

uploaded_file = st.file_uploader("📂 Chọn file âm thanh (.wav)", type=['wav'])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav", format="audio/wav")
    prepared_path = prepare_uploaded_file("temp.wav")

    with st.spinner("🔎 Đang phân tích và tìm kiếm..."):
        results = search_top_k(prepared_path, top_k=3)

        if not results:
            st.error("Không thể trích đặc trưng hoặc tìm kiếm không thành công.")
        else:
            st.subheader("📄 Top 3 file giống nhất:")
            for rank, (fname, score) in enumerate(results, start=1):
                file_path = os.path.join("dataset/train", fname)
                st.markdown(f"**{rank}. {fname}** — Similarity: `{score*100:.2f}%`")
                with open(file_path, 'rb') as f:
                    st.audio(f.read(), format='audio/wav')
