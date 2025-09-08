import streamlit as st
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt

# --- Load model & vectorizer ---
model = joblib.load("nb_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# --- Hàm làm sạch văn bản ---
def clean_text_basic(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # bỏ link
    text = re.sub(r"[\r\n\t]", " ", text)           # bỏ xuống dòng
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)  # bỏ ký tự đặc biệt
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Mapping nhãn ---
label_map = {
    -1: "Tiêu cực 😞",
     0: "Trung tính 😐",
     1: "Tích cực 😀"
}

# --- Hàm dự đoán ---
def predict_comment(comment):
    text = clean_text_basic(comment)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    probas = model.predict_proba(vec)[0]
    max_prob = np.max(probas)
    return label_map[pred], max_prob, probas

# --- Giao diện Streamlit ---
st.title("💬 Dự đoán cảm xúc bình luận sản phẩm thời trang Shopee")
# Layout 2 cột
col1, col2 = st.columns([2, 1])

with col1:
    comment = st.text_area("✍️ Nhập bình luận của bạn:")

    if st.button("🚀 Dự đoán"):
        if comment.strip() != "":
            label, prob, probas = predict_comment(comment)
            st.subheader("👉 Kết quả:")
            st.success(f"{label} (Độ tin cậy: {prob:.2f})")
            # Hiển thị text xác suất
            st.markdown("### 📌 Xác suất từng nhãn:")
            st.write(f"- Tiêu cực 😞: {probas[0]:.2f}")
            st.write(f"- Trung tính 😐: {probas[1]:.2f}")
            st.write(f"- Tích cực 😀: {probas[2]:.2f}")

            # Vẽ biểu đồ bên cột phải
            with col2:
                labels = [" 😞", " 😐", " 😀"]
                colors = ["#FF6B6B", "#FFD93D", "#6BCB77"]  # đỏ, vàng, xanh lá

                fig, ax = plt.subplots(figsize=(4,5))
                ax.bar(labels, probas, color=colors)
                ax.set_ylim(0,1)
                ax.set_ylabel("Xác suất")
                ax.set_title("Biểu đồ xác suất")

                # Thêm số trên cột
                for i, v in enumerate(probas):
                    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

                st.pyplot(fig)
        else:
            st.warning("⚠️ Vui lòng nhập bình luận trước khi dự đoán!")
