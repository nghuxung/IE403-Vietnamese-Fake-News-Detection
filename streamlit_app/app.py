import streamlit as st
import pandas as pd

from utils.predict import predict_phobert
from utils.features import extract_features
from utils.preprocessing import clean_text

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Vietnamese Fake News Detection")
st.write("Demo phát hiện nội dung báo chí đáng tin cậy / không đáng tin cậy bằng PhoBERT.")


tab1, tab2, tab3 = st.tabs([
    "🔍 Dự đoán",
    "📊 Phân tích đặc trưng",
    "🧪 Mẫu thử"
])


with tab1:
    st.subheader("Nhập nội dung cần kiểm tra")

    title = st.text_input("Tiêu đề bài viết")

    content = st.text_area(
        "Nội dung bài viết",
        height=220
    )

    if st.button("Dự đoán", type="primary"):
        full_text = clean_text(title + " " + content)

        if full_text.strip() == "":
            st.warning("Vui lòng nhập tiêu đề hoặc nội dung bài viết.")
        else:
            with st.spinner("Đang phân tích bằng PhoBERT..."):
                result = predict_phobert(full_text)

            prediction = result["prediction"]
            confidence = result["confidence"]
            real_prob = result["real_prob"]
            fake_prob = result["fake_prob"]

            if prediction == "FAKE":
                st.error(f"Kết quả: FAKE NEWS")
            else:
                st.success(f"Kết quả: REAL NEWS")

            col1, col2, col3 = st.columns(3)

            col1.metric("Prediction", prediction)
            col2.metric("Confidence", f"{confidence * 100:.2f}%")
            col3.metric("Model", "PhoBERT")

            st.write("### Xác suất dự đoán")

            prob_df = pd.DataFrame({
                "Label": ["REAL", "FAKE"],
                "Probability": [real_prob, fake_prob]
            })

            st.bar_chart(
                prob_df,
                x="Label",
                y="Probability"
            )

            st.write(f"REAL: **{real_prob * 100:.2f}%**")
            st.write(f"FAKE: **{fake_prob * 100:.2f}%**")


with tab2:
    st.subheader("Phân tích đặc trưng văn bản")

    sample_text = st.text_area(
        "Nhập văn bản để phân tích đặc trưng",
        height=220,
        key="feature_text"
    )

    if st.button("Phân tích đặc trưng"):
        cleaned = clean_text(sample_text)

        if cleaned.strip() == "":
            st.warning("Vui lòng nhập văn bản.")
        else:
            features = extract_features("", cleaned)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Text length", features["text_length"])
            col2.metric("Clickbait score", features["clickbait_score"])
            col3.metric("Dấu !", features["exclamation_count"])
            col4.metric("Dấu ?", features["question_count"])

            st.write("Từ/cụm từ nghi vấn")

            matched_words = features.get("matched_clickbait_words", [])
            if len(matched_words) > 0:
                 for word in matched_words:
                     st.warning(f"Phát hiện: {word}")
            else:
                st.success("Không phát hiện từ khóa clickbait rõ ràng.")

            st.json(features)


with tab3:
    st.subheader("Mẫu thử nhanh")

    examples = {
        "Tin thật về vaccine": "WHO xác nhận vaccine COVID-19 an toàn và hiệu quả khi được sử dụng đúng theo hướng dẫn của cơ quan y tế.",
        "Tin giả sức khỏe": "Uống nước chanh mỗi sáng có thể chữa khỏi ung thư 100%, hãy chia sẻ ngay trước khi bị xoá!",
        "Tin lừa đảo": "Khẩn cấp! Ngân hàng yêu cầu bạn bấm vào link lạ để xác minh tài khoản nếu không sẽ bị khóa ngay."
    }

    selected = st.selectbox(
        "Chọn mẫu",
        list(examples.keys())
    )

    st.text_area(
        "Nội dung mẫu",
        examples[selected],
        height=160
    )

    if st.button("Dự đoán mẫu"):
        text = clean_text(examples[selected])

        with st.spinner("Đang dự đoán..."):
            result = predict_phobert(text)

        if result["prediction"] == "FAKE":
            st.error(f"FAKE NEWS - {result['confidence'] * 100:.2f}%")
        else:
            st.success(f"REAL NEWS - {result['confidence'] * 100:.2f}%")

        st.write(result)