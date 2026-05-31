import streamlit as st
import pandas as pd
import plotly.express as px

from utils.predict import predict_phobert
from utils.features import extract_features
from utils.preprocessing import clean_text

st.set_page_config(
    page_title="Fake News Detection",
    layout="wide"
)

st.title("Vietnamese Fake News Detection")
st.write("Demo phát hiện nội dung báo chí đáng tin cậy / không đáng tin cậy bằng PhoBERT.")


tab1, tab2, tab3, tab4 = st.tabs([
    "Dự đoán",
    "Phân tích đặc trưng",
    "Mẫu thử",
    "So sánh hiệu suất các mô hình"
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
        "Tin thật về vaccine": {
            "text": "WHO xác nhận vaccine COVID-19 an toàn và hiệu quả khi được sử dụng đúng theo hướng dẫn của cơ quan y tế.",
            "expected": "REAL",
            "note": "Thông tin mang tính xác nhận từ tổ chức y tế."
        },
        "Tin giả sức khỏe": {
            "text": "Uống nước chanh mỗi sáng có thể chữa khỏi ung thư 100%, hãy chia sẻ ngay trước khi bị xoá!",
            "expected": "FAKE",
            "note": "Nội dung có dấu hiệu phóng đại, tuyệt đối hóa và kêu gọi chia sẻ."
        },
        "Tin lừa đảo": {
            "text": "Khẩn cấp! Ngân hàng yêu cầu bạn bấm vào link lạ để xác minh tài khoản nếu không sẽ bị khóa ngay.",
            "expected": "FAKE",
            "note": "Nội dung có dấu hiệu lừa đảo, tạo cảm giác khẩn cấp và yêu cầu bấm link."
        },
        "Tin xã hội trung tính": {
            "text": "UBND TP.HCM thông báo điều chỉnh lịch làm việc của một số tuyến xe buýt trong dịp lễ nhằm giảm ùn tắc giao thông.",
            "expected": "REAL",
            "note": "Nội dung hành chính, ngôn ngữ trung tính, không có dấu hiệu giật tít."
        },
        "Tin giật tít": {
            "text": "Sốc! Một loại thuốc bí mật có thể giúp khỏi mọi bệnh chỉ sau 3 ngày, chuyên gia không muốn bạn biết điều này.",
            "expected": "FAKE",
            "note": "Nội dung có nhiều từ khóa clickbait như sốc, bí mật, khỏi mọi bệnh."
        }
    }

    selected = st.selectbox(
        "Chọn mẫu",
        list(examples.keys())
    )

    sample = examples[selected]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.text_area(
            "Nội dung mẫu",
            sample["text"],
            height=160
        )

    with col2:
        st.metric("Nhãn kỳ vọng", sample["expected"])
        st.info(sample["note"])

    if st.button("Dự đoán mẫu"):
        text = clean_text(sample["text"])

        with st.spinner("Đang dự đoán bằng PhoBERT..."):
            result = predict_phobert(text)

        prediction = result["prediction"]
        confidence = result["confidence"]
        real_prob = result["real_prob"]
        fake_prob = result["fake_prob"]

        st.write("### Kết quả dự đoán")

        col1, col2, col3 = st.columns(3)

        col1.metric("Expected", sample["expected"])
        col2.metric("Predicted", prediction)
        col3.metric("Confidence", f"{confidence * 100:.2f}%")

        if prediction == sample["expected"]:
            st.success("Dự đoán đúng so với nhãn kỳ vọng.")
        else:
            st.error("Dự đoán khác với nhãn kỳ vọng. Đây là trường hợp nên phân tích lỗi.")

        prob_df = pd.DataFrame({
            "Label": ["REAL", "FAKE"],
            "Probability": [real_prob, fake_prob]
        })

        st.write("### Xác suất dự đoán")

        st.bar_chart(
            prob_df,
            x="Label",
            y="Probability"
        )

        with st.expander("Xem chi tiết output"):
            st.json(result)

with tab4:
    st.subheader("So sánh hiệu suất các mô hình")

    st.write(
        """
        Phần demo trực tiếp sử dụng mô hình **PhoBERT**.  
        Các mô hình còn lại được trình bày để so sánh hiệu quả giữa nhiều hướng tiếp cận khác nhau.
        """
    )

    model_data = pd.DataFrame({
        "Model": [
            "TF-IDF + LR",
            "TF-IDF + SVM",
            "TextCNN",
            "BiLSTM",
            "PhoBERT",
            "LLaMA QLoRA",
            "Gemma QLoRA",
            "Zero-shot Prompting",
            "Few-shot Prompting"
        ],
        "Type": [
            "Machine Learning",
            "Machine Learning",
            "Deep Learning",
            "Deep Learning",
            "Transformer",
            "LLM Fine-tuning",
            "LLM Fine-tuning",
            "Prompting",
            "Prompting"
        ],
        "Accuracy": [
            0.9342,
            0.9498,
            0.9458,
            0.9315,
            0.9749,
            0.9758,
            0.9557,
            0.6480,
            0.6480
        ],
        "Precision": [
            0.9250,
            0.9400,
            0.9457,
            0.9227,
            0.9706,
            0.9700,
            0.9511,
            0.6480,
            0.6480
        ],
        "Recall": [
            0.9327,
            0.9500,
            0.9458,
            0.8793,
            0.9748,
            0.9700,
            0.9519,
            1.0000,
            1.0000
        ],
        "F1-score": [
            0.9286,
            0.9500,
            0.9457,
            0.9005,
            0.9727,
            0.9735,
            0.9557,
            0.7864,
            0.7864
        ],
        "ROC-AUC": [
            0.9771,
            0.9875,
            0.9855,
            0.9767,
            0.9963,
            0.9965,
            0.9875,
            0.5000,
            0.5000
        ]
    })

    st.dataframe(model_data, use_container_width=True)

    best_model = model_data.sort_values("Accuracy", ascending=False).iloc[0]

    st.success(
        f"Mô hình có Accuracy cao nhất: **{best_model['Model']}** "
        f"với Accuracy = **{best_model['Accuracy'] * 100:.2f}%**"
    )

    st.write("### Biểu đồ so sánh các giá trị độ đo của các mô hình")

    selected_metric = st.selectbox(
        "Chọn chỉ số muốn so sánh",
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    )

    fig_metric = px.bar(
        model_data.sort_values(selected_metric, ascending=True),
        x=selected_metric,
        y="Model",
        color="Type",
        orientation="h",
        text=selected_metric,
        title=f"So sánh {selected_metric} giữa các mô hình"
    )

    st.plotly_chart(fig_metric, use_container_width=True, key="metric_chart")