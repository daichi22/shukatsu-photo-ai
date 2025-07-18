# app.py
import streamlit as st
from PIL import Image
from features import extract_and_draw_features
from gpt_advice import generate_feedback

def score_color(value, min_val, max_val):
    if min_val <= value <= max_val:
        return "🟢"
    elif value < min_val:
        return "🔵"
    else:
        return "🔴"

st.set_page_config(page_title="就活写真AI添削", page_icon="📸")
st.title("📸 就活写真AI添削アプリ")
st.write("証明写真をアップロードすると、AIがその写真の印象を分析し、より良くするためのアドバイスを生成します。")

uploaded_file = st.file_uploader("画像をアップロードしてください（jpg, png）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    brightness, smile_score, tilt, landmarks, annotated_image = extract_and_draw_features(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("アップロード画像")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("AIの分析結果")
        st.image(annotated_image, caption="緑の枠が分析対象の顔領域です", use_column_width=True)

    if brightness is not None:
        st.markdown("---")
        st.subheader("📊 分析スコア")

        score_col1, score_col2, score_col3 = st.columns(3)
        score_col1.metric("顔の明るさ", f"{brightness:.1f} {score_color(brightness, 120, 180)}", "目標: 120-180")
        score_col2.metric("笑顔スコア", f"{smile_score:.3f} {score_color(smile_score, 0.5, 1.0)}", "目標: 0.5以上")
        score_col3.metric("顔の傾き", f"{tilt:.3f} {score_color(tilt, 0, 0.015)}", "目標: 0.015以下")

        st.markdown("---")
        st.subheader("💡 AIからのアドバイス")
        with st.spinner("AIがあなたの写真を丁寧に分析中です..."):
            advice = generate_feedback(brightness, smile_score, tilt)
            st.success(advice, icon="✅")
    else:
        st.warning("顔を検出できませんでした。顔がはっきりと写っている、正面からの写真をお試しください。")

st.markdown("---")
st.caption("※ アップロードされた画像は、分析後すぐに破棄され、サーバーには保存されません。特徴量データのみが匿名でOpenAIのAPIに送信されます。")
