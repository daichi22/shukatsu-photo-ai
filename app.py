# app.py (修正箇所)
import streamlit as st
from PIL import Image
# 改善版のファイルをインポート
from features_improved import FaceAnalyzer
from gpt_advice_improved import generate_feedback

# FaceAnalyzerをインスタンス化
analyzer = FaceAnalyzer()

# (score_color関数、st.titleなどは変更なし)

uploaded_file = st.file_uploader("画像をアップロードしてください（jpg, png）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # analyzer.analyzeを呼び出す
    analysis_data = analyzer.analyze(image)

    if analysis_data:
        # 辞書から各値を取得
        brightness = analysis_data["brightness"]
        smile_score = analysis_data["smile_score"]
        smile_metrics = analysis_data["smile_metrics"]
        tilt = analysis_data["tilt_score"]
        annotated_image = analysis_data["annotated_image"]

        # (UI表示部分は変更なし)
        # ...

        st.markdown("---")
        st.subheader("💡 AIからのアドバイス")
        with st.spinner("AIがあなたの写真を丁寧に分析中です..."):
            # generate_feedbackに関数の引数を渡す
            advice = generate_feedback(brightness, smile_score, smile_metrics, tilt)
            st.success(advice, icon="✅")
    else:
        st.warning("顔を検出できませんでした。顔がはっきりと写っている、正面からの写真をお試しください。")

# (フッター部分は変更なし)