import streamlit as st
from PIL import Image
import yaml
import plotly.graph_objects as go
import io
from features import FaceAnalyzer
from gpt_advice import generate_summary_advice, generate_detailed_advice

# --- 設定ファイルの読み込み ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.set_page_config(page_title="エラー", page_icon="🚨")
    st.error("重大なエラー: `config.yaml` が見つかりません。")
    st.info("プロジェクトのルートフォルダに `config.yaml` ファイルを配置してください。")
    st.stop()

# --- キャッシュ機能 ---
@st.cache_resource
def get_face_analyzer():
    return FaceAnalyzer()

@st.cache_data
def run_analysis(_analyzer, image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return _analyzer.analyze(image)

# --- UI描画関数 ---
def create_radar_chart(results):
    categories = [r.label for r in results.values()]
    scores = [r.normalized_score for r in results.values()]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]], # 閉じるために始点を追加
        theta=categories + [categories[0]],
        fill='toself',
        name='あなたのスコア'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=350, margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# --- メインUI ---
st.set_page_config(page_title="就活写真添削アプリ", page_icon="📸")
st.title("📸 就活写真添削アプリ")
st.write("表情や構図、写真の品質まで多角的に分析し、具体的な改善アクションを提案します。")

analyzer = get_face_analyzer()
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image_bytes = uploaded_file.getvalue()
        analysis_data = run_analysis(analyzer, image_bytes)
        image = Image.open(io.BytesIO(image_bytes))

        if analysis_data:
            results = analysis_data["results"]
            final_score = analysis_data["final_score"]
            
            st.subheader("💯 総合スコア")
            score_rank = "S" if final_score >= 90 else "A" if final_score >= 80 else "B" if final_score >= 60 else "C"
            st.metric(label="あなたの写真の完成度", value=f"{final_score:.1f} 点", delta=f"{score_rank} ランク")
            
            st.subheader("📊 総合評価（レーダーチャート）")
            st.plotly_chart(create_radar_chart(results), use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("アップロード画像")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("AIの分析結果 (可視化)")
                st.image(analysis_data["annotated_image"], use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 詳細スコアと個別アドバイス")
            
            # ▼▼▼▼▼ UIを2つのグループ、3列x2段のレイアウトに修正 ▼▼▼▼▼
            # 顔の評価
            st.markdown("##### 👩 顔の評価")
            c1, c2, c3 = st.columns(3)
            face_keys = ['brightness', 'smile', 'tilt']
            columns = [c1, c2, c3]
            for i, key in enumerate(face_keys):
                with columns[i]:
                    res = results[key]
                    delta_text = "OK" if res.status == "OK" else ("改善要" if res.status == "WARN" else "深刻な問題")
                    st.metric(label=res.label, value=f"{res.value:.2f}", delta=delta_text, delta_color=("normal" if res.status == "OK" else "inverse"))
                    if st.button("アドバイス", key=f"advice_{key}", use_container_width=True):
                        st.session_state.selected_advice = res.message_key

            # 構図と品質の評価
            st.markdown("##### 🖼️ 構図と品質の評価")
            c4, c5, c6 = st.columns(3)
            composition_keys = ['face_ratio', 'center_offset', 'sharpness']
            columns = [c4, c5, c6]
            for i, key in enumerate(composition_keys):
                with columns[i]:
                    res = results[key]
                    if key == 'sharpness':
                         delta_text = "OK" if res.status == "OK" else "ピンボケの可能性"
                    else:
                        delta_text = "OK" if res.status == "OK" else "改善要"
                    st.metric(label=res.label, value=f"{res.value:.2f}", delta=delta_text, delta_color=("normal" if res.status == "OK" else "inverse"))
                    if st.button("アドバイス", key=f"advice_{key}", use_container_width=True):
                        st.session_state.selected_advice = res.message_key
            # ▲▲▲▲▲ UI修正ここまで ▲▲▲▲▲

            # 詳細アドバイスの表示ロジック
            if 'selected_advice' in st.session_state and st.session_state.selected_advice:
                key = st.session_state.selected_advice
                target_label = ""
                for res_key, res_val in results.items():
                    if res_val.message_key == key:
                        target_label = res_val.label
                        break
                
                with st.spinner(f"「{target_label}」についてAIがアドバイス中..."):
                    detailed_advice = generate_detailed_advice(key)
                    st.info(detailed_advice, icon="💡")
                if st.button("閉じる", key=f"close_{key}"):
                    st.session_state.selected_advice = None
            
            st.markdown("---")
            st.subheader("💡 総合的なアドバイス")
            problem_keys = [r.message_key for r in results.values() if r.status != "OK"]
            
            with st.spinner("AIが改善アクションを考案中です..."):
                summary_advice = generate_summary_advice(problem_keys, final_score)
                
                if any(r.status == "ERROR" for r in results.values()):
                    st.error(summary_advice, icon="🚨")
                elif problem_keys:
                    st.warning(summary_advice, icon="⚠️")
                else:
                    st.success(summary_advice, icon="✨")
        else:
            st.warning("顔を検出できませんでした。")

    except Exception as e:
        st.error(f"分析中に予期せぬエラーが発生しました。別の画像で試してください。\n詳細: {e}")