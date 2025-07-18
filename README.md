# Shukatsu Photo AI

就職活動用の証明写真を、画像解析と自然言語処理を用いて評価するアプリケーションです。顔の明るさ・笑顔の自然さ・傾きといった要素を数値化し、改善ポイントをフィードバックします。

---

## 主な機能

- 顔画像から特徴量（明るさ、笑顔スコア、顔の傾き）を自動抽出
- 基準に基づいたスコア表示と簡潔なフィードバック生成
- StreamlitによるWebアプリ形式（ローカル実行可）
- OpenAI API を活用した自然な日本語出力

---

## 使用技術

- Python 3.x
- Streamlit
- MediaPipe
- OpenCV
- OpenAI API（GPT）

---

## セットアップ手順

1. 仮想環境を作成・起動（任意）

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
