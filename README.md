# Shukatsu Photo AI

就職活動用の証明写真をAIが客観的に分析し、印象を良くするためのアドバイスを生成するWebアプリです。

---

## 🔍 概要

アップロードされた証明写真をもとに、顔の明るさ・笑顔スコア・傾きを数値化し、
OpenAIのAPIを通じて自然なフィードバック文章を自動生成します。

---

## ✨ 特徴

* MediaPipeを用いた顔検出とキーポイント取得
* 顔の明るさや口角・目線をもとにスコアを自動算出
* スコアを正規化し、顔の大きさに依存しない分析を実現
* GPTによる自然な日本語のアドバイス生成（約150字）
* StreamlitによるUIで誰でも簡単に操作可能

---

## 🛠️ 使用技術

* Python 3.10.11
* Streamlit
* MediaPipe 
* OpenAI API 

---

## 🚀 セットアップ

1. このリポジトリをクローン

```bash
git clone https://github.com/daichi22/shukatsu-photo-ai.git
cd shukatsu-photo-ai
```

2. 仮想環境の作成と有効化（任意）

```bash
python -m venv .venv
.venv\Scripts\activate （Windows）
source .venv/bin/activate （Mac/Linux）
```

3. パッケージのインストール

```bash
pip install -r requirements.txt
```

4. `.env` ファイルを作成し、OpenAI APIキーを記入

```env
OPENAI_API_KEY=sk-xxxxxx
```

5. アプリの起動

```bash
streamlit run app.py
```

---

## 📷 使い方

1. アプリを開くと画像アップロード欄が表示されます。
2. 就活写真をアップロードすると、自動で顔を検出。
3. 明るさ・笑顔・傾きのスコアを表示。
4. GPTが生成した改善アドバイスが表示されます。

---

## 📁 ディレクトリ構成（例）

```
shukatsu-photo-ai/
├── app.py
├── requirements.txt
├── .env (← gitignore済み)
├── README.md
```

---

## 🔒 プライバシーについて

アップロードされた画像は分析後すぐに破棄され、サーバーには保存されません。
分析時に特徴量のみが匿名でOpenAI APIに送信されます。

---

## 📝 ライセンス

MIT License

---

## 👤 制作意図

就活生が自分の印象を客観視し改善できるツールを作りたいと考え、本アプリを開発しました。就活の不安を少しでも軽くし、自信を持って臨めるようサポートしたいという思いを込めています。

