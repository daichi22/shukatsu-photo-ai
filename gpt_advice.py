from dotenv import load_dotenv
import os
from openai import OpenAI

# 環境変数からAPIキー読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_feedback(brightness, smile_score, tilt):
    prompt = (
    f"あなたは、就職活動の証明写真を専門に評価するプロのキャリアアドバイザーです。\n\n"
    f"以下のスコアは、アップロードされた証明写真の顔画像からAIによって自動算出された評価指標です。\n"
    f"あなたの役割は、これらのスコアをもとに写真の印象を総合的に分析し、好印象を与えるための具体的かつ実践的なアドバイスを生成することです。\n\n"
    f"【評価スコア】\n"
    f"- 顔の明るさ（理想: 120～180）: {brightness:.1f}\n"
    f"- 笑顔スコア（理想: 0.5以上）: {smile_score:.3f}\n"
    f"- 顔の傾き（理想: 0.015以下）: {tilt:.3f}\n\n"
    f"【アドバイス指針】\n"
    f"- 3つの数値を参考にしつつ、写真全体としてどんな印象を受けるかを踏まえてアドバイスをしてください。\n"
    f"- 数値が基準内でも、改善余地があれば明記してください（例：笑顔スコアが0.51でも「もう少し口角を上げるとさらに良い印象」といった具体的な改善案）。\n"
    f"- 数値のバランスも見てください（例：明るさが適切でも顔が傾いていると悪目立ちする等）。\n"
    f"- 文章は100〜150文字以内、前向きで丁寧な口調でお願いします。\n\n"

    f"【出力形式】\n"
    f"丁寧でポジティブなアドバイスを1段落で出力してください。"
  )



    response = client.chat.completions.create(
        model="gpt-4o-mini",  # gpt-3.5-turboでも可
        messages=[
            {"role": "system", "content": "あなたは、就職活動を支援するプロのキャリアアドバイザーです。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


