from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_feedback(brightness, smile_score, smile_metrics, tilt):
    # 笑顔スコアの内訳を説明する文章を生成
    smile_detail = (
        f"総合的な笑顔スコアは {smile_score:.3f} です。これは口元の動き（スコア: {smile_metrics['mouth']:.3f}）と、"
        f"目元の表情（スコア: {smile_metrics['eyes']:.3f}）から総合的に判断されています。"
    )

    prompt = (
        f"あなたは、就職活動の証明写真を専門に評価するプロのキャリアアドバイザーです。\n\n"
        f"以下のスコアは、証明写真からAIが算出した評価指標です。これらの指標と、特に笑顔スコアの詳細な内訳を元に、具体的で実践的なアドバイスを生成してください。\n\n"
        f"【評価スコア】\n"
        f"- 顔の明るさ（理想: 120～180）: {brightness:.1f}\n"
        f"- 笑顔スコア（総合）（理想: 0.5以上）: {smile_score:.3f}\n"
        f"- 顔の傾き（理想: 0.015以下）: {tilt:.3f}\n\n"
        f"【笑顔スコアの詳細分析】\n"
        f"{smile_detail}\n\n"
        f"【アドバイス指針】\n"
        f"- 笑顔の詳細分析を最重要視してください。例えば、「口角は上がっていますが、目元が少し硬い印象です。リラックスして目を少し細める意識をすると、より自然な笑顔になります」のように、口と目の両方に言及してください。\n"
        f"- 全ての数値を考慮し、写真全体の印象について100〜150文字以内で、前向きかつ丁寧な口調でアドバイスを生成してください。\n\n"
        f"【出力形式】\n"
        f"丁寧でポジティブなアドバイスを1段落で出力してください。"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは、就職活動を支援するプロのキャリアアドバイザーです。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


