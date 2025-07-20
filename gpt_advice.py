from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 総合アドバイス用の状況説明マッピング
SUMMARY_PROMPTS = {
    "SHARPNESS": "写真がピンボケ気味で不鮮明です。",
    "BRIGHTNESS": "顔の明るさが不適切（暗すぎるか明るすぎる）です。",
    "SMILE": "笑顔が少し硬い印象です。",
    "FRINGE": "前髪が眉毛にかかっている可能性があり、表情が暗く見えがちです。",
    "FACE_RATIO": "写真に占める顔の割合が適切ではありません。",
    "CENTER_OFFSET": "顔が写真の中央からズレています。",
    "TILT": "顔が少し傾いています。"
}

# 詳細アドバイス用の深掘り指示マッピング
DETAILED_PROMPTS = {
    "BRIGHTNESS": "「顔の明るさが不適切」と診断された就活生に、適切な照明環境を作るための具体的なアクション（例: リングライトの使用、窓際での撮影、レフ板の代用）を2〜3つ提案してください。",
    "SMILE": "「笑顔スコアが低い」と診断された就活生に、自然で自信のある表情を作るための具体的な表情筋トレーニングや意識すべき点を3つのステップで提案してください。",
    "TILT": "「顔の傾き」を指摘された就活生に、まっすぐな姿勢を保つための撮影時の具体的なコツ（例: 顎の引き方、カメラとの位置関係）を2つ提案してください。",
    "FACE_RATIO": "「顔の比率」が不適切と診断された就活生に、証明写真における適切な顔の大きさの目安と、撮影時にそれを実現するためのカメラとの距離の調整方法を説明してください。",
    "CENTER_OFFSET": "「顔の中心位置」のズレを指摘された就活生に、カメラのガイドライン機能を使うなど、簡単に中央で撮影できる具体的な方法を2つ提案してください。",
    "FRINGE": "「前髪」の問題を指摘された就活生に、清潔感を出し、表情を明るく見せるための髪型の整え方について、具体的なアクションを2つ提案してください。",
    "SHARPNESS": "「写真がピンボケしている」と診断された就活生に、手ブレを防いでシャープな写真を撮るための具体的なアクション（例: 三脚やセルフタイマーの使用、撮影設定の見直し）を3つ提案してください。"
}

def generate_summary_advice(problem_keys, final_score):
    if not problem_keys:
        prompt_core = f"総合スコアは{final_score:.0f}点でした。全ての項目で素晴らしい結果です。自信を持って就職活動に臨んでください、というポジティブな激励のメッセージを生成してください。"
    else:
        descriptions = [SUMMARY_PROMPTS[key] for key in problem_keys if key in SUMMARY_PROMPTS]
        prompt_core = (
            f"総合スコアは{final_score:.0f}点でした。以下の【写真の状況】を踏まえ、改善のための**最も重要なポイント**を強調しつつ、励ますようなアドバイスを一つの自然な段落としてまとめてください。\n\n"
            "【写真の状況】\n" + "\n".join(descriptions)
        )
    final_prompt = (
        "あなたはプロの就活キャリアアドバイザーです。証明写真の分析結果を元に、就活生を励ますフィードバックをしてください。"
        "丁寧かつ前向きな口調で、150文字程度でお願いします。\n\n"
        f"{prompt_core}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": final_prompt}],
            temperature=0.7, max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"アドバイスの生成中にエラーが発生しました: {e}"

def generate_detailed_advice(problem_key):
    if problem_key not in DETAILED_PROMPTS:
        return "この項目に関する詳細なアドバイスはありません。"
    
    specific_instruction = DETAILED_PROMPTS[problem_key]
    final_prompt = (
        "あなたはプロの就活キャリアアドバイザーです。就活生の悩みに答える形で、以下の指示に厳密に従って、具体的で実践的なアドバイスを200文字程度で生成してください。\n\n"
        f"指示：{specific_instruction}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": final_prompt}],
            temperature=0.6, max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"アドバイスの生成中にエラーが発生しました: {e}"