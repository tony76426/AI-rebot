from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 設定 OpenAI API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")

# 載入資料庫
with open("vector_database.json", "r", encoding="utf-8") as f:
    database = json.load(f)

questions = [item["question"] for item in database]
answers = [item["answer"] for item in database]

# 初始化 TF-IDF
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# 初始化 Flask
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_file("airobt.html")

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    user_question = data.get("question", "")
    if not user_question:
        return jsonify({"question": "", "answer": "⚠️ 請輸入問題。", "score": 0.0})

    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    top_index = int(similarities.argmax())
    top_score = float(similarities[top_index])

    if top_score >= 0.25:
        return jsonify({
            "question": questions[top_index],
            "answer": answers[top_index],
            "score": top_score
        })
    else:
        # 查無相近問題時呼叫 OpenAI GPT 回答
        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"請以簡明法律說明回答以下問題：{user_question}"
                }]
            )
            gpt_answer = gpt_response["choices"][0]["message"]["content"].strip()
            return jsonify({
                "question": "由 GPT 模型生成",
                "answer": gpt_answer,
                "score": 0.0
            })
        except Exception as e:
            return jsonify({
                "question": "GPT 回答失敗",
                "answer": "⚠️ 無法取得 GPT 回應，請稍後再試。",
                "score": 0.0
            })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)