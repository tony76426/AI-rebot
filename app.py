
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

# 設定 OpenAI API 金鑰（使用環境變數）
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# 載入向量資料庫
with open("vector_database.json", "r", encoding="utf-8") as f:
    database = json.load(f)

# 提取問題與向量
questions = [item["question"] for item in database]
vectors = np.array([item["vector"] for item in database])
answers = [item["answer"] for item in database]

# 初始化 Flask
app = Flask(__name__)
CORS(app)

# 處理 API 請求
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    user_question = data.get("question", "")

    # 將使用者問題轉換成向量（使用 OpenAI embedding 模型）
    embedding_response = openai.Embedding.create(
        input=user_question,
        model="text-embedding-ada-002"
    )
    user_vector = np.array(embedding_response["data"][0]["embedding"]).reshape(1, -1)

    # 計算與資料庫的相似度
    similarities = cosine_similarity(user_vector, vectors)[0]
    top_index = int(np.argmax(similarities))

    matched_answer = answers[top_index]
    matched_question = questions[top_index]

    # 回傳匹配結果
    return jsonify({
        "question": matched_question,
        "answer": matched_answer,
        "score": float(similarities[top_index])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
