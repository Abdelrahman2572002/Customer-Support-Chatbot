from flask import Flask, request, jsonify, render_template
from customer_support_chatbot import chain
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
app = Flask(__name__)
CORS(app)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "").strip()
 
    if not message:
        return jsonify({"error": "Empty message"}), 400
 
    try:
        
        answer = chain.invoke(message)
        return jsonify({"reply": answer})
 
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
 
 

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
