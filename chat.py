from flask import Flask, request, jsonify
import torch
import random
import json
from nltk_util import bag_of_words, tokenize
from model import NeuralNet

app = Flask(__name__)

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

user_context = {}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id')
    sentence = data['message']

    if user_id not in user_context:
        user_context[user_id] = {}

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return jsonify({"response": random.choice(intent['responses']), "tag": tag})
    else:
        return jsonify({"response": "I do not understand..."})

def forward_to_laravel(data, endpoint):
    import requests
    laravel_url = 'http://127.0.0.1:8000/api/' + endpoint
    response = requests.post(laravel_url, json=data)
    return response.json()

if __name__ == '__main__':
    app.run(debug=True)
