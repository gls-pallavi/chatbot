import json
import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize the Flask app
app = Flask(__name__)

# Load intents.json data
with open('intents.json') as file:
    intents_data = json.load(file)

# Extract tags and responses from intents.json
tags = [intent['tag'] for intent in intents_data['intents']]

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./trained_model')
model = BertForSequenceClassification.from_pretrained('./trained_model')

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Root route for testing the server
@app.route("/", methods=["GET"])
def home():
    return "Flask backend is running! Use the /chatbot endpoint to interact with the chatbot.", 200

# Route to handle chatbot responses
@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        user_input = request.json.get('message', '').strip()
        if not user_input:
            return jsonify({"response": "Please enter a valid message."})

        print(f"User Input: {user_input}")  # Debug log
        
        # Tokenize user input
        encoded_input = tokenizer(user_input, truncation=True, padding='max_length', max_length=128, return_tensors="pt").to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**encoded_input)
            logits = outputs.logits

        # Identify predicted tag
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_tag = tags[predicted_class]
        print(f"Predicted Tag: {predicted_tag}")  # Debug log

        # Find response for the predicted tag
        response = next(intent['responses'] for intent in intents_data['intents'] if intent['tag'] == predicted_tag)
        print(f"Response: {response[0]}")  # Debug log

        return jsonify({"response": response[0]})
    except Exception as e:
        print(f"Error occurred: {e}")  # Debug log
        return jsonify({"response": "Sorry, I encountered an error!"}), 500

# Custom error handler for 404
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "The requested URL was not found on the server."}), 404

# Main driver function
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
