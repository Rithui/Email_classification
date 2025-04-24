from flask import Flask, request, jsonify
from models import load_classifier, BERTEmailClassifier
from utils import process_email_with_pii_handling
import os

app = Flask(__name__)

# Load the classifier
model_path = './ticket_classifier_bert.pt'
try:
    if os.path.exists(model_path):
        classifier = load_classifier(model_path)
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Model file not found at {model_path}. API will run in demo mode.")
        classifier = None
except Exception as e:
    print(f"Error loading model: {e}")
    print("API will start but classification will not work until a model is available")
    classifier = None

@app.route('/classify_email', methods=['POST'])
def classify_email():
    """API endpoint to classify an email"""
    data = request.json
    
    if not data or 'email_body' not in data:
        return jsonify({"error": "No email body provided"}), 400
    
    email_body = data['email_body']
    
    if classifier is None:
        # Return a demo response when no model is available
        demo_response = {
            "input_email_body": email_body,
            "masked_email": "This is a demo response. No model is loaded.",
            "category_of_the_email": "DEMO",
            "list_of_masked_entities": [
                {
                    "position": [0, 10],
                    "classification": "demo",
                    "entity": "demo entity"
                }
            ]
        }
        return jsonify(demo_response)
    
    # Process the email with PII handling
    result = process_email_with_pii_handling(classifier, email_body)
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": classifier is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
