from flask import Flask, request, jsonify
from models import load_classifier, BERTEmailClassifier
from utils import process_email_with_pii_handling
import os

# Initialize a Flask web application
app = Flask(__name__)

# Define the path to the saved BERT email classifier model
model_path = './ticket_classifier_bert.pt'

try:
    # Check if the model file exists at the given path
    if os.path.exists(model_path):
        # Load the classifier using the custom loader function
        classifier = load_classifier(model_path)
        print(f"Model loaded successfully from {model_path}")
    else:
        # If model file doesn't exist, set classifier to None and run in demo mode
        print(f"Model file not found at {model_path}. API will run in demo mode.")
        classifier = None
except Exception as e:
    # Catch and print any error while loading the model
    print(f"Error loading model: {e}")
    print("API will start but classification will not work until a model is available")
    classifier = None

@app.route('/classify_email', methods=['POST'])
def classify_email():
    """API endpoint to classify an email"""
    data = request.json  # Get JSON data from POST request

    # Check if 'email_body' is present in the request
    if not data or 'email_body' not in data:
        return jsonify({"error": "No email body provided"}), 400

    email_body = data['email_body']  # Extract email body from the request

    # If classifier is not loaded, return a demo response
    if classifier is None:
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

    # Use the utility function to process the email and handle PII with the classifier
    result = process_email_with_pii_handling(classifier, email_body)

    # Return the result as a JSON response
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Return the health status of the API and model loading status
    return jsonify({
        "status": "healthy", 
        "model_loaded": classifier is not None
    })

# Run the Flask app on host 0.0.0.0 and port 5000 with debug mode enabled
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
