# Change this line:
self.api_endpoint = "https://huggingface.co/spaces/rithuikprakash/Intern/api/classify"

# To this (using environment variables for flexibility):
self.api_endpoint = os.getenv("API_ENDPOINT", "http://localhost:7860/api/classify")

# Change the request format:
response = requests.post(
    self.api_endpoint,
    json={"email": email_data["body"]},  # Current format
    headers={"Content-Type": "application/json"},
    timeout=30
)

# To match the expected format in api.py:
response = requests.post(
    self.api_endpoint,
    json={"email_body": email_data["body"]},  # Updated format
    headers={"Content-Type": "application/json"},
    timeout=30
)


#server.py
# Change this line:
response = requests.post(
    "https://your-huggingface-space-url/predict",
    json={"text": request.email},
    headers={"Content-Type": "application/json"}
)

# To this (using environment variables):
model_url = os.getenv("MODEL_URL", "http://localhost:7860/api/classify")
response = requests.post(
    model_url,
    json={"email_body": request.email},
    headers={"Content-Type": "application/json"}
)
# Update the response mapping to match actual API response:
result = response.json()
        
return EmailResponse(
    category=result.get("category_of_the_email", "Unknown"),
    confidence=float(result.get("confidence", 0.0)),  # May need to handle missing field
    processed_text=result.get("masked_email", "")
)
