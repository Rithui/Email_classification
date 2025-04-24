from fastapi import FastAPI, HTTPException
import requests
import json
from pydantic import BaseModel
import logging
import os
from gradio_client import Client  # Add this import

# Set up proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='proxy_server.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

app = FastAPI()

class EmailRequest(BaseModel):
    email: str

class EmailResponse(BaseModel):
    category: str
    confidence: float
    processed_text: str = ""

# Create a Gradio client for the HuggingFace space
gradio_client = Client("rithuikprakash/Intern")


@app.post("/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    try:
        logger.info(f"Sending request to HuggingFace Gradio API")
        
        # Use the Gradio client to make the prediction
        result = gradio_client.predict(
            email_body=request.email,
            api_name="rithuikprakash/Intern"
        )
        
        logger.info(f"Successfully received response from HuggingFace API: {result}")
        
        # Parse the result based on the format returned by your Gradio interface
        # You may need to adjust this based on what your Gradio interface returns
        if isinstance(result, list):
            # If result is a list, extract relevant information
            category = result[5] if len(result) > 5 else "unknown"
            confidence = 0.9  # Default confidence since Gradio might not return this
            processed_text = result[0] if len(result) > 0 else request.email[:100] + "..."
        elif isinstance(result, dict):
            # If result is a dictionary
            category = result.get("category_of_the_email", "unknown")
            confidence = result.get("confidence", 0.9)
            processed_text = result.get("input_email_body", request.email[:100] + "...")
        else:
            # Fallback
            category = "unknown"
            confidence = 0.0
            processed_text = request.email[:100] + "..."
        
        return EmailResponse(
            category=category,
            confidence=confidence,
            processed_text=processed_text
        )
    
    except Exception as e:
        logger.error(f"Error calling HuggingFace Gradio API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "email-classification-proxy"}

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting proxy server on port {port}")
    print(f"Starting proxy server on port {port}. See logs for details.")
    
    uvicorn.run(app, host="0.0.0.0", port=port)