from fastapi import FastAPI, HTTPException
import requests
import json
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class EmailRequest(BaseModel):
    email: str

class EmailResponse(BaseModel):
    category: str
    confidence: float
    processed_text: str

@app.post("/predict", response_model=EmailResponse)
async def predict_email(request: EmailRequest):
    try:
        # Call your deployed HuggingFace space
        response = requests.post(
            "https://your-huggingface-space-url/predict",
            json={"text": request.email},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Model prediction failed")
        
        result = response.json()
        
        return EmailResponse(
            category=result["category"],
            confidence=result["confidence"],
            processed_text=result["processed_text"]
        )
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling HuggingFace API: {str(e)}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
