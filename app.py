# Importing required libraries
import gradio as gr
import json
import zipfile
import os
import sys
import shutil
import pandas as pd
from models import load_classifier, BERTEmailClassifier
from utils import process_email_with_pii_handling, load_and_preprocess_data
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Adding the current directory to the system path if not already present
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Email Classifier API",
    description="API for classifying emails and detecting PII",
    version="1.0.0"
)

# Enabling CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define dataset and extraction paths
zip_path = "data.zip"
extract_path = "."
dataset_path = os.path.join('./data', 'customer_support.csv')
processed_dataset_path = os.path.join('./data', 'processed_customer_support.csv')

# Handle fallback if data is in project root
customer_support_path = os.path.join(current_dir, 'customer_support.csv')
if os.path.exists(customer_support_path):
    dataset_path = customer_support_path
    processed_dataset_path = os.path.join(current_dir, 'processed_customer_support.csv')
    print(f"Using customer_support.csv dataset at {dataset_path}")
else:
    # Extract ZIP if no CSV found directly
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Extracted data from {zip_path}")
            
            # Try various paths to locate the extracted dataset
            possible_paths = [
                './data/customer_support.csv',
                './customer_support.csv',
                './data/data/customer_support.csv'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    if path != dataset_path:
                        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                        shutil.copy(path, dataset_path)
                        print(f"Copied dataset from {path} to {dataset_path}")
                    break
            
        except Exception as e:
            print(f"Error extracting data: {e}")
    else:
        print(f"Warning: Neither customer_support.csv nor data.zip found. Will use demo model.")

# Define model and dataset variables
model_path = './ticket_classifier_bert.pt'
classifier = None
processed_df = None

# Try loading a preprocessed dataset if available
if os.path.exists(processed_dataset_path):
    try:
        processed_df = pd.read_csv(processed_dataset_path)
        print(f"Loaded processed dataset from {processed_dataset_path}")
    except Exception as e:
        print(f"Error loading processed dataset: {e}")

# Create a demo model if no model file exists
if not os.path.exists(model_path):
    try:
        print("Creating a simple model for production use...")
        import torch
        
        if os.path.exists(dataset_path):
            # Load and preprocess the dataset
            X_train, X_val, y_train, y_val, reverse_mapping, df = load_and_preprocess_data(dataset_path)
            df.to_csv(processed_dataset_path, index=False)
            processed_df = df
            print(f"Created and saved processed dataset to {processed_dataset_path}")
            
            # Prepare label mapping for model
            num_categories = len(reverse_mapping)
            label_map = reverse_mapping
        else:
            # Default category map if dataset not found
            num_categories = 5
            label_map = {
                0: "Incident", 
                1: "Problem", 
                2: "Request", 
                3: "Change",
                4: "Other"
            }
        
        # Initialize and save a dummy BERT model
        demo_classifier = BERTEmailClassifier(num_categories)
        demo_classifier.label_map = label_map
        
        torch.save({
            'model_state_dict': demo_classifier.model.state_dict(),
            'label_map': demo_classifier.label_map
        }, model_path)
        print(f"Model created and saved to {model_path}")
        
        classifier = demo_classifier
    except Exception as e:
        print(f"Error creating model: {e}")

# Attempt to load model if it exists
if classifier is None:
    try:
        if os.path.exists(model_path):
            print(f"Found model at {model_path}, attempting to load...")
            classifier = load_classifier(model_path)
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Model file not found at {model_path}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("UI will start but classification will not work until a model is available")

# Train model triggered from UI
def train_model_from_ui():
    try:
        if os.path.exists(customer_support_path):
            dataset_to_use = customer_support_path
        elif os.path.exists(dataset_path):
            dataset_to_use = dataset_path
        else:
            return f"Error: Dataset not found. Please ensure customer_support.csv exists."
        
        print(f"Training model using dataset: {dataset_to_use}")
        
        # Preprocess data
        X_train, X_val, y_train, y_val, reverse_mapping, df = load_and_preprocess_data(dataset_to_use)
        df.to_csv(processed_dataset_path, index=False)
        global processed_df
        processed_df = df
        print(f"Saved processed dataset to: {processed_dataset_path}")
        
        # Reduce training size for quick training
        max_samples = min(300, len(X_train))
        X_train_small = X_train[:max_samples]
        y_train_small = y_train[:max_samples]
        
        # Initialize new classifier
        num_labels = len(reverse_mapping)
        new_classifier = BERTEmailClassifier(num_labels)
        new_classifier.label_map = reverse_mapping
        
        # Train the model
        print("Starting model training with enhanced features...")
        history = new_classifier.train(
            X_train_small, 
            y_train_small, 
            X_val[:50],
            y_val[:50],
            epochs=1,
            batch_size=4
        )
        
        new_classifier.save_model(model_path)
        global classifier
        classifier = new_classifier
        
        return f"Model trained and saved. Enhanced dataset saved to {processed_dataset_path}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during training: {error_details}")
        return f"Error training model: {str(e)}. Consider using a pre-trained model instead."

# Classify email and return results
def classify_email(email_body):
    if not email_body.strip():
        return "Please enter an email to classify.", "", "", "", "", "", "", ""
    
    if classifier is None:
        return (
            email_body,
            "",
            "",
            "",
            "Error: Model not loaded properly. Please check the logs.",
            "Error - Model not available",
            [["Error", "Model not loaded", ""]],
            json.dumps({"error": "Model not loaded"}, indent=2)
        )
    
    result = process_email_with_pii_handling(classifier, email_body)
    
    # Format extracted entities
    entities_table = []
    for entity in result["list_of_masked_entities"]:
        entities_table.append([
            entity["classification"],
            entity["entity"],
            f"{entity['position'][0]}-{entity['position'][1]}"
        ])
    
    if not entities_table:
        entities_table = [["No entities found", "", ""]]
    
    formatted_json = json.dumps(result, indent=2)
    
    return (
        result["input_email_body"],
        result["subject"],
        result["body"],
        ", ".join(result["technical_keywords"]),
        result["masked_email"],
        result["category_of_the_email"],
        entities_table,
        formatted_json
    )

# FastAPI endpoint to classify an email via JSON POST
@app.post("/api/classify")
async def api_classify_email(request: Request):
    try:
        try:
            data = await request.json()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid JSON in request: {str(e)}"}
            )
        
        if 'email' not in data:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'email' field in request"}
            )
        
        email_text = data['email']
        
        if classifier is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Model not loaded"}
            )
        
        try:
            result = process_email_with_pii_handling(classifier, email_text)
            
            # Reformat result entities as JSON dictionary
            formatted_entities = {}
            for i, entity in enumerate(result["list_of_masked_entities"]):
                formatted_entities[str(i)] = {
                    "position": entity["position"],
                    "classification": entity["classification"],
                    "entity": entity["entity"]
                }
            
            formatted_result = {
                "input_email_body": result["input_email_body"],
                "subject": result["subject"],
                "body": result["body"],
                "technical_keywords": result["technical_keywords"],
                "masked_email": result["masked_email"],
                "category_of_the_email": result["category_of_the_email"],
                "list_of_masked_entities": formatted_entities
            }
            
            return JSONResponse(content=formatted_result)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing email: {error_details}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing email: {str(e)}"}
            )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Unexpected error: {error_details}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        )

# Endpoint for specific client integration
@app.post("rithuikprakash/Intern ")
async def classify_email_endpoint(request: Request):
    """Endpoint to handle requests from email_client.py"""
    return await api_classify_email(request)

# Build Gradio UI
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("# 📧 Smart Email Classifier & PII Detector")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("Manual Email Processing"):
                # Email input and submit button
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## ✍️ Input Email")
                        email_input = gr.Textbox(
                            label="",
                            placeholder="Type or paste your support email here...",
                            lines=10
                        )
                        submit_btn = gr.Button("Analyze Email", variant="primary", size="lg")
                    
                    # Output columns
                    with gr.Column(scale=3):
                        gr.Markdown("## 🔍 Analysis Results")
                        
                        with gr.Row():
                            with gr.Column():
                                category_label = gr.Textbox(label="📋 Predicted Ticket Type")
                        
                        with gr.Tabs():
                            with gr.TabItem("Subject"):
                                subject_output = gr.Textbox(label="Extracted Subject", lines=2)
                            
                            with gr.TabItem("Technical Keywords"):
                                keywords_output = gr.Textbox(label="Detected Keywords", lines=2)
                            
                            with gr.TabItem("Body"):
                                body_output = gr.Textbox(label="Email Body", lines=8)
                            
                            with gr.TabItem("Masked Email (PII Removed)"):
                                masked_email = gr.Textbox(label="", lines=8)
                            
                            with gr.TabItem("Original Email"):
                                original_email = gr.Textbox(label="", lines=8)
                        
                        gr.Markdown("### 🔒 Detected PII Entities")
                        entities_table = gr.Dataframe(
                            headers=["Entity Type", "Original Value", "Position"],
                            datatype=["str", "str", "str"],
                            label="",
                            row_count=(1, "dynamic")
                        )
                        
                        with gr.Accordion("API Response (JSON)", open=False):
                            json_output = gr.JSON()
            
            with gr.TabItem("Train Model"):
                # Training interface
                gr.Markdown("## 🧠 Train the Model")
                train_btn = gr.Button("Train Model with customer_support.csv", variant="primary")
                train_output = gr.Textbox(label="Training Output", lines=5)
                
                gr.Markdown("""
                ### Training Information
                This will train the model using the customer_support.csv dataset. The training process:
                1. Creates enhanced dataset with subject and keywords extraction
                2. Uses a subset of the data for faster training
                3. Runs for just one epoch to quickly provide a working model
                4. Saves both the model and processed dataset
                For production use, you would want to train with more data and for more epochs.
                """)
            
            with gr.TabItem("API Documentation"):
                # API documentation UI tab
                gr.Markdown("""
                ## 🔧 API Documentation
                
                ### Endpoint
                POST /api/classify
                
                ### Request Format
                ```json
                {
                    "email": "Your email content here"
                }
                ```
                
                ### Response Format
                ```json
                {
                    "input_email_body": "Original email",
                    "subject": "Extracted subject",
                    "body": "Email body",
                    "technical_keywords": ["keyword1", "keyword2"],
                    "masked_email": "Email with PII masked",
                    "category_of_the_email": "Predicted category",
                    "list_of_masked_entities": {
                        "0": {
                            "position": [start, end],
                            "classification": "entity_type",
                            "entity": "original_value"
                        }
                    }
                }
                ```
                """)
        
        # Link button events with functions
        submit_btn.click(
            fn=classify_email,
            inputs=[email_input],
            outputs=[
                original_email,
                subject_output,
                body_output,
                keywords_output,
                masked_email,
                category_label,
                entities_table,
                json_output
            ]
        )
        
        train_btn.click(
            fn=train_model_from_ui,
            inputs=[],
            outputs=[train_output]
        )
    
    return demo

# Mount Gradio UI to FastAPI
demo = create_interface()
app = gr.mount_gradio_app(app, demo, path="/")

# Launch the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
