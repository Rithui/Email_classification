import os
from pathlib import Path
from models import BERTEmailClassifier
from utils import load_and_preprocess_data

def train_model():
    """Train the email classification model"""
    # Create data directory if it doesn't exist
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    # Path to your CSV dataset
    dataset_path = data_dir / 'support_emails.csv'
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please place your CSV file in the data directory.")
    
    # Load and preprocess data
    X_train, X_val, y_train, y_val, reverse_mapping = load_and_preprocess_data(str(dataset_path))
    
    # Initialize the classifier
    num_labels = len(reverse_mapping)
    classifier = BERTEmailClassifier(num_labels)
    classifier.label_map = reverse_mapping
    
    # Train the model
    print("Starting model training...")
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=3, batch_size=8)
    
    # Save the model
    model_path = './ticket_classifier_bert.pt'
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return classifier, history
if __name__ == "__main__":
    train_model()
