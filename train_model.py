import os
from pathlib import Path
from models import BERTEmailClassifier
from utils import load_and_preprocess_data

def train_model():
    """Train the email classification model"""
    
    # Create a 'data' directory if it doesn't already exist
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    # Define the path to the dataset CSV file
    dataset_path = data_dir / 'support_emails.csv'
    
    # Check if the dataset file exists; raise an error if not
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please place your CSV file in the data directory.")
    
    # Load and preprocess the dataset using utility function
    # This should return training and validation splits, along with the label reverse mapping
    X_train, X_val, y_train, y_val, reverse_mapping = load_and_preprocess_data(str(dataset_path))
    
    # Get the number of unique labels (classes) for classification
    num_labels = len(reverse_mapping)
    # Initialize the BERT email classifier with the number of labels
    classifier = BERTEmailClassifier(num_labels)
    # Assign label-to-index mapping to the classifier for interpretation during prediction
    classifier.label_map = reverse_mapping
    
    # Start training the classifier model
    print("Starting model training...")
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=3, batch_size=8)
    
    # Save the trained model to a file
    model_path = './ticket_classifier_bert.pt'
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Return the trained model and its training history
    return classifier, history

# Run the training function if this script is executed as the main program
if __name__ == "__main__":
    train_model()
