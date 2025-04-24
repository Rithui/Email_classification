import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import os

class BERTEmailClassifier:
    def __init__(self, num_labels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = EmailClassifierModel(num_labels).to(self.device)
        self.label_map = None

    def preprocess_text(self, text):
        from utils import extract_subject_and_body, extract_technical_keywords
        
        subject, body = extract_subject_and_body(text)
        technical_keywords = extract_technical_keywords(subject)
        
        combined_text = f"Subject: {subject}\n"
        if technical_keywords:
            combined_text += f"Keywords: {', '.join(kw[1] for kw in technical_keywords)}\n"
        combined_text += f"\n{body}"
        
        encoded = self.tokenizer(
            combined_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return encoded

    def train(self, X_train, y_train, X_val, y_val, epochs=3, batch_size=8):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch + 1}/{epochs}')
            
            for i in progress_bar:
                batch_texts = X_train[i:i + batch_size]
                batch_labels = y_train[i:i + batch_size]
                
                batch_encodings = [self.preprocess_text(text) for text in batch_texts]
                
                input_ids = torch.cat([enc['input_ids'] for enc in batch_encodings], dim=0).to(self.device)
                attention_mask = torch.cat([enc['attention_mask'] for enc in batch_encodings], dim=0).to(self.device)
                labels = torch.tensor(batch_labels).to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (i + batch_size)})
            
            avg_train_loss = total_loss / len(X_train)
            train_losses.append(avg_train_loss)
            
            val_loss = self.evaluate(X_val, y_val, criterion, batch_size)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')
        
        return {'train_loss': train_losses, 'val_loss': val_losses}

    def evaluate(self, X_val, y_val, criterion, batch_size=8):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_texts = X_val[i:i + batch_size]
                batch_labels = y_val[i:i + batch_size]
                
                batch_encodings = [self.preprocess_text(text) for text in batch_texts]
                
                input_ids = torch.cat([enc['input_ids'] for enc in batch_encodings], dim=0).to(self.device)
                attention_mask = torch.cat([enc['attention_mask'] for enc in batch_encodings], dim=0).to(self.device)
                labels = torch.tensor(batch_labels).to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()
        
        return total_loss / len(X_val)

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            encoded = self.preprocess_text(text)
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(predictions, dim=1).item()
            
            if self.label_map is not None:
                return self.label_map[predicted_label]
            return predicted_label

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_map': self.label_map
        }, path)

class EmailClassifierModel(nn.Module):
    def __init__(self, num_labels):
        super(EmailClassifierModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return type('BertOutput', (), {'logits': logits})()

def load_classifier(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    if 'label_map' not in checkpoint:
        raise ValueError("Checkpoint does not contain label_map")
    
    num_labels = len(checkpoint['label_map'])
    classifier = BERTEmailClassifier(num_labels)
    classifier.model.load_state_dict(checkpoint['model_state_dict'])
    classifier.label_map = checkpoint['label_map']
    
    return classifier
