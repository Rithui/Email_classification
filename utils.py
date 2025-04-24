import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import spacy
from transformers import BertTokenizer

# Load spaCy English model, download if not already available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Function to split email into subject and body
def extract_subject_and_body(email_text):
    parts = email_text.split('\n', 2)  # Split into at most 3 parts
    subject = ''
    body = email_text
    
    # Look for a line starting with 'Subject:' and extract the rest as body
    for part in parts:
        if part.lower().startswith('subject:'):
            subject = part.replace('subject:', '', 1).strip()
            body = '\n'.join(parts[parts.index(part) + 1:]).strip()
            break
    
    return subject, body

# Extract keywords from the subject to infer potential issue categories
def extract_technical_keywords(subject):
    technical_keywords = {
        "Password Recovery": [...],
        "Refund Request": [...],
        "Hardware Issue": [...],
        "Software Issue": [...],
        "Payment Issue": [...],
        "Account Issue": [...]
    }
    
    subject_lower = subject.lower()
    matched_categories = []
    
    # Match keywords from subject to predefined categories
    for category, keywords in technical_keywords.items():
        for keyword in keywords:
            if keyword in subject_lower:
                matched_categories.append((category, keyword))
                
    return matched_categories

# Load dataset and preprocess: extract features and encode labels
def load_and_preprocess_data(dataset_path, test_size=0.2, random_state=42):
    df = pd.read_csv(dataset_path)
    
    # Handle both 'email' and 'text' column names
    if 'email' in df.columns:
        df['subject'] = df['email'].apply(lambda x: extract_subject_and_body(x)[0])
        df['technical_keywords'] = df['subject'].apply(lambda x: [kw[1] for kw in extract_technical_keywords(x)])
        df['email_body'] = df['email'].apply(lambda x: extract_subject_and_body(x)[1])
        X = df['email'].values
    elif 'text' in df.columns:
        df['subject'] = df['text'].apply(lambda x: extract_subject_and_body(x)[0])
        df['technical_keywords'] = df['subject'].apply(lambda x: [kw[1] for kw in extract_technical_keywords(x)])
        df['email_body'] = df['text'].apply(lambda x: extract_subject_and_body(x)[1])
        X = df['text'].values
    else:
        raise ValueError("No email/text column found in dataset")
    
    # Handle both 'category' and 'type' as target columns
    if 'category' in df.columns:
        y = df['category'].values
    elif 'type' in df.columns:
        y = df['type'].values
    else:
        raise ValueError("No category/type column found in dataset")
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    reverse_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    return X_train, X_val, y_train, y_val, reverse_mapping, df

# Detect Personally Identifiable Information (PII) in text using regex and spaCy
def detect_pii(text):
    entities = []
    
    doc = nlp(text)
    
    # Regex-based product detection patterns
    product_patterns = [
        ...
    ]
    
    # Add matched products to entities list
    for pattern in product_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "position": [match.start(), match.end()],
                "classification": "PRODUCT",
                "entity": match.group()
            })
    
    # Add named entities from spaCy, avoiding overlaps
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "MONEY", "CARDINAL"]:
            overlaps = False
            for existing in entities:
                if (ent.start_char >= existing["position"][0] and 
                    ent.start_char < existing["position"][1]):
                    overlaps = True
                    break
            
            if not overlaps:
                entities.append({
                    "position": [ent.start_char, ent.end_char],
                    "classification": ent.label_,
                    "entity": ent.text
                })
    
    # Match email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        entities.append({
            "position": [match.start(), match.end()],
            "classification": "EMAIL",
            "entity": match.group()
        })
    
    # Match phone numbers
    phone_patterns = [
        ...
    ]
    
    for pattern in phone_patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "position": [match.start(), match.end()],
                "classification": "PHONE",
                "entity": match.group()
            })
    
    # Sort entities by position in text
    entities.sort(key=lambda x: x["position"][0])
    return entities

# Replace detected PII with corresponding entity tags
def mask_pii(text, entities):
    masked_text = text
    
    # Replace from end to avoid affecting positions of earlier matches
    for entity in sorted(entities, key=lambda x: x["position"][0], reverse=True):
        start, end = entity["position"]
        entity_type = entity["classification"]
        
        mask_mapping = {
            ...
        }
        
        mask = mask_mapping.get(entity_type, f"[{entity_type}]")
        masked_text = masked_text[:start] + mask + masked_text[end:]
    
    return masked_text

# Use keyword-based heuristics to classify the email topic
def classify_email_by_keywords(email_text):
    email_lower = email_text.lower()
    
    # Define patterns and scoring weights for different categories
    patterns = {
        ...
    }
    
    scores = {category: 0 for category in patterns.keys()}
    
    # Score each category based on matched patterns
    for category, pattern_list in patterns.items():
        for pattern, weight in pattern_list:
            if re.search(pattern, email_lower):
                scores[category] += weight
    
    # Add weights for specific terms to bias scoring
    if "refund" in email_lower:
        scores["Refund Request"] += 2
    if "broken" in email_lower or "damaged" in email_lower:
        scores["Hardware Issue"] += 2
    if "password" in email_lower:
        scores["Password Recovery"] += 2
    
    # Return the category with the highest score
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    return None

# Process email: extract PII, mask it, and classify email
def process_email_with_pii_handling(classifier, email_text):
    subject, body = extract_subject_and_body(email_text)
    technical_keywords = extract_technical_keywords(subject)
    
    # Detect and mask PII
    entities = detect_pii(email_text)
    masked_email = mask_pii(email_text, entities)
    
    # Classify using keywords first, fallback to classifier
    category = None
    if technical_keywords:
        category = technical_keywords[0][0]
    
    if category is None:
        combined_text = f"Subject: {subject}\n\n{body}"
        category = classify_email_by_keywords(combined_text)
        
        if category is None:
            category = classifier.predict(combined_text)
    
    return {
        "input_email_body": email_text,
        "subject": subject,
        "body": body,
        "technical_keywords": [kw[1] for kw in technical_keywords],
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
