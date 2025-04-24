import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import spacy
from transformers import BertTokenizer

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_subject_and_body(email_text):
    parts = email_text.split('\n', 2)
    subject = ''
    body = email_text
    
    for part in parts:
        if part.lower().startswith('subject:'):
            subject = part.replace('subject:', '', 1).strip()
            body = '\n'.join(parts[parts.index(part) + 1:]).strip()
            break
    
    return subject, body

def extract_technical_keywords(subject):
    technical_keywords = {
        "Password Recovery": [
            "forgot password", "reset password", "password recovery",
            "cannot login", "login issues", "account access"
        ],
        "Refund Request": [
            "refund", "money back", "cancel order", "return", "reimbursement",
            "refund not received", "refund status"
        ],
        "Hardware Issue": [
            "broken", "damaged", "not working", "hardware", "device issue",
            "screen", "battery", "keyboard", "headphone", "speaker"
        ],
        "Software Issue": [
            "error", "bug", "crash", "software", "application", "not loading",
            "update failed", "installation failed"
        ],
        "Payment Issue": [
            "payment failed", "transaction", "billing", "charge", "payment error"
        ],
        "Account Issue": [
            "account locked", "account blocked", "verification", "authenticate"
        ]
    }
    
    subject_lower = subject.lower()
    matched_categories = []
    
    for category, keywords in technical_keywords.items():
        for keyword in keywords:
            if keyword in subject_lower:
                matched_categories.append((category, keyword))
                
    return matched_categories

def load_and_preprocess_data(dataset_path, test_size=0.2, random_state=42):
    df = pd.read_csv(dataset_path)
    
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
    
    if 'category' in df.columns:
        y = df['category'].values
    elif 'type' in df.columns:
        y = df['type'].values
    else:
        raise ValueError("No category/type column found in dataset")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    reverse_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    return X_train, X_val, y_train, y_val, reverse_mapping, df

def detect_pii(text):
    entities = []
    
    doc = nlp(text)
    
    product_patterns = [
        r'\b(?:Bose|Sony|Samsung|Apple|HP|Dell|Lenovo)\s+(?:[A-Za-z0-9]+([-\s][A-Za-z0-9]+)*\s*)(?:Series|Pro|Max|Ultra|Plus|\d+)?\b',
        r'\b(?:iPhone|iPad|MacBook|Surface|ThinkPad|XPS|Spectre)\s*(?:[A-Za-z0-9]+([-\s][A-Za-z0-9]+)*)\b',
        r'\b(?:Bose\s+(?:QuietComfort|SoundLink|Noise\s+Cancelling\s+Headphones)\s*[A-Za-z0-9]*)\b',
        r'\b(?:Headphones|Laptop|Phone|Tablet|Watch)\s+(?:Pro|Air|Max|Ultra|Plus|\d+)?\b',
        r'\b(?:Bose Noise Cancelling Headphones \d+)\b'
    ]
    
    for pattern in product_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "position": [match.start(), match.end()],
                "classification": "PRODUCT",
                "entity": match.group()
            })
    
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
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        entities.append({
            "position": [match.start(), match.end()],
            "classification": "EMAIL",
            "entity": match.group()
        })
    
    phone_patterns = [
        r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
    ]
    
    for pattern in phone_patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "position": [match.start(), match.end()],
                "classification": "PHONE",
                "entity": match.group()
            })
    
    entities.sort(key=lambda x: x["position"][0])
    return entities

def mask_pii(text, entities):
    masked_text = text
    
    for entity in sorted(entities, key=lambda x: x["position"][0], reverse=True):
        start, end = entity["position"]
        entity_type = entity["classification"]
        
        mask_mapping = {
            "PERSON": "[PERSON]",
            "EMAIL": "[EMAIL]",
            "PHONE": "[PHONE]",
            "ORG": "[ORGANIZATION]",
            "ORGANIZATION": "[ORGANIZATION]",
            "GPE": "[LOCATION]",
            "LOC": "[LOCATION]",
            "LOCATION": "[LOCATION]",
            "PRODUCT": "[PRODUCT]",
            "MONEY": "[MONEY]",
            "CARDINAL": "[NUMBER]"
        }
        
        mask = mask_mapping.get(entity_type, f"[{entity_type}]")
        masked_text = masked_text[:start] + mask + masked_text[end:]
    
    return masked_text

def classify_email_by_keywords(email_text):
    email_lower = email_text.lower()
    
    patterns = {
        "Refund Request": [
            (r'(refund|money back|reimbursement) (not received|missing|pending|status|process)', 3),
            (r'(refund|money back) (request|status|process|inquiry)', 3),
            (r'(cancel|cancellation|cancelled).*(refund|money back|return)', 3),
            (r'(haven\'t received|waiting for|where is) (refund|money back)', 3),
            (r'(status|update).*(refund|money back)', 2),
            (r'type:.*refund', 4),
            (r'(return|exchange).*(money|refund)', 2),
            (r'(payment|charge).*(refund|return)', 2)
        ],
        "Hardware Issue": [
            (r'(screen|display|monitor) (flicker|flickering|issue|problem|not working|broken)', 2),
            (r'(keyboard|keys|mouse|touchpad|hardware) (not working|issue|problem|broken|damaged)', 2),
            (r'(battery|charging|power) (issue|problem|not working|draining|dead)', 2),
            (r'(wifi|bluetooth|connection|internet) (issue|problem|not working|slow|dropping)', 2),
            (r'(speaker|audio|sound|headphone) (issue|problem|not working|quality)', 2),
            (r'(device|laptop|computer|phone) (broken|damaged|not working|faulty)', 2)
        ],
        "Software Issue": [
            (r'(software|program|application) (issue|problem|not working|crash)', 2),
            (r'(windows|macos|linux|os) (issue|problem|not working|crash)', 2),
            (r'(update|upgrade) (issue|problem|not working|failed)', 2),
            (r'(install|installation) (issue|problem|not working|failed)', 2),
            (r'(error|bug) (message|code|screen)', 2)
        ],
        "Password Recovery": [
            (r'(password) (reset|recovery|forgot|change)', 3),
            (r'(account) (access|recovery|locked)', 2),
            (r'(login|sign in) (issue|problem|not working)', 2),
            (r'(forgot|reset|change) (password|credentials)', 3)
        ],
        "Request": [
            (r'(request|asking for) (information|help|assistance)', 1),
            (r'(how|what|when|where) (to|do|can|should)', 1),
            (r'(need|want) (help|assistance|information|guidance)', 1),
            (r'(please|kindly) (help|assist|guide|provide)', 1)
        ],
        "Incident": [
            (r'(urgent|emergency|critical|immediate) (issue|problem|attention)', 2),
            (r'(service|system|application) (down|outage|unavailable)', 2),
            (r'(cannot|unable to) (access|use|connect)', 1),
            (r'(error|failure|crash) (occurred|happened)', 2)
        ]
    }
    
    scores = {category: 0 for category in patterns.keys()}
    
    for category, pattern_list in patterns.items():
        for pattern, weight in pattern_list:
            if re.search(pattern, email_lower):
                scores[category] += weight
    
    if "refund" in email_lower:
        scores["Refund Request"] += 2
    if "broken" in email_lower or "damaged" in email_lower:
        scores["Hardware Issue"] += 2
    if "password" in email_lower:
        scores["Password Recovery"] += 2
    
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    return None

def process_email_with_pii_handling(classifier, email_text):
    subject, body = extract_subject_and_body(email_text)
    technical_keywords = extract_technical_keywords(subject)
    
    entities = detect_pii(email_text)
    masked_email = mask_pii(email_text, entities)
    
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
