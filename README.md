# Email Classification and PII Handling System

This repository showcases an advanced email classification system built with **BERT**, capable of categorizing emails into specific categories (such as Refund Request, Password Recovery, Hardware Issues, etc.). Additionally, the system incorporates **PII (Personally Identifiable Information)** detection and masking to ensure data privacy and compliance with regulations.

## Project Overview

This project demonstrates the integration of **natural language processing (NLP)** and **machine learning** techniques to:
- **Classify emails** based on predefined categories.
- **Extract and mask PII** entities such as names, phone numbers, and email addresses from the email content.
- **Utilize BERT** for email classification with the added capability of keyword-based classification and PII masking.
  
The system is intended to help businesses automate email handling, categorize support tickets efficiently, and protect sensitive user data from exposure.

## Key Features

- **BERT-based Email Classification**: Categorizes emails into predefined classes (e.g., Refund Request, Hardware Issue, etc.).
- **PII Detection & Masking**: Identifies and replaces sensitive personal information in emails with placeholder tokens (e.g., `[PERSON]`, `[EMAIL]`).
- **Technical Keyword Identification**: Extracts technical keywords from the email subject to aid in further classification and decision-making.
- **Data Privacy Compliance**: Masks PII information to ensure compliance with data privacy regulations like GDPR and CCPA.

## Technologies Used

- **BERT**: For text classification and understanding.
- **Python**: Primary programming language for model training and evaluation.
- **SpaCy**: For Named Entity Recognition (NER) to detect and mask PII.
- **Scikit-learn**: For model evaluation and utility functions.
- **Regular Expressions**: For detecting technical keywords and PII patterns.
- **Pandas**: For data manipulation and handling the dataset.
- **TensorFlow/PyTorch**: For model training and deployment.

## Installation

### Prerequisites
- Python 3.7+
- Pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/email-classification-pii-handling.git
cd email-classification-pii-handling


###Step 2: Install Dependencies
