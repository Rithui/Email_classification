import requests
import json
import argparse
import os
import imaplib
import email
from email.header import decode_header
import time
import configparser

def clean_text(text):
    """Clean and decode email text"""
    if isinstance(text, bytes):
        text = text.decode()
    return text.strip()

def decode_email_header(header):
    """Decode email header"""
    decoded_header = decode_header(header)
    header_text = ""
    for part, encoding in decoded_header:
        if isinstance(part, bytes):
            if encoding:
                header_text += part.decode(encoding)
            else:
                header_text += part.decode()
        else:
            header_text += part
    return header_text.strip()

def get_email_body(msg):
    """Extract the email body text"""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
            
            # Get the body text
            if content_type == "text/plain":
                return clean_text(part.get_payload(decode=True))
            elif content_type == "text/html":
                # If only HTML is available, use it
                return clean_text(part.get_payload(decode=True))
    else:
        # Not multipart - get the payload directly
        return clean_text(msg.get_payload(decode=True))
    
    return ""

def classify_single_email(api_url, email_text, subject=""):
    """Send a single email to the API for classification"""
    try:
        response = requests.post(
            api_url,
            json={"email": email_text, "subject": subject},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Classification: {result.get('category_of_the_email', 'Unknown')}")
            print(f"Masked Email: {result.get('masked_email', '')[:100]}...")
            print(f"Detected {len(result.get('list_of_masked_entities', []))} PII entities")
            return result
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_inbox(api_url, config):
    """Process emails from inbox and send to API"""
    try:
        # Connect to the IMAP server
        mail = imaplib.IMAP4_SSL(config['EMAIL']['imap_server'])
        mail.login(config['EMAIL']['email'], config['EMAIL']['password'])
        mail.select("INBOX")
        
        # Search for unread emails
        status, messages = mail.search(None, "UNSEEN")
        if status != "OK":
            print("No messages found!")
            return
        
        # Get the list of email IDs
        email_ids = messages[0].split()
        if not email_ids:
            print("No new emails to process")
            return
        
        print(f"Found {len(email_ids)} new emails to process")
        
        for email_id in email_ids:
            # Fetch the email
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            if status != "OK":
                print(f"Error fetching email {email_id}")
                continue
            
            # Parse the email
            msg = email.message_from_bytes(msg_data[0][1])
            
            # Get email details
            subject = decode_email_header(msg["Subject"])
            from_email = decode_email_header(msg["From"])
            body = get_email_body(msg)
            
            if not body:
                print(f"Could not extract body from email: {subject}")
                continue
            
            print(f"Processing email: {subject}")
            
            # Send to API for classification
            result = classify_single_email(api_url, body, subject)
            
            if result:
                print(f"Successfully processed email from {from_email}")
                # You could save results to a database or take other actions here
        
        # Close the connection
        mail.close()
        mail.logout()
    
    except Exception as e:
        print(f"Error connecting to email server: {e}")

def main():
    parser = argparse.ArgumentParser(description='Email Classification Client')
    parser.add_argument('--mode', choices=['single', 'inbox'], default='single',
                        help='Mode: single email or process inbox')
    parser.add_argument('--api', default='https://rithuikprakash-email-classifier.hf.space/api/classify',
                        help='API URL')
    parser.add_argument('--email', help='Email text for single mode')
    parser.add_argument('--subject', default='', help='Email subject for single mode')
    parser.add_argument('--file', help='File containing email text')
    parser.add_argument('--config', default='email_config.ini', help='Config file for inbox mode')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        email_text = ""
        if args.email:
            email_text = args.email
        elif args.file and os.path.exists(args.file):
            with open(args.file, 'r', encoding='utf-8') as f:
                email_text = f.read()
        else:
            print("Please provide email text via --email or --file")
            return
        
        classify_single_email(args.api, email_text, args.subject)
    
    elif args.mode == 'inbox':
        if not os.path.exists(args.config):
            print(f"Config file not found: {args.config}")
            print("Creating a template config file. Please edit it with your email credentials.")
            
            config = configparser.ConfigParser()
            config['EMAIL'] = {
                'imap_server': 'imap.gmail.com',
                'email': 'your_email@gmail.com',
                'password': 'your_app_password',  # Use app password for Gmail
                'check_interval': '300'  # seconds
            }
            
            with open(args.config, 'w') as f:
                config.write(f)
            
            return
        
        config = configparser.ConfigParser()
        config.read(args.config)
        
        # Process inbox once
        process_inbox(args.api, config)
        
        # If check_interval is set, continue processing
        if 'check_interval' in config['EMAIL']:
            interval = int(config['EMAIL']['check_interval'])
            if interval > 0:
                print(f"Continuing to check inbox every {interval} seconds. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(interval)
                        process_inbox(args.api, config)
                except KeyboardInterrupt:
                    print("Stopped by user")

if __name__ == "__main__":
    main()
