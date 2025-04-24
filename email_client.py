import imaplib
import email
import json
import requests
import schedule
import time
import os
from email.header import decode_header
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os
from dotenv import load_dotenv

# Improved logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='email_automation.log',
    filemode='a'  # Append mode instead of overwrite
)

class EmailProcessor:
    def __init__(self, email_address: str, app_password: str):
        self.email_address = email_address
        self.app_password = app_password
        self.imap_server = "imap.gmail.com"
        self.api_endpoint = "http://0.0.0.0:8000"
        self.processed_emails = set()
        # Add a session object for better connection handling
        self.session = requests.Session()

    def connect_to_gmail(self) -> Optional[imaplib.IMAP4_SSL]:
        try:
            imap = imaplib.IMAP4_SSL(self.imap_server)
            imap.login(self.email_address, self.app_password)
            return imap
        except imaplib.IMAP4.error as e:
            logging.error(f"IMAP error when connecting to Gmail: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Failed to connect to Gmail: {str(e)}")
            return None

    def get_email_body(self, msg) -> str:
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        try:
                            body = part.get_payload(decode=True)
                            if body:
                                return body.decode('utf-8', errors='replace')
                        except Exception as e:
                            logging.warning(f"Error decoding email part: {str(e)}")
                            continue
            
            # Not multipart - get payload directly
            try:
                body = msg.get_payload(decode=True)
                if body:
                    return body.decode('utf-8', errors='replace')
            except Exception as e:
                logging.warning(f"Error decoding email body: {str(e)}")
            
            return "No readable content found"
        except Exception as e:
            logging.error(f"Error extracting email body: {str(e)}")
            return "Error extracting email content"

    def process_email(self, email_data: Dict) -> Optional[Dict]:
        try:
            # Add retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.post(
                        self.api_endpoint,
                        json={"email": email_data["body"]},
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"API request attempt {attempt+1} failed: {str(e)}. Retrying...")
                        time.sleep(2)  # Wait before retrying
                    else:
                        logging.error(f"API request failed after {max_retries} attempts: {str(e)}")
                        return None
        except Exception as e:
            logging.error(f"Unexpected error in process_email: {str(e)}")
            return None

    def fetch_and_process_emails(self):
        logging.info("Starting to fetch and process emails")
        imap = self.connect_to_gmail()
        if not imap:
            logging.error("Could not connect to Gmail. Skipping this run.")
            return

        try:
            status, mailboxes = imap.list()
            if status != 'OK':
                logging.error(f"Failed to list mailboxes: {status}")
                return

            status, messages_count = imap.select('INBOX')
            if status != 'OK':
                logging.error(f"Failed to select INBOX: {status}")
                return
                
            logging.info(f"Connected to INBOX, found {messages_count[0].decode()} messages")
            
            # Search for unseen messages
            status, messages = imap.search(None, 'UNSEEN')
            if status != 'OK':
                logging.error(f"Failed to search for UNSEEN messages: {status}")
                return
                
            message_ids = messages[0].split()
            logging.info(f"Found {len(message_ids)} unread messages")

            for msg_num in message_ids:
                try:
                    status, msg_data = imap.fetch(msg_num, '(RFC822)')
                    if status != 'OK':
                        logging.error(f"Failed to fetch message {msg_num}: {status}")
                        continue
                        
                    email_body = email.message_from_bytes(msg_data[0][1])
                    
                    message_id = email_body.get('Message-ID', '')
                    if message_id in self.processed_emails:
                        logging.info(f"Skipping already processed email: {message_id}")
                        continue

                    # Get subject with proper decoding
                    subject_parts = decode_header(email_body.get("subject", "No Subject"))
                    subject = ""
                    for part, encoding in subject_parts:
                        if isinstance(part, bytes):
                            subject += part.decode(encoding or 'utf-8', errors='replace')
                        else:
                            subject += str(part)

                    sender = email_body.get("from", "Unknown Sender")
                    date = email_body.get("date", "Unknown Date")
                    
                    logging.info(f"Processing email: '{subject}' from {sender}")

                    email_content = {
                        "subject": subject,
                        "body": self.get_email_body(email_body),
                        "date": date,
                        "from": sender
                    }

                    result = self.process_email(email_content)
                    if result:
                        logging.info(f"Successfully processed email: {subject}")
                        logging.info(f"Classification result: {result}")
                        self.processed_emails.add(message_id)
                    else:
                        logging.warning(f"Failed to process email: {subject}")
                    
                except Exception as e:
                    logging.error(f"Error processing individual email: {str(e)}")

        except Exception as e:
            logging.error(f"Error in fetch_and_process_emails: {str(e)}")
        finally:
            try:
                imap.close()
                logging.info("IMAP connection closed")
            except:
                pass
            try:
                imap.logout()
                logging.info("IMAP logout successful")
            except:
                pass

def main():
    # Get credentials from environment variables
    load_dotenv()
    email_address = os.getenv("GMAIL_ADDRESS")
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not email_address or not app_password:
        logging.error("Missing email credentials. Please set GMAIL_ADDRESS and GMAIL_APP_PASSWORD environment variables.")
        print("Error: Missing email credentials. Check the log for details.")
        return

    processor = EmailProcessor(email_address, app_password)
    
    try:
        # Initial run
        logging.info("Starting initial email processing run")
        processor.fetch_and_process_emails()
        
        # Schedule to run every 5 minutes
        schedule.every(5).minutes.do(processor.fetch_and_process_emails)

        logging.info("Email automation started - checking every 5 minutes")
        print("Email automation started - checking every 5 minutes. See logs for details.")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Email automation stopped by user")
        print("Email automation stopped")
    except Exception as e:
        logging.critical(f"Fatal error in main loop: {str(e)}")
        print(f"Fatal error: {str(e)}. Check logs for details.")

if __name__ == "__main__":
    main()