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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='email_automation.log'
)

class EmailProcessor:
    def __init__(self, email_address: str, app_password: str):
        self.email_address = email_address
        self.app_password = app_password
        self.imap_server = "imap.gmail.com"
        self.api_endpoint = "https://huggingface.co/spaces/rithuikprakash/Intern/api/classify"
        self.processed_emails = set()

    def connect_to_gmail(self) -> Optional[imaplib.IMAP4_SSL]:
        try:
            imap = imaplib.IMAP4_SSL(self.imap_server)
            imap.login(self.email_address, self.app_password)
            return imap
        except Exception as e:
            logging.error(f"Failed to connect to Gmail: {str(e)}")
            return None

    def get_email_body(self, msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode()
        return msg.get_payload(decode=True).decode()

    def process_email(self, email_data: Dict) -> Optional[Dict]:
        try:
            response = requests.post(
                self.api_endpoint,
                json={"email": email_data["body"]},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {str(e)}")
            return None

    def fetch_and_process_emails(self):
        imap = self.connect_to_gmail()
        if not imap:
            return

        try:
            imap.select('INBOX')
            _, messages = imap.search(None, 'UNSEEN')

            for msg_num in messages[0].split():
                try:
                    _, msg_data = imap.fetch(msg_num, '(RFC822)')
                    email_body = email.message_from_bytes(msg_data[0][1])
                    
                    message_id = email_body.get('Message-ID', '')
                    if message_id in self.processed_emails:
                        continue

                    subject = decode_header(email_body["subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode()

                    email_content = {
                        "subject": subject,
                        "body": self.get_email_body(email_body),
                        "date": email_body["date"],
                        "from": email_body["from"]
                    }

                    result = self.process_email(email_content)
                    if result:
                        logging.info(f"Successfully processed email: {subject}")
                        logging.info(f"Classification result: {result}")
                        self.processed_emails.add(message_id)
                    
                except Exception as e:
                    logging.error(f"Error processing individual email: {str(e)}")

        except Exception as e:
            logging.error(f"Error in fetch_and_process_emails: {str(e)}")
        finally:
            try:
                imap.close()
                imap.logout()
            except:
                pass

def main():
    email_address = os.getenv("GMAIL_ADDRESS")
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not email_address or not app_password:
        logging.error("Missing email credentials. Please set GMAIL_ADDRESS and GMAIL_APP_PASSWORD environment variables.")
        return

    processor = EmailProcessor(email_address, app_password)
    
    # Initial run
    processor.fetch_and_process_emails()
    
    # Schedule to run every 5 minutes
    schedule.every(5).minutes.do(processor.fetch_and_process_emails)

    logging.info("Email automation started - checking every 5 minutes")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
