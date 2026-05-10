import random
import requests
import os
from dotenv import load_dotenv
load_dotenv()
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
FROM_EMAIL = "kenzykhaled660@gmail.com"
FROM_NAME = "Hydroscope"

def generate_otp() -> str:
    return str(random.randint(100000, 999999))

def send_otp_email(to_email: str, otp: str, name: str):
    url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "accept": "application/json",
        "api-key": BREVO_API_KEY,
        "content-type": "application/json"
    }
    payload = {
        "sender": {"name": FROM_NAME, "email": FROM_EMAIL},
        "to": [{"email": to_email}],
        "subject": "Password Reset OTP",
        "htmlContent": f"""
        <p>Hello {name},</p>
        <p>Your OTP for password reset is: <strong>{otp}</strong></p>
        <p>This OTP is valid for 10 minutes.</p>
        <p>If you did not request this, please ignore this email.</p>
        """
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 201:
        raise Exception(f"Failed to send email: {response.text}")