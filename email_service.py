import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587
SMTP_LOGIN = "aad7e6001@smtp-brevo.com"
SMTP_PASSWORD = "q87EkQaMUbfRJz5w"
FROM_EMAIL = "kenzykhaled660@gmail.com"

def generate_otp() -> str:
    return str(random.randint(100000, 999999))

def send_otp_email(to_email: str, otp: str, name: str):
    msg = MIMEMultipart()
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    msg['Subject'] = "Password Reset OTP"

    body = f"""
    Hello {name},

    Your OTP for password reset is: {otp}

    This OTP is valid for 10 minutes.

    If you did not request this, please ignore this email.
    """
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_LOGIN, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())