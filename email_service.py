import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

EMAIL = "kenzykhaled660@gmail.com"
PASSWORD = "pinr twxt zimh hmdp"

def generate_otp() -> str:
    return str(random.randint(100000, 999999))

def send_otp_email(to_email: str, otp: str, name: str):
    msg = MIMEMultipart()
    msg['From'] = EMAIL
    msg['To'] = to_email
    msg['Subject'] = "Password Reset OTP"

    body = f"""
    Hello {name},

    Your OTP for password reset is: {otp}

    This OTP is valid for 10 minutes.

    If you did not request this, please ignore this email.
    """

    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, to_email, msg.as_string())