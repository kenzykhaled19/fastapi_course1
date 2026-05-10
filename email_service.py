import resend
import random

resend.api_key = "re_ebFrvGgb_CR2G4NmSqkBG3cBXLssprMrY"  

def generate_otp() -> str:
    return str(random.randint(100000, 999999))

def send_otp_email(to_email: str, otp: str, name: str):
    resend.Emails.send({
        "from": "onboarding@resend.dev",
        "to": to_email,
        "subject": "Password Reset OTP",
        "html": f"""
        <p>Hello {name},</p>
        <p>Your OTP for password reset is: <strong>{otp}</strong></p>
        <p>This OTP is valid for 10 minutes.</p>
        <p>If you did not request this, please ignore this email.</p>
        """
    })