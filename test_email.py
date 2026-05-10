import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

EMAIL = "kenzykhaled660@gmail.com"
PASSWORD = "pinr twxt zimh hmdp"

msg = MIMEMultipart()
msg['From'] = EMAIL
msg['To'] = EMAIL  # بعتيه لنفسك
msg['Subject'] = "Test"
msg.attach(MIMEText("test message", 'plain'))

try:
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, EMAIL, msg.as_string())
    print("✅ تم الإرسال بنجاح")
except Exception as e:
    print(f"❌ Error: {e}")