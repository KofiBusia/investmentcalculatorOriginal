from app import app, mail
from flask_mail import Message
from blast_payment_reminder import html_body, plain_body, SUBJECT

contacts = [
    ("Sampson Amoako-Gyamfi", "akwasiamoako2004@gmail.com", True),
    ("",                       "anettehamilton2@gmail.com",  False),
    ("",                       "anangwinfred33@gmail.com",   False),
    ("",                       "buaduvictoroa1@gmail.com",   False),
]

with app.app_context():
    for name, email, completed_all in contacts:
        msg = Message(
            subject=SUBJECT,
            sender=("InvestIQ — YIAP", app.config.get('MAIL_USERNAME')),
            recipients=[email],
        )
        msg.html = html_body(name, completed_all)
        msg.body = plain_body(name, completed_all)
        mail.send(msg)
        tag = "All 3 done" if completed_all else "Incomplete"
        print(f"Sent → {email} ({name or 'Participant'}) [{tag}]")

print("Done.")
