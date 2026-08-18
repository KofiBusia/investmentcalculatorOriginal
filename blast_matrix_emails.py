# Blast assessment email to all contacts in yiap_participation_matrix.csv

import csv
import sys
import time
from app import app, mail
from flask_mail import Message

CSV_PATH = r"C:\Users\kkyei\Desktop\yiap_participation_matrix.csv"
SUBJECT  = "Action Required: Complete Your Payment to Receive Your YIAP Certificate"

def html_body(name):
    first = name.strip().split()[0].title() if name.strip() else "there"
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:32px 0;">
  <tr><td align="center">
    <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.08);">

      <!-- Header -->
      <tr>
        <td style="background:#0d2b55;padding:28px 36px;text-align:center;">
          <p style="margin:0;font-size:13px;color:#7eb3e0;letter-spacing:2px;text-transform:uppercase;">Young Investors Analyst Program</p>
          <h1 style="margin:6px 0 0;font-size:22px;color:#ffffff;font-weight:700;">InvestIQ</h1>
        </td>
      </tr>

      <!-- Body -->
      <tr>
        <td style="padding:36px 36px 24px;">
          <p style="margin:0 0 16px;font-size:15px;color:#374151;">Dear {first},</p>
          <p style="margin:0 0 20px;font-size:15px;color:#374151;line-height:1.6;">
            Thank you for being part of the <strong>Young Investors Analyst Programme</strong>.
            Whether you have already completed the assessment or are yet to take it —
            <strong>your certificate will only be issued once payment is confirmed.</strong>
          </p>

          <!-- Two-path banner -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px;border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">
            <tr style="background:#eff6ff;">
              <td style="padding:14px 20px;border-bottom:1px solid #e5e7eb;">
                <p style="margin:0;font-size:13px;font-weight:700;color:#1e40af;">&#10003; Already completed the assessment?</p>
                <p style="margin:4px 0 0;font-size:13px;color:#374151;">Great work! Simply make your payment below and your certificate will be processed.</p>
              </td>
            </tr>
            <tr style="background:#f0fdf4;">
              <td style="padding:14px 20px;">
                <p style="margin:0;font-size:13px;font-weight:700;color:#166534;">&#9654; Haven't taken all the assessments yet?</p>
                <p style="margin:4px 0 0;font-size:13px;color:#374151;">All study materials and practice questions are available at the link below — complete the remaining tests today.</p>
              </td>
            </tr>
          </table>

          <!-- Pricing Table -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px;border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">
            <tr style="background:#f8fafc;">
              <td style="padding:14px 20px;border-bottom:1px solid #e5e7eb;">
                <p style="margin:0;font-size:14px;font-weight:700;color:#0d2b55;">&#9733; Complete &amp; Pass &nbsp;—&nbsp; <span style="color:#c9a02c;">GH&#8373;70</span></p>
                <p style="margin:4px 0 0;font-size:13px;color:#6b7280;">&#10003; Certificate of Completion &nbsp;&nbsp; &#10003; Certificate of Participation</p>
              </td>
            </tr>
            <tr style="background:#ffffff;">
              <td style="padding:14px 20px;">
                <p style="margin:0;font-size:14px;font-weight:700;color:#0d2b55;">&#9675; Participation Only &nbsp;—&nbsp; <span style="color:#c9a02c;">GH&#8373;30</span></p>
                <p style="margin:4px 0 0;font-size:13px;color:#6b7280;">&#10003; Certificate of Participation</p>
              </td>
            </tr>
          </table>

          <!-- Payment Details -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px;border-radius:10px;overflow:hidden;background:#fffbeb;border:1px solid #fcd34d;">
            <tr>
              <td style="padding:16px 20px;">
                <p style="margin:0 0 6px;font-size:13px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:1px;">&#128179; How to Pay</p>
                <p style="margin:0 0 4px;font-size:14px;color:#374151;">Send your payment via <strong>Mobile Money</strong> to:</p>
                <p style="margin:6px 0 2px;font-size:20px;font-weight:700;color:#0d2b55;letter-spacing:2px;">0245871167</p>
                <p style="margin:0;font-size:13px;color:#6b7280;">Account Name: <strong>Young Investors Network</strong></p>
                <p style="margin:8px 0 0;font-size:12px;color:#9ca3af;">After payment, visit the link below to complete your registration.</p>
              </td>
            </tr>
          </table>

          <p style="margin:0 0 8px;font-size:14px;color:#374151;line-height:1.6;">
            &#128218; All <strong>study materials and practice questions</strong> are also available at the same link — use them to prepare or complete remaining tests.
          </p>

          <!-- CTA Button -->
          <table cellpadding="0" cellspacing="0" style="margin:28px auto;">
            <tr>
              <td align="center" style="background:#097a6e;border-radius:8px;">
                <a href="https://investright.onrender.com/yiap-practice"
                   style="display:inline-block;padding:14px 36px;font-size:15px;font-weight:700;color:#ffffff;text-decoration:none;letter-spacing:.3px;">
                  Access Assessment &amp; Materials &rarr;
                </a>
              </td>
            </tr>
          </table>

          <p style="margin:0;font-size:14px;color:#6b7280;line-height:1.6;text-align:center;">
            <em>Test Your Knowledge &nbsp;&bull;&nbsp; Validate Your Skills &nbsp;&bull;&nbsp; Earn Your Certification</em>
          </p>
        </td>
      </tr>

      <!-- Footer -->
      <tr>
        <td style="background:#f8fafc;padding:20px 36px;border-top:1px solid #e5e7eb;text-align:center;">
          <p style="margin:0;font-size:12px;color:#9ca3af;">
            Young Investors Analyst Program &nbsp;&bull;&nbsp; InvestIQ<br>
            Building the Next Generation of Financial Analysts
          </p>
        </td>
      </tr>

    </table>
  </td></tr>
</table>
</body>
</html>"""

def plain_body(name):
    first = name.strip().split()[0].title() if name.strip() else "there"
    return f"""Dear {first},

Thank you for being part of the Young Investors Analyst Programme.
Whether you have already completed the assessment or are yet to take it — your certificate will only be issued once payment is confirmed.

Already completed the assessment?
  Great work! Simply make your payment below and your certificate will be processed.

Haven't taken all the assessments yet?
  All study materials and practice questions are available at the link below — complete the remaining tests today.

CHOOSE YOUR TRACK:
  Complete & Pass — GHc70
    Certificate of Completion + Certificate of Participation

  Participation Only — GHc30
    Certificate of Participation

HOW TO PAY:
  Send Mobile Money to: 0245871167
  Account Name: Young Investors Network
  After payment, visit the link below to complete your registration.

ACCESS ASSESSMENT & MATERIALS:
https://investright.onrender.com/yiap-practice

Test Your Knowledge • Validate Your Skills • Earn Your Certification

Warm regards,
Kofi Kyei
Young Investors Analyst Program | InvestIQ
Building the Next Generation of Financial Analysts
"""

def run():
    with app.app_context():
        contacts = []
        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name  = row.get('Full Name', '').strip()
                email = row.get('Email', '').strip()
                if email:
                    contacts.append((name, email))

        total = len(contacts)
        print(f"Loaded {total} contacts from CSV. Starting blast...")

        sent   = 0
        failed = []

        for name, email in contacts:
            try:
                msg = Message(
                    subject=SUBJECT,
                    sender=("InvestIQ — YIAP", app.config.get('MAIL_USERNAME')),
                    recipients=[email],
                )
                msg.html = html_body(name)
                msg.body = plain_body(name)
                mail.send(msg)
                sent += 1
                print(f"  [{sent}/{total}] Sent → {email} ({name})")
                time.sleep(0.5)
            except Exception as e:
                failed.append(email)
                print(f"  FAILED → {email}: {e}", file=sys.stderr)

        print(f"\nDone. {sent} sent, {len(failed)} failed.")
        if failed:
            print("Failed addresses:")
            for addr in failed:
                print(f"  {addr}")

if __name__ == "__main__":
    run()
