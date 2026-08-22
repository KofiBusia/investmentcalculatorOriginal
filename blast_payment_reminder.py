# Payment reminder blast to yiap_participation_matrix (1).csv

import csv
import sys
import time
from app import app, mail
from flask_mail import Message

CSV_PATH = r"C:\Users\kkyei\Desktop\yiap_participation_matrix (1).csv"
SUBJECT  = "Your YIAP Certificate Is One Step Away — Confirm Your Payment"

def html_body(name, completed_all):
    first = name.strip().split()[0].title() if name.strip() else "Participant"

    if completed_all:
        status_block = """\
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:22px;border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">
          <tr style="background:#f0fdf4;">
            <td style="padding:14px 20px;">
              <p style="margin:0;font-size:13px;font-weight:700;color:#166534;">&#10003; You have completed all 3 assessments — well done!</p>
              <p style="margin:5px 0 0;font-size:13px;color:#374151;">Your certificate is ready to be issued. Simply make your payment below and it will be processed promptly.</p>
            </td>
          </tr>
        </table>"""
    else:
        status_block = """\
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:22px;border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">
          <tr style="background:#eff6ff;">
            <td style="padding:14px 20px;border-bottom:1px solid #e5e7eb;">
              <p style="margin:0;font-size:13px;font-weight:700;color:#1e40af;">&#10003; Already completed your assessments?</p>
              <p style="margin:5px 0 0;font-size:13px;color:#374151;">Simply make your payment below and your certificate will be issued.</p>
            </td>
          </tr>
          <tr style="background:#fefce8;">
            <td style="padding:14px 20px;">
              <p style="margin:0;font-size:13px;font-weight:700;color:#854d0e;">&#9654; Still have outstanding tests?</p>
              <p style="margin:5px 0 0;font-size:13px;color:#374151;">All study materials and practice questions are available at the link below — complete your remaining assessments and pay together.</p>
            </td>
          </tr>
        </table>"""

    return f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:32px 0;">
  <tr><td align="center">
    <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.08);">

      <!-- Header -->
      <tr>
        <td style="background:#0d2b55;padding:28px 36px;text-align:center;">
          <p style="margin:0;font-size:12px;color:#7eb3e0;letter-spacing:2px;text-transform:uppercase;">Young Investors Analyst Program</p>
          <h1 style="margin:6px 0 0;font-size:22px;color:#ffffff;font-weight:700;">InvestIQ</h1>
        </td>
      </tr>

      <!-- Body -->
      <tr>
        <td style="padding:36px 36px 28px;">

          <p style="margin:0 0 16px;font-size:15px;color:#374151;">Dear {first},</p>

          <p style="margin:0 0 20px;font-size:15px;color:#374151;line-height:1.7;">
            Thank you for being part of the <strong>Young Investors Analyst Programme (YIAP)</strong>.
            We are now processing certificates and your <strong>payment confirmation is the final step</strong>.
          </p>

          {status_block}

          <!-- Pricing -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:22px;border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;">
            <tr style="background:#f8fafc;">
              <td style="padding:14px 20px;border-bottom:1px solid #e5e7eb;">
                <p style="margin:0;font-size:14px;font-weight:700;color:#0d2b55;">&#9733; Completed &amp; Passed &nbsp;—&nbsp; <span style="color:#c9a02c;">GH&#8373; 70</span></p>
                <p style="margin:4px 0 0;font-size:13px;color:#6b7280;">&#10003; Certificate of Completion &nbsp;&nbsp; &#10003; Certificate of Participation</p>
              </td>
            </tr>
            <tr style="background:#ffffff;">
              <td style="padding:14px 20px;">
                <p style="margin:0;font-size:14px;font-weight:700;color:#0d2b55;">&#9675; Participation Only &nbsp;—&nbsp; <span style="color:#c9a02c;">GH&#8373; 30</span></p>
                <p style="margin:4px 0 0;font-size:13px;color:#6b7280;">&#10003; Certificate of Participation</p>
              </td>
            </tr>
          </table>

          <!-- Payment Details -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px;border-radius:10px;background:#fffbeb;border:1px solid #fcd34d;">
            <tr>
              <td style="padding:18px 22px;">
                <p style="margin:0 0 6px;font-size:12px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:1px;">&#128179; How to Pay</p>
                <p style="margin:0 0 6px;font-size:14px;color:#374151;">Send via <strong>Mobile Money</strong> to:</p>
                <p style="margin:0 0 4px;font-size:26px;font-weight:700;color:#0d2b55;letter-spacing:3px;">0245871167</p>
                <p style="margin:0;font-size:13px;color:#6b7280;">Account Name: <strong>Young Investors Network</strong></p>
              </td>
            </tr>
          </table>

          <!-- CTA -->
          <table cellpadding="0" cellspacing="0" style="margin:0 auto 24px;">
            <tr>
              <td align="center" style="background:#097a6e;border-radius:8px;">
                <a href="https://investright.onrender.com/yiap-practice"
                   style="display:inline-block;padding:14px 36px;font-size:15px;font-weight:700;color:#ffffff;text-decoration:none;">
                  Access Assessment &amp; Materials &rarr;
                </a>
              </td>
            </tr>
          </table>

          <p style="margin:0;font-size:13px;color:#6b7280;text-align:center;line-height:1.6;">
            <em>Test Your Knowledge &nbsp;&bull;&nbsp; Validate Your Skills &nbsp;&bull;&nbsp; Earn Your Certification</em>
          </p>
        </td>
      </tr>

      <!-- Footer -->
      <tr>
        <td style="background:#f8fafc;padding:18px 36px;border-top:1px solid #e5e7eb;text-align:center;">
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


def plain_body(name, completed_all):
    first = name.strip().split()[0].title() if name.strip() else "Participant"
    status = ("You have completed all 3 assessments — well done! Simply make your payment below and your certificate will be issued promptly."
              if completed_all else
              "Whether you have completed your assessments or still have outstanding tests, make your payment today and complete any remaining assessments at the link below.")
    return f"""\
Dear {first},

Thank you for being part of the Young Investors Analyst Programme (YIAP).
We are now processing certificates and your payment confirmation is the final step.

{status}

CERTIFICATE TRACKS:
  Completed & Passed all 3  —  GHc 70
    Certificate of Completion + Certificate of Participation

  Participation Only  —  GHc 30
    Certificate of Participation

HOW TO PAY:
  Send Mobile Money to: 0245871167
  Account Name: Young Investors Network

ACCESS ASSESSMENTS & MATERIALS:
  https://investright.onrender.com/yiap-practice

Test Your Knowledge • Validate Your Skills • Earn Your Certification

Warm regards,
Kofi Kyei
Young Investors Analyst Programme | InvestIQ
"""


def run():
    with app.app_context():
        contacts = []
        with open(CSV_PATH, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name  = row.get('Full Name', '').strip()
                email = row.get('Email', '').strip()
                completed = row.get('Completed All 3?', '').strip().lower() == 'yes'
                if email:
                    contacts.append((name, email, completed))

        total = len(contacts)
        print(f"Loaded {total} contacts. Starting blast...")
        sent, failed = 0, []

        for name, email, completed_all in contacts:
            try:
                msg = Message(
                    subject=SUBJECT,
                    sender=("InvestIQ — YIAP", app.config.get('MAIL_USERNAME')),
                    recipients=[email],
                )
                msg.html = html_body(name, completed_all)
                msg.body = plain_body(name, completed_all)
                mail.send(msg)
                sent += 1
                tag = "All 3 done" if completed_all else "Incomplete"
                print(f"  [{sent}/{total}] Sent → {email} ({name}) [{tag}]")
                time.sleep(0.4)
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
