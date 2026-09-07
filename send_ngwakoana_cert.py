"""Send Ngwakoana Succes Ngele's YIAP certificate — PNG + PDF, CC Kofi."""
import io, os, ssl, smtplib
from PIL import Image, ImageDraw, ImageFont
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

load_dotenv()
SENDER_EMAIL = os.getenv('MAIL_USERNAME')
SENDER_PASS  = os.getenv('MAIL_PASSWORD')

CERT_PATH   = r"C:\Users\kkyei\Desktop\yiap certificate.PNG"
NAME        = "Ngwakoana Succes Ngele"
TO_EMAIL    = "Ngwakoanasucces@gmail.com"
CC_EMAIL    = "kyeikofi@gmail.com"
BLANK_TOP   = 1760
UNDERLINE_Y = 2072

FONT_PATHS = [
    r"C:\Windows\Fonts\georgiai.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
    r"C:\Windows\Fonts\Arial.ttf",
]
FONT_PATH = next(p for p in FONT_PATHS if os.path.exists(p))

# ── Stamp certificate ─────────────────────────────────────────
img  = Image.open(CERT_PATH).convert("RGB")
draw = ImageDraw.Draw(img)
W    = img.width

font_size = 90
while font_size > 40:
    font = ImageFont.truetype(FONT_PATH, font_size)
    bbox = draw.textbbox((0, 0), NAME, font=font)
    if (bbox[2] - bbox[0]) <= 1500:
        break
    font_size -= 4

bbox   = draw.textbbox((0, 0), NAME, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
x = (W - text_w) // 2 - bbox[0]
y = BLANK_TOP + (UNDERLINE_Y - BLANK_TOP - text_h) // 2 - bbox[1]
draw.text((x, y), NAME, fill="#0D2B55", font=font)

# PNG bytes
png_buf = io.BytesIO()
img.save(png_buf, format="PNG", dpi=(300, 300))
png_bytes = png_buf.getvalue()

# PDF bytes
pdf_buf = io.BytesIO()
img.save(pdf_buf, format="PDF", resolution=300)
pdf_bytes = pdf_buf.getvalue()

# ── Email ─────────────────────────────────────────────────────
SUBJECT = "Your YIAP Certificate of Achievement — Young Investors Analyst Programme"

HTML = """\
<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:32px 0;">
  <tr><td align="center">
    <table width="600" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.08);">
      <tr><td style="background:#0d2b55;padding:28px 36px;text-align:center;">
        <p style="margin:0;font-size:12px;color:#7eb3e0;letter-spacing:2px;text-transform:uppercase;">Young Investors Analyst Programme</p>
        <h1 style="margin:6px 0 0;font-size:22px;color:#fff;font-weight:700;">InvestIQ</h1>
      </td></tr>
      <tr><td style="padding:36px 36px 28px;">
        <p style="margin:0 0 16px;font-size:15px;color:#374151;">Dear Ngwakoana,</p>
        <p style="margin:0 0 20px;font-size:15px;color:#374151;line-height:1.7;">
          Congratulations on successfully completing the <strong>Young Investors Analyst Programme (YIAP)</strong>.
          Your <strong>Certificate of Achievement</strong> is attached to this email in both
          <strong>PDF</strong> and <strong>image (PNG)</strong> formats.
        </p>
        <p style="margin:0 0 20px;font-size:15px;color:#374151;line-height:1.7;">
          This certificate recognises your competence in analysing financial statements and evaluating the
          performance of companies listed on the Ghana Stock Exchange — a valuable skill that sets you apart
          as a young finance professional.
        </p>
        <p style="margin:0 0 20px;font-size:15px;color:#374151;line-height:1.7;">
          We are proud of your dedication and commitment throughout the programme. We wish you all the
          best as you continue your journey in finance and investment.
        </p>
        <p style="margin:0;font-size:13px;color:#6b7280;text-align:center;line-height:1.6;">
          <em>Organised by Young Investors Network (YIN) &bull; In collaboration with GSE &amp; CSD</em>
        </p>
      </td></tr>
      <tr><td style="background:#f8fafc;padding:18px 36px;border-top:1px solid #e5e7eb;text-align:center;">
        <p style="margin:0;font-size:12px;color:#9ca3af;">
          Young Investors Analyst Programme &bull; InvestIQ<br>
          Building the Next Generation of Financial Analysts
        </p>
      </td></tr>
    </table>
  </td></tr>
</table>
</body></html>"""

PLAIN = """\
Dear Ngwakoana,

Congratulations on successfully completing the Young Investors Analyst Programme (YIAP).

Your Certificate of Achievement is attached in both PDF and image (PNG) formats.
It recognises your competence in analysing financial statements and evaluating the
performance of companies listed on the Ghana Stock Exchange.

We are proud of your dedication throughout the programme and wish you all the best.

Organised by Young Investors Network (YIN)
In collaboration with Ghana Stock Exchange (GSE) & Central Securities Depository (CSD)
"""

msg = MIMEMultipart('mixed')
msg['Subject'] = SUBJECT
msg['From']    = f"InvestIQ - YIAP <{SENDER_EMAIL}>"
msg['To']      = TO_EMAIL
msg['Cc']      = CC_EMAIL

alt = MIMEMultipart('alternative')
alt.attach(MIMEText(PLAIN, 'plain'))
alt.attach(MIMEText(HTML,  'html'))
msg.attach(alt)

for data, fname, ctype in [
    (png_bytes, f"YIAP Certificate - {NAME}.png", "image/png"),
    (pdf_bytes, f"YIAP Certificate - {NAME}.pdf", "application/pdf"),
]:
    part = MIMEBase(*ctype.split('/'))
    part.set_payload(data)
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename=fname)
    msg.attach(part)

ctx = ssl.create_default_context()
with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ctx) as server:
    server.login(SENDER_EMAIL, SENDER_PASS)
    server.sendmail(SENDER_EMAIL, [TO_EMAIL, CC_EMAIL], msg.as_bytes())

print(f"Sent to {TO_EMAIL}, CC: {CC_EMAIL} — PNG + PDF attached.")
