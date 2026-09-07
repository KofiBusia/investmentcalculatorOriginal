"""Stamp a name onto the YIAP certificate PNG and email it."""
import os, io
from PIL import Image, ImageDraw, ImageFont
from app import app, mail
from flask_mail import Message

CERT_PATH = r"C:\Users\kkyei\Desktop\yiap certificate.PNG"
NAME      = "Kofi Busia Kyei"
RECIPIENT = "kyeikofi@gmail.com"

# ── Stamp name onto certificate ──────────────────────────────
img  = Image.open(CERT_PATH).convert("RGB")
W, H = img.size
draw = ImageDraw.Draw(img)

# Blank area for name: y=1760 to y=2072 (real underline is at y=2075)
BLANK_TOP    = 1760
UNDERLINE_Y  = 2072

# Italic Georgia at 90px — elegant, plenty of space in the 312px blank
font_size = 90
for font_path in [
    r"C:\Windows\Fonts\georgiai.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
    r"C:\Windows\Fonts\Arial.ttf",
]:
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
        break

bbox   = draw.textbbox((0, 0), NAME, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]

# Centre horizontally; place baseline just above the underline
x = (W - text_w) // 2 - bbox[0]
y = BLANK_TOP + (UNDERLINE_Y - BLANK_TOP - text_h) // 2 - bbox[1]

draw.text((x, y), NAME, fill="#0D2B55", font=font)

# Save preview to Desktop to verify
img.save(r"C:\Users\kkyei\Desktop\cert_preview.png", dpi=(300, 300))
print(f"Preview saved. Name placed at y={y}, text_h={text_h}")

# Save to bytes for email attachment
buf = io.BytesIO()
img.save(buf, format="PNG", dpi=(300, 300))
buf.seek(0)
cert_bytes = buf.read()

# ── Email ────────────────────────────────────────────────────
SUBJECT = "Your YIAP Certificate of Achievement — Young Investors Analyst Programme"

HTML = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:32px 0;">
  <tr><td align="center">
    <table width="600" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.08);">
      <tr>
        <td style="background:#0d2b55;padding:28px 36px;text-align:center;">
          <p style="margin:0;font-size:12px;color:#7eb3e0;letter-spacing:2px;text-transform:uppercase;">Young Investors Analyst Programme</p>
          <h1 style="margin:6px 0 0;font-size:22px;color:#fff;font-weight:700;">InvestIQ</h1>
        </td>
      </tr>
      <tr>
        <td style="padding:36px 36px 28px;">
          <p style="margin:0 0 16px;font-size:15px;color:#374151;">Dear Kofi,</p>
          <p style="margin:0 0 20px;font-size:15px;color:#374151;line-height:1.7;">
            Congratulations on successfully completing the <strong>Young Investors Analyst Programme (YIAP)</strong>.
            Your <strong>Certificate of Achievement</strong> is attached to this email.
          </p>
          <p style="margin:0 0 20px;font-size:15px;color:#374151;line-height:1.7;">
            This certificate recognises your competence in analysing financial statements and evaluating the
            performance of companies listed on the Ghana Stock Exchange — a valuable skill that sets you apart
            as a young finance professional.
          </p>
          <p style="margin:0 0 20px;font-size:15px;color:#374151;line-height:1.7;">
            We wish you all the best as you continue your journey in finance and investment.
          </p>
          <p style="margin:0;font-size:13px;color:#6b7280;text-align:center;line-height:1.6;">
            <em>Organised by Young Investors Network (YIN) &bull; In collaboration with GSE &amp; CSD</em>
          </p>
        </td>
      </tr>
      <tr>
        <td style="background:#f8fafc;padding:18px 36px;border-top:1px solid #e5e7eb;text-align:center;">
          <p style="margin:0;font-size:12px;color:#9ca3af;">
            Young Investors Analyst Programme &bull; InvestIQ<br>
            Building the Next Generation of Financial Analysts
          </p>
        </td>
      </tr>
    </table>
  </td></tr>
</table>
</body>
</html>"""

PLAIN = f"""\
Dear Kofi,

Congratulations on successfully completing the Young Investors Analyst Programme (YIAP).

Your Certificate of Achievement is attached to this email. It recognises your competence in
analysing financial statements and evaluating the performance of companies listed on the
Ghana Stock Exchange.

We wish you all the best as you continue your journey in finance and investment.

Organised by Young Investors Network (YIN)
In collaboration with Ghana Stock Exchange (GSE) & Central Securities Depository (CSD)
"""

with app.app_context():
    msg = Message(
        subject=SUBJECT,
        sender=("InvestIQ — YIAP", app.config.get('MAIL_USERNAME')),
        recipients=[RECIPIENT],
    )
    msg.html  = HTML
    msg.body  = PLAIN
    msg.attach(
        filename    = "YIAP Certificate — Kofi Busia Kyei.png",
        content_type= "image/png",
        data        = cert_bytes,
    )
    mail.send(msg)
    print(f"Certificate sent to {RECIPIENT}")
