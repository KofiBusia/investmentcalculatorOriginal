"""Generate personalised YIAP certificates and email each recipient."""
import os, io, time
from PIL import Image, ImageDraw, ImageFont
from app import app, mail
from flask_mail import Message

CERT_PATH  = r"C:\Users\kkyei\Desktop\yiap certificate.PNG"
BLANK_TOP  = 1760   # white space starts here (below "presented to")
UNDERLINE_Y = 2072  # underline sits here

# Name as it appears on certificate → email
# Duplicates (Osei Agyemang Sarfo #9, Ezekiel Agyei #14) removed
RECIPIENTS = [
    ("Theophilus Tawiah",              "theo.tawiah62@gmail.com"),
    ("Josephine Odeibea Oduro",        "josephine.oduro@stu.ucc.edu.gh"),
    ("Bernard Okomfo",                 "okomfobernard24@gmail.com"),
    ("Dodoo Rebecca Naa Densua",       "dodoorebecca31@gmail.com"),
    ("Ofosu Naomi Boatemaa",           "naomiofosuu@gmail.com"),
    ("Osei Agyemang Sarfo",            "oseiagyemangsarfo@gmail.com"),
    ("Ezekiel Agyei",                  "ezekielagyei123@gmail.com"),
    ("Magdalene Awennisoa Afelik",     "afelikmagdalene@gmail.com"),
    ("Betty Esi Sam",                  "bettyesisam7@gmail.com"),
    ("Ahiataku Paul Etornam",          "paulkahiataku@gmail.com"),
    ("Isaac Ampofo",                   "isaac87ampofo@gmail.com"),
    ("Andrew Kwabena Panor Asare",     "asareandrew245@gmail.com"),
    ("Adama Salifu",                   "adamasalifu90@gmail.com"),
    ("Daniel Kwame Adikah",            "adikadaniel12@gmail.com"),
    ("Jason Joy Lamptey",              "llampteyjason@gmail.com"),
    ("Ayeyi Korankyewaa Hayford",      "hayfordayeyi@gmail.com"),
    ("Christian Nii Lantei Golightly", "christgolightly@gmail.com"),
    ("Vivian Konadu Yeboah",           "viviyeboah56@gmail.com"),
    ("Micheal Kesse Addo",             "michealkesseaddo@gmail.com"),
    ("Agyemfra Acheampong Rebecca",    "agyemfrarebecca@gmail.com"),
    ("Felix Owusu Nana Yaw Junior",    "felixowusu383@gmail.com"),
    ("Isaac Sarpong",                  "isaacsarpong127@gmail.com"),
    ("Sesi Esi Dede Kuwornoo",         "sesikuwornoo@gmail.com"),
    ("George Yeboah",                  "bestofgesy@gmail.com"),
    ("Sampson Amoako-Gyamfi",          "akwasiamoako2004@gmail.com"),
    ("Wilhelmina Elinam Adjah",        "wilhelminaelinamadjah@gmail.com"),
    ("Winfred Oko Anang",              "anangwinfred33@gmail.com"),
]

FONT_PATHS = [
    r"C:\Windows\Fonts\georgiai.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
    r"C:\Windows\Fonts\Arial.ttf",
]
FONT_PATH = next(p for p in FONT_PATHS if os.path.exists(p))

MAX_TEXT_WIDTH = 1500   # leave side margins on 2480px canvas

def make_cert(name):
    img  = Image.open(CERT_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)
    W    = img.width

    # Auto-scale font so long names still fit
    font_size = 90
    while font_size > 40:
        font = ImageFont.truetype(FONT_PATH, font_size)
        bbox = draw.textbbox((0, 0), name, font=font)
        if (bbox[2] - bbox[0]) <= MAX_TEXT_WIDTH:
            break
        font_size -= 4

    bbox   = draw.textbbox((0, 0), name, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (W - text_w) // 2 - bbox[0]
    y = BLANK_TOP + (UNDERLINE_Y - BLANK_TOP - text_h) // 2 - bbox[1]

    draw.text((x, y), name, fill="#0D2B55", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(300, 300))
    buf.seek(0)
    return buf.read()


SUBJECT = "Your YIAP Certificate of Achievement — Young Investors Analyst Programme"

def html_email(first):
    return f"""\
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
        <p style="margin:0 0 16px;font-size:15px;color:#374151;">Dear {first},</p>
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

def plain_email(first, name):
    return f"""\
Dear {first},

Congratulations on successfully completing the Young Investors Analyst Programme (YIAP).

Your Certificate of Achievement is attached to this email. It recognises your competence in
analysing financial statements and evaluating the performance of companies listed on the
Ghana Stock Exchange.

We are proud of your dedication throughout the programme and wish you all the best as you
continue your journey in finance and investment.

Organised by Young Investors Network (YIN)
In collaboration with Ghana Stock Exchange (GSE) & Central Securities Depository (CSD)
"""


with app.app_context():
    total  = len(RECIPIENTS)
    sent   = 0
    failed = []

    for name, email in RECIPIENTS:
        first = name.strip().split()[0].title()
        try:
            cert_bytes = make_cert(name)

            msg = Message(
                subject=SUBJECT,
                sender=("InvestIQ — YIAP", app.config.get('MAIL_USERNAME')),
                recipients=[email],
            )
            msg.html = html_email(first)
            msg.body = plain_email(first, name)
            msg.attach(
                filename     = f"YIAP Certificate — {name}.png",
                content_type = "image/png",
                data         = cert_bytes,
            )
            mail.send(msg)
            sent += 1
            print(f"  [{sent}/{total}] Sent → {email} ({name})")
            time.sleep(0.5)
        except Exception as e:
            failed.append((name, email, str(e)))
            print(f"  FAILED → {email} ({name}): {e}")

    print(f"\nDone. {sent} sent, {len(failed)} failed.")
    if failed:
        print("Failed:")
        for n, e, err in failed:
            print(f"  {n} <{e}>: {err}")
