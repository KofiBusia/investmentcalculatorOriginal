"""NSS accounting role blast to YIAP participants."""
import os, ssl, smtplib, time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()
SENDER_EMAIL = os.getenv('MAIL_USERNAME')
SENDER_PASS  = os.getenv('MAIL_PASSWORD')

SUBJECT = "STILL OPEN — National Service Accounting Role"

RECIPIENTS = [
    "asareandrew245@gmail.com",
    "angmortjohn@gmail.com",
    "nicecewell@gmail.com",
    "hayfordayeyi@gmail.com",
    "davidbati85@gmail.com",
    "wassilazakarimustapha@gmail.com",
    "abrokwahgyamfiboakyewaa@gmail.com",
    "quarcooisrael1@gmail.com",
    "viviyeboah56@gmail.com",
    "michaelbonni.93@gmail.com",
    "gyedueben18@gmail.com",
    "senabenton@gmail.com",
    "easumanimensah@gmail.com",
    "oluwafemiolatunji93@gmail.com",
    "ibenjy7@gmail.com",
    "isaacaidooerzuah@gmail.com",
    "jeremensah77@gmail.com",
    "gbewonyoerica8@gmail.com",
    "dikajosephine66@gmail.com",
    "osei31492@gmail.com",
    "okomfobernard24@gmail.com",
    "favourikechukwu495@gmail.com",
    "debrahb94@gmail.com",
    "merajanuwe9@gmail.com",
    "marfopapayaw@gmail.com",
    "nuhurahana756@gmail.com",
    "afelikmagdalene@gmail.com",
    "sesikuwornoo@gmail.com",
    "sharkoor94@gmail.com",
    "isaac87ampofo@gmail.com",
    "azurkokougbilah1@gmail.com",
    "psomuah@rocketmail.com",
    "naomiofosuu@gmail.com",
    "paulkahiataku@gmail.com",
    "morantj577@gmail.com",
    "adamsabdul771@gmail.com",
    "bestofgesy@gmail.com",
    "oneboychicharito@gmail.com",
    "thindmanackah11@gmail.com",
    "winfredokoanang@gmail.com",
    "yvopris2023@gmail.com",
    "abdulhakimmohammed059@gmail.com",
    "ezekielagyei123@gmail.com",
    "benedictasowu03@gmail.com",
    "revbernardbaah@gmail.com",
    "larrysomuah@gmail.com",
    "justicekonlanxr@gmail.com",
    "brightokyere0000@gmail.com",
    "oseiagyemangsarfo@gmail.com",
    "isaacsarpong127@gmail.com",
    "dadziekwesikingsford17@gmail.com",
    "akwasiamoako2004@gmail.com",
    "eshun5685@gmail.com",
    "kennethacheampong023@gmail.com",
    "atadzas@gmail.com",
    "k.raphaelganyo@gmail.com",
    "samuelasiwbour44@gmail.com",
    "dagbuiedem@gmail.com",
    "adjeidoris130@gmail.com",
    "christgolightly@gmail.com",
    "iblaymouah@gmail.com",
    "felixowusu383@gmail.com",
    "sekyijehu@gmail.com",
    "samoahbaidoo@gmail.com",
    "anangchristabel373@gmail.com",
    "accamelijah755@gmail.com",
    "asedaaidoo17@gmail.com",
    "nkrumahfrank1999@gmail.com",
    "magnusdarko229@gmail.com",
    "dogbeydavina4@gmail.com",
    "dennis.timbilla@stu.ucc.edu.gh",
    "preciousadjei03@gmail.com",
    "llampteyjason@gmail.com",
    "awopetufeyisayo72@gmail.com",
    "anidavidosei@gmail.com",
    "nlantei92@gmail.com",
    "norteybenjamin000@gmail.com",
    "enuhu16@gmail.com",
    "rolmensah127@gmail.com",
    "lemmensah1234@gmail.com",
    "awukueric212@gmail.com",
    "evasheila96@gmail.com",
    "asodeangela@gmail.com",
    "mosesart1234@gmail.com",
    "damianowusu4@gmail.com",
    "dinakeren257@gmail.com",
    "dodoorebecca31@gmail.com",
    "ampofoeddy@gmail.com",
    "josephine.oduro@stu.ucc.edu.gh",
    "benspykeabbey@gmail.com",
    "bettyesisam7@gmail.com",
    "bernard.akwetey001@stu.ucc.edu.gh",
    "stephendanzerl2004@gmail.com",
    "afyadanso374@gmail.com",
    "amedorj7@gmail.com",
    "bransahakosua@gmail.com",
    "elizabethmawutor904@gmail.com",
    "lampteyeliot@gmail.com",
    "nanaadjowa831@gmail.com",
    "adamasalifu90@gmail.com",
    "nii.laryea7@yahoo.com",
    "lawrencialamptey556@gmail.com",
    "adikadaniel12@gmail.com",
    "daudabansi@gmail.com",
]

HTML = """\
<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:32px 0;">
  <tr><td align="center">
    <table width="600" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.08);">
      <tr><td style="background:#0d2b55;padding:24px 36px;text-align:center;">
        <p style="margin:0;font-size:12px;color:#7eb3e0;letter-spacing:2px;text-transform:uppercase;">Young Investors Analyst Programme</p>
        <h1 style="margin:6px 0 0;font-size:20px;color:#fff;font-weight:700;">National Service Opportunity</h1>
      </td></tr>
      <tr><td style="background:#c9a02c;padding:12px 36px;text-align:center;">
        <p style="margin:0;font-size:14px;font-weight:800;color:#fff;letter-spacing:1px;text-transform:uppercase;">&#9679; The Opportunity Is Still Open &mdash; Apply Now</p>
      </td></tr>
      <tr><td style="padding:32px 36px 28px;">
        <p style="margin:0 0 16px;font-size:15px;color:#374151;">Dear YIAP Participant,</p>
        <p style="margin:0 0 16px;font-size:15px;color:#374151;line-height:1.7;">
          A <strong>reputable firm</strong> is looking for a <strong>National Service person for an Accounting role</strong>,
          and we are opening applications to YIAP participants first.
        </p>
        <p style="margin:0 0 16px;font-size:15px;color:#374151;line-height:1.7;">
          You do not need to be a first-class student &mdash; if you have gone through the YIAP training
          and taken the tests, you qualify.
        </p>
        <p style="margin:0 0 24px;font-size:15px;color:#374151;line-height:1.7;">
          Send your CV via <strong>WhatsApp</strong> &mdash; you already have the number.
          We are <strong>still receiving CVs</strong>. Apply now.
        </p>
        <p style="margin:0;font-size:13px;color:#6b7280;line-height:1.6;">
          Warm regards,<br>
          <strong style="color:#0d2b55;">Kofi Kyei</strong><br>
          <em>Young Investors Analyst Programme</em>
        </p>
      </td></tr>
    </table>
  </td></tr>
</table>
</body></html>"""

PLAIN = """\
*** THE OPPORTUNITY IS STILL OPEN — APPLY NOW ***

Dear YIAP Participant,

A reputable firm is looking for a National Service person for an Accounting role,
and we are opening applications to YIAP participants first.

You do not need to be a first-class student — if you have gone through the YIAP
training and taken the tests, you qualify.

Send your CV via WhatsApp — you already have the number.
We are still receiving CVs. Apply now.

Warm regards,
Kofi Kyei
Young Investors Analyst Programme
"""

ctx   = ssl.create_default_context()
total = len(RECIPIENTS)
sent, failed = 0, []

for email in RECIPIENTS:
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = SUBJECT
        msg['From']    = f"Kofi Kyei - YIAP <{SENDER_EMAIL}>"
        msg['To']      = email
        msg.attach(MIMEText(PLAIN, 'plain'))
        msg.attach(MIMEText(HTML,  'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ctx) as server:
            server.login(SENDER_EMAIL, SENDER_PASS)
            server.send_message(msg)
        sent += 1
        print(f"  [{sent}/{total}] Sent → {email}", flush=True)
        time.sleep(0.4)
    except Exception as e:
        failed.append((email, str(e)))
        print(f"  FAILED → {email}: {e}", flush=True)

print(f"\nDone. {sent} sent, {len(failed)} failed.")
if failed:
    for addr, err in failed:
        print(f"  {addr}: {err}")
