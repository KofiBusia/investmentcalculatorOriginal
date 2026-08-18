# Blast YIAP assessment invite to full email list. Joshua Mensah receives one copy.

import sys
import time
from app import app, mail
from flask_mail import Message

SUBJECT  = "Your YIAP Certificate Is One Step Away — Take the Assessment Now"
JOSHUA   = "joshuamens67@gmail.com"

EMAILS = [
    "asareandrew245@gmail.com", "angmortjohn@gmail.com", "nicecewell@gmail.com",
    "hayfordayeyi@gmail.com", "davidbati85@gmail.com", "wassilazakarimustapha@gmail.com",
    "abrokwahgyamfiboakyewaa@gmail.com", "quarcooisrael1@gmail.com", "viviyeboah56@gmail.com",
    "michaelbonni.93@gmail.com", "gyedueben18@gmail.com", "senabenton@gmail.com",
    "easumanimensah@gmail.com", "oluwafemiolatunji93@gmail.com", "ibenjy7@gmail.com",
    "isaacaidooerzuah@gmail.com", "jeremensah77@gmail.com", "gbewonyoerica8@gmail.com",
    "dikajosephine66@gmail.com", "osei31492@gmail.com", "okomfobernard24@gmail.com",
    "favourikechukwu495@gmail.com", "debrahb94@gmail.com", "merajanuwe9@gmail.com",
    "marfopapayaw@gmail.com", "nuhurahana756@gmail.com", "afelikmagdalene@gmail.com",
    "sesikuwornoo@gmail.com", "sharkoor94@gmail.com", "isaac87ampofo@gmail.com",
    "azurkokougbilah1@gmail.com", "psomuah@rocketmail.com", "naomiofosuu@gmail.com",
    "paulkahiataku@gmail.com", "morantj577@gmail.com", "adamsabdul771@gmail.com",
    "bestofgesy@gmail.com", "oneboychicharito@gmail.com", "thindmanackah11@gmail.com",
    "winfredokoanang@gmail.com", "yvopris2023@gmail.com", "abdulhakimmohammed059@gmail.com",
    "ezekielagyei123@gmail.com", "benedictasowu03@gmail.com", "revbernardbaah@gmail.com",
    "larrysomuah@gmail.com", "justicekonlanxr@gmail.com", "brightokyere0000@gmail.com",
    "oseiagyemangsarfo@gmail.com", "isaacsarpong127@gmail.com", "dadziekwesikingsford17@gmail.com",
    "akwasiamoako2004@gmail.com", "eshun5685@gmail.com", "kennethacheampong023@gmail.com",
    "atadzas@gmail.com", "k.raphaelganyo@gmail.com", "samuelasiwbour44@gmail.com",
    "dagbuiedem@gmail.com", "adjeidoris130@gmail.com", "christgolightly@gmail.com",
    "iblaymouah@gmail.com", "felixowusu383@gmail.com", "sekyijehu@gmail.com",
    "samoahbaidoo@gmail.com", "anangchristabel373@gmail.com", "accamelijah755@gmail.com",
    "asedaaidoo17@gmail.com", "nkrumahfrank1999@gmail.com", "magnusdarko229@gmail.com",
    "dogbeydavina4@gmail.com", "dennis.timbilla@stu.ucc.edu.gh", "preciousadjei03@gmail.com",
    "llampteyjason@gmail.com", "awopetufeyisayo72@gmail.com", "anidavidosei@gmail.com",
    "nlantei92@gmail.com", "norteybenjamin000@gmail.com", "enuhu16@gmail.com",
    "rolmensah127@gmail.com", "lemmensah1234@gmail.com", "awukueric212@gmail.com",
    "evasheila96@gmail.com", "asodeangela@gmail.com", "mosesart1234@gmail.com",
    "damianowusu4@gmail.com", "dinakeren257@gmail.com", "dodoorebecca31@gmail.com",
    "ampofoeddy@gmail.com", "josephine.oduro@stu.ucc.edu.gh", "benspykeabbey@gmail.com",
    "bettyesisam7@gmail.com", "bernard.akwetey001@stu.ucc.edu.gh", "stephendanzerl2004@gmail.com",
    "afyadanso374@gmail.com", "amedorj7@gmail.com", "bransahakosua@gmail.com",
    "elizabethmawutor904@gmail.com", "lampteyeliot@gmail.com", "nanaadjowa831@gmail.com",
    "adamasalifu90@gmail.com", "nii.laryea7@yahoo.com", "lawrencialamptey556@gmail.com",
    "adikadaniel12@gmail.com", "christianaagyiri246@gmail.com", "ygloriasb@gmail.com",
    "pkowiredu77@gmail.com", "theo.tawiah62@gmail.com", "enochbaffoe73@gmail.com",
]

def make_html():
    return """\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:32px 0;">
  <tr><td align="center">
    <table width="600" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.08);">

      <!-- Header -->
      <tr>
        <td style="background:#0d2b55;padding:28px 36px;text-align:center;">
          <p style="margin:0;font-size:12px;color:#7eb3e0;letter-spacing:2px;text-transform:uppercase;">Young Investors Analyst Program</p>
          <h1 style="margin:6px 0 0;font-size:22px;color:#fff;font-weight:700;">InvestIQ</h1>
        </td>
      </tr>

      <!-- Body -->
      <tr>
        <td style="padding:36px 36px 28px;">

          <p style="margin:0 0 16px;font-size:15px;color:#374151;">Dear Participant,</p>

          <p style="margin:0 0 18px;font-size:15px;color:#374151;line-height:1.7;">
            We hope this message finds you well. This is a reminder that your
            <strong>YIAP assessments are open</strong> and your certificate is within reach.
          </p>

          <!-- GSE notice -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:22px;border-radius:10px;background:#eff6ff;border:1px solid #bfdbfe;">
            <tr>
              <td style="padding:14px 18px;">
                <p style="margin:0;font-size:14px;color:#1e40af;line-height:1.6;">
                  &#128203; <strong>GSE Submission:</strong> As promised, your details will be submitted to the
                  <strong>Ghana Stock Exchange (GSE)</strong>. We encourage you to complete the assessment
                  before submission so your certificate is included.
                </p>
              </td>
            </tr>
          </table>

          <!-- Why it matters -->
          <p style="margin:0 0 18px;font-size:15px;color:#374151;line-height:1.7;">
            Your certificate demonstrates <strong>verified financial analysis skills</strong> —
            an asset that sets you apart in your job search and career progression.
          </p>

          <!-- CTA -->
          <table cellpadding="0" cellspacing="0" style="margin:0 auto 28px;">
            <tr>
              <td align="center" style="background:#097a6e;border-radius:8px;">
                <a href="https://investright.onrender.com/yiap-practice"
                   style="display:inline-block;padding:14px 40px;font-size:15px;font-weight:700;color:#fff;text-decoration:none;">
                  Take the Assessment Now &rarr;
                </a>
              </td>
            </tr>
          </table>
          <p style="margin:-18px 0 26px;font-size:12px;color:#9ca3af;text-align:center;">
            Study materials &amp; practice questions are also available at the same link.
          </p>

          <!-- Payment -->
          <table width="100%" cellpadding="0" cellspacing="0" style="border-radius:10px;background:#fffbeb;border:1px solid #fcd34d;margin-bottom:22px;">
            <tr>
              <td style="padding:16px 20px;">
                <p style="margin:0 0 8px;font-size:12px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:1px;">&#128179; Payment — to receive your certificate</p>
                <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:10px;">
                  <tr style="background:#fff8e1;">
                    <td style="padding:8px 12px;font-size:13px;color:#374151;border-radius:6px 6px 0 0;border:1px solid #fde68a;">
                      <strong>Complete &amp; Pass — GH&#8373;70</strong><br>
                      <span style="color:#6b7280;">&#10003; Certificate of Completion &nbsp; &#10003; Certificate of Participation</span>
                    </td>
                  </tr>
                  <tr style="background:#fff;">
                    <td style="padding:8px 12px;font-size:13px;color:#374151;border-radius:0 0 6px 6px;border:1px solid #fde68a;border-top:none;">
                      <strong>Participation Only — GH&#8373;30</strong><br>
                      <span style="color:#6b7280;">&#10003; Certificate of Participation</span>
                    </td>
                  </tr>
                </table>
                <p style="margin:6px 0 2px;font-size:14px;color:#374151;">Send via <strong>Mobile Money</strong> to:</p>
                <p style="margin:4px 0 2px;font-size:22px;font-weight:700;color:#0d2b55;letter-spacing:2px;">0245871167</p>
                <p style="margin:0;font-size:13px;color:#6b7280;">Account Name: <strong>Young Investors Network</strong></p>
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

def make_plain():
    return """\
Dear Participant,

This is a reminder that your YIAP assessments are open and your certificate is within reach.

GSE SUBMISSION NOTICE:
As promised, your details will be submitted to the Ghana Stock Exchange (GSE).
We encourage you to complete the assessment before submission so your certificate is included.

Your certificate demonstrates verified financial analysis skills — an asset that sets you apart in your job search.

TAKE THE ASSESSMENT:
https://investright.onrender.com/yiap-practice
(Study materials and practice questions are also available at the same link.)

PAYMENT — to receive your certificate:
  Complete & Pass  — GHc70  (Certificate of Completion + Certificate of Participation)
  Participation Only — GHc30  (Certificate of Participation)

Send Mobile Money to: 0245871167
Account Name: Young Investors Network

Test Your Knowledge • Validate Your Skills • Earn Your Certification

Warm regards,
Kofi Kyei
Young Investors Analyst Program | InvestIQ
Building the Next Generation of Financial Analysts
"""

def run():
    with app.app_context():
        total  = len(EMAILS)
        sent   = 0
        failed = []

        print(f"Blasting to {total} recipients + CC to Joshua Mensah...")

        for i, email in enumerate(EMAILS, 1):
            try:
                msg = Message(
                    subject=SUBJECT,
                    sender=("InvestIQ — YIAP", app.config.get('MAIL_USERNAME')),
                    recipients=[email],
                    # CC Joshua only on the first email so he receives it exactly once
                    cc=[JOSHUA] if i == 1 else [],
                )
                msg.html = make_html()
                msg.body = make_plain()
                mail.send(msg)
                sent += 1
                cc_note = " [+CC Joshua]" if i == 1 else ""
                print(f"  [{i}/{total}] Sent → {email}{cc_note}")
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
