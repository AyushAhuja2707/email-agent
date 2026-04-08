import html
import os

from flask import Flask, request

from agent import agent
from env import EmailEnv


app = Flask(__name__)


def run_episode():
    env = EmailEnv()
    rewards = []
    logs = []

    email = env.reset()
    done = False
    step = 0

    logs.append("[START] task=email-classification env=custom model=rule-based-agent")

    while not done:
        step += 1
        action = agent(email)
        email, reward, done, _ = env.step(action)
        rewards.append(reward)
        logs.append(
            f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null"
        )

    success = sum(rewards) > 0
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    logs.append(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")

    return "\n".join(logs)


@app.route("/", methods=["GET", "POST"])
def home():
    email_text = ""
    prediction = None

    if request.method == "POST":
        email_text = request.form.get("email", "")
        prediction = agent(email_text)

    episode_logs = run_episode()

    prediction_html = ""
    if prediction is not None:
        prediction_html = (
            "<div class='result'>"
            f"<strong>Prediction:</strong> {html.escape(prediction)}"
            "</div>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Email Agent</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #22c55e;
      --border: #334155;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: linear-gradient(180deg, #020617, #0f172a);
      color: var(--text);
    }}
    .wrap {{
      max-width: 900px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .card {{
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    p {{
      color: var(--muted);
      line-height: 1.5;
    }}
    textarea {{
      width: 100%;
      min-height: 140px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      padding: 14px;
      font-size: 15px;
      resize: vertical;
    }}
    button {{
      margin-top: 14px;
      border: 0;
      border-radius: 10px;
      background: var(--accent);
      color: #052e16;
      font-weight: 700;
      padding: 12px 18px;
      cursor: pointer;
    }}
    .result {{
      margin-top: 16px;
      padding: 14px;
      border-radius: 12px;
      background: #052e16;
      border: 1px solid #166534;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #020617;
      color: #cbd5e1;
      padding: 16px;
      border-radius: 12px;
      border: 1px solid var(--border);
      overflow-x: auto;
    }}
    code {{
      font-family: Consolas, monospace;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Email Agent</h1>
      <p>Classify an email into <code>IMPORTANT</code>, <code>SPAM</code>, or <code>WORK</code> using the current rule-based agent.</p>
      <form method="post">
        <textarea name="email" placeholder="Paste an email here...">{html.escape(email_text)}</textarea>
        <br>
        <button type="submit">Classify Email</button>
      </form>
      {prediction_html}
    </div>

    <div class="card">
      <h2>Sample Environment Run</h2>
      <p>Below is one full pass through the toy environment used in this project.</p>
      <pre>{html.escape(episode_logs)}</pre>
    </div>
  </div>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port)
