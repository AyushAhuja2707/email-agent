import html
import os
import uuid

from flask import Flask, jsonify, request

from agent import agent
from env import EmailEnv


app = Flask(__name__)

VALID_ACTIONS = ["IMPORTANT", "SPAM", "WORK"]

CURRENT_RUN = {
    "env": None,
    "episode_id": None,
    "task_id": None,
    "step_count": 0,
    "total_reward": 0.0,
    "done": False,
    "current_email": None,
}

TASKS = [
    {
        "id": "task_easy_001",
        "difficulty": "easy",
        "description": "Classify 3 emails from a small inbox.",
        "max_steps": 3,
        "grader": "smoothed_accuracy",
    },
    {
        "id": "task_medium_001",
        "difficulty": "medium",
        "description": "Classify 5 emails from a medium inbox.",
        "max_steps": 5,
        "grader": "smoothed_accuracy",
    },
    {
        "id": "task_hard_001",
        "difficulty": "hard",
        "description": "Classify all 8 emails from the full inbox.",
        "max_steps": 8,
        "grader": "smoothed_accuracy",
    },
]


def build_observation(email_text):
    env = CURRENT_RUN["env"]
    remaining = 0
    if env is not None:
        remaining = max(len(env.data) - env.index, 0)

    return {
        "email": email_text,
        "valid_actions": VALID_ACTIONS,
        "task_id": CURRENT_RUN["task_id"],
        "step_count": CURRENT_RUN["step_count"],
        "remaining_emails": remaining,
        "done": CURRENT_RUN["done"],
        "task_description": "Classify the current email as IMPORTANT, SPAM, or WORK.",
    }


def start_new_episode(task_id):
    env = EmailEnv(task_id=task_id)
    first_email = env.reset()

    CURRENT_RUN["env"] = env
    CURRENT_RUN["episode_id"] = str(uuid.uuid4())
    CURRENT_RUN["task_id"] = env.task_id
    CURRENT_RUN["step_count"] = 0
    CURRENT_RUN["total_reward"] = 0.0
    CURRENT_RUN["done"] = False
    CURRENT_RUN["current_email"] = first_email

    return first_email


def extract_action(payload):
    if not isinstance(payload, dict):
        return None

    for key in ("action", "label", "prediction", "category"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip().upper()

    action_type = payload.get("action_type")
    if isinstance(action_type, str) and action_type.strip().upper() in VALID_ACTIONS:
        return action_type.strip().upper()

    return None


def run_episode_preview():
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


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/reset")
def reset():
    requested_task_id = request.args.get("task_id", "task_easy_001")
    first_email = start_new_episode(requested_task_id)
    return jsonify(
        {
            "observation": build_observation(first_email),
            "reward": 0.0,
            "done": False,
            "info": {
                "episode_id": CURRENT_RUN["episode_id"],
                "message": "Environment reset successfully.",
            },
        }
    )


@app.post("/step")
def step():
    if CURRENT_RUN["env"] is None or CURRENT_RUN["done"]:
        first_email = start_new_episode("task_easy_001")
        return jsonify(
            {
                "observation": build_observation(first_email),
                "reward": 0.0,
                "done": False,
                "info": {
                    "episode_id": CURRENT_RUN["episode_id"],
                    "message": "A new episode was started because no active episode was available.",
                },
            }
        )

    payload = request.get_json(silent=True) or {}
    action = extract_action(payload)

    if action not in VALID_ACTIONS:
        return (
            jsonify(
                {
                    "error": "Invalid action. Use one of IMPORTANT, SPAM, or WORK.",
                    "valid_actions": VALID_ACTIONS,
                }
            ),
            400,
        )

    next_email, reward, done, _ = CURRENT_RUN["env"].step(action)
    CURRENT_RUN["step_count"] += 1
    CURRENT_RUN["total_reward"] += reward
    CURRENT_RUN["done"] = done
    CURRENT_RUN["current_email"] = next_email

    return jsonify(
        {
            "observation": build_observation(next_email),
            "reward": float(reward),
            "done": done,
            "info": {
                "episode_id": CURRENT_RUN["episode_id"],
                "message": "Step executed successfully.",
            },
        }
    )


@app.get("/state")
def state():
    if CURRENT_RUN["env"] is None:
        return jsonify(
            {
                "episode_id": None,
                "step_count": 0,
                "total_reward": 0.0,
                "done": False,
                "current_email": None,
            }
        )

    return jsonify(
        {
            "episode_id": CURRENT_RUN["episode_id"],
            "step_count": CURRENT_RUN["step_count"],
            "total_reward": CURRENT_RUN["total_reward"],
            "done": CURRENT_RUN["done"],
            "current_email": CURRENT_RUN["current_email"],
        }
    )


@app.get("/tasks")
def tasks():
    return jsonify({"tasks": TASKS})


@app.route("/", methods=["GET", "POST"])
def home():
    email_text = ""
    prediction = None

    if request.method == "POST":
        email_text = request.form.get("email", "")
        prediction = agent(email_text)

    episode_logs = run_episode_preview()

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
      <p>OpenEnv endpoints: <code>POST /reset</code>, <code>POST /step</code>, <code>GET /state</code>, <code>GET /health</code>.</p>
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
