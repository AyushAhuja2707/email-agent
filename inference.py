import os
from openai import OpenAI
from env import EmailEnv


RAW_API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

if RAW_API_BASE_URL:
    API_BASE_URL = RAW_API_BASE_URL.rstrip("/")
    if not API_BASE_URL.endswith("/v1"):
        API_BASE_URL = f"{API_BASE_URL}/v1"
else:
    API_BASE_URL = None

MODEL_NAME = (
    os.getenv("MODEL_NAME")
    or os.getenv("MODEL")
    or os.getenv("OPENAI_MODEL")
    or "gpt-4o-mini"
)

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


def log_start():
    print(f"[START] task=email-classification env=custom model={MODEL_NAME}")

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")


def classify_email(email):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify emails into exactly one label: IMPORTANT, SPAM, or WORK. "
                    "Reply with only one of those labels."
                ),
            },
            {
                "role": "user",
                "content": f"Email: {email}",
            },
        ],
    )

    label = (response.choices[0].message.content or "").strip().upper()
    if label not in {"IMPORTANT", "SPAM", "WORK"}:
        if "SPAM" in label:
            return "SPAM"
        if "WORK" in label:
            return "WORK"
        return "IMPORTANT"
    return label


def main():
    if not API_BASE_URL or not API_KEY:
        raise RuntimeError(
            "Missing API_BASE_URL or API_KEY. This script must use the injected LLM proxy credentials."
        )

    env = EmailEnv()

    rewards = []
    step = 0

    log_start()

    email = env.reset()
    done = False

    while not done:
        step += 1

        action = classify_email(email)

        email, reward, done, _ = env.step(action)

        rewards.append(reward)

        log_step(step, action, reward, done)

    success = sum(rewards) > 0

    log_end(success, step, rewards)


if __name__ == "__main__":
    main()
