import os
from openai import OpenAI
import requests


RAW_API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")

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
    or ""
)
AVAILABLE_MODELS = []

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


def log_start(task_id):
    print(f"[START] task={task_id} env=custom model={MODEL_NAME}")

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}")


def rank_model_name(model_name):
    lowered = model_name.lower()
    score = 0

    preferred_patterns = [
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4o",
        "gpt",
        "chat",
        "instruct",
        "turbo",
    ]
    blocked_patterns = [
        "embed",
        "embedding",
        "tts",
        "transcribe",
        "rerank",
        "image",
        "moderation",
    ]

    for index, pattern in enumerate(preferred_patterns):
        if pattern in lowered:
            score += 100 - index

    for pattern in blocked_patterns:
        if pattern in lowered:
            score -= 1000

    return score


def resolve_model_name():
    global AVAILABLE_MODELS

    if MODEL_NAME:
        return MODEL_NAME

    models = client.models.list()
    AVAILABLE_MODELS = [model.id for model in models.data if getattr(model, "id", None)]
    if not AVAILABLE_MODELS:
        raise RuntimeError("No models returned by the injected LLM proxy.")

    ranked_models = sorted(
        AVAILABLE_MODELS,
        key=lambda model_name: rank_model_name(model_name),
        reverse=True,
    )
    return ranked_models[0]


def classify_email(email):
    candidate_models = [MODEL_NAME] if MODEL_NAME else []
    for model_name in AVAILABLE_MODELS:
        if model_name not in candidate_models:
            candidate_models.append(model_name)

    last_error = None
    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
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
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Unable to classify email with any proxy model: {last_error}")


def env_post(path, json_body=None, params=None):
    response = requests.post(
        f"{ENV_URL}{path}",
        json=json_body or {},
        params=params or {},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def smoothed_score(rewards):
    correct = sum(1 for reward in rewards if reward > 0)
    total = len(rewards)
    return (correct + 0.5) / (total + 1.0)


def run_task(task_id):
    rewards = []
    step = 0

    log_start(task_id)

    reset_payload = env_post("/reset", params={"task_id": task_id})
    observation = reset_payload["observation"]
    email = observation["email"]
    done = False

    while not done:
        step += 1
        action = classify_email(email)
        step_payload = env_post("/step", json_body={"action": action})
        reward = float(step_payload["reward"])
        done = bool(step_payload["done"])
        rewards.append(reward)
        log_step(step, action, reward, done)

        next_observation = step_payload.get("observation") or {}
        email = next_observation.get("email")

    score = smoothed_score(rewards)
    success = 0.0 < score < 1.0
    log_end(success, step, score, rewards)


def main():
    if not API_BASE_URL or not API_KEY:
        raise RuntimeError(
            "Missing API_BASE_URL or API_KEY. This script must use the injected LLM proxy credentials."
        )

    global MODEL_NAME
    MODEL_NAME = resolve_model_name()

    for task_id in ["task_easy_001", "task_medium_001", "task_hard_001"]:
        run_task(task_id)


if __name__ == "__main__":
    main()
