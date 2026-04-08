import os
from openai import OpenAI
from env import EmailEnv
from agent import agent

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"


def log_start():
    print(f"[START] task=email-classification env=custom model={MODEL_NAME}")

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")


def main():
    env = EmailEnv()

    rewards = []
    step = 0

    log_start()

    email = env.reset()
    done = False

    while not done:
        step += 1

        action = agent(email)

        email, reward, done, _ = env.step(action)

        rewards.append(reward)

        log_step(step, action, reward, done)

    success = sum(rewards) > 0

    log_end(success, step, rewards)


if __name__ == "__main__":
    main()