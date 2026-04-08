# Email Agent

This project is a minimal email-classification environment and baseline agent. The agent receives a short email-like text observation and predicts one of three labels:

- `IMPORTANT`
- `SPAM`
- `WORK`

The environment then returns a reward based on whether the prediction matches the expected class.

## Project Structure

- `env.py`: custom environment with a fixed dataset and `reset` / `step` methods
- `agent.py`: simple keyword-based baseline policy
- `inference.py`: runner that executes one full episode and prints structured logs
- `requirements.txt`: Python dependencies
- `Dockerfile`: container setup for running the project

## Environment Description

The environment is implemented in [env.py](/E:/Scaler/hackathon/email-agent/env.py). It contains a small in-memory dataset of `(email_text, label)` pairs:

- `"Your OTP is "` -> `IMPORTANT`
- `"Win money now!!!"` -> `SPAM`
- `"Meeting at 10 AM"` -> `WORK`
- `"Your bank account update"` -> `IMPORTANT`
- `"You won a lottery!!!"` -> `SPAM`
- `"Project deadline tomorrow"` -> `WORK`
- `"Reset your password now"` -> `IMPORTANT`
- `"Click here for free gift"` -> `SPAM`

Each episode iterates through this dataset from start to finish in order.

### Reset

`reset()`:

- resets the internal pointer to the first example
- returns the first email text as the initial observation

### Step

`step(action)`:

- compares the chosen action with the ground-truth label for the current email
- returns `(next_observation, reward, done, info)`

Return values:

- `next_observation`: the next email text, or `None` when the episode ends
- `reward`: `1` for a correct classification, `-1` for an incorrect classification
- `done`: `True` when all emails have been processed
- `info`: empty dictionary `{}` in the current implementation

## Observation Space

The observation space is a single email string.

Examples:

- `"Meeting at 10 AM"`
- `"Click here for free gift"`
- `"Reset your password now"`

In practical terms, you can think of the observation space as:

- Type: `str`
- Meaning: raw email text to classify

## Action Space

The action space is a discrete set of three class labels:

- `IMPORTANT`
- `SPAM`
- `WORK`

Any agent interacting with the environment should return exactly one of these strings on each step.

## Baseline Agent

The baseline policy is implemented in [agent.py](/E:/Scaler/hackathon/email-agent/agent.py). It uses keyword matching:

- security-related words like `otp`, `bank`, or `password` -> `IMPORTANT`
- promotional words like `win`, `lottery`, `click`, or `free` -> `SPAM`
- workplace words like `meeting`, `deadline`, or `project` -> `WORK`

If no rule matches, the current fallback is `IMPORTANT`.

## Running the Project

### Local Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set the Hugging Face token environment variable if needed by your runtime setup:

```bash
export HF_TOKEN=your_token_here
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

4. Run the inference script:

```bash
python inference.py
```

### Docker Setup

Build the image:

```bash
docker build -t email-agent .
```

Run the container:

```bash
docker run --rm -e HF_TOKEN=your_token_here email-agent
```

## Expected Output

The runner in [inference.py](/E:/Scaler/hackathon/email-agent/inference.py) prints logs in three phases:

- `[START]`: metadata about the task and model
- `[STEP]`: predicted action, reward, and done flag for each step
- `[END]`: final success flag, total steps, and reward history

Example log format:

```text
[START] task=email-classification env=custom model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=IMPORTANT reward=1.00 done=false error=null
[END] success=true steps=8 rewards=1.00,1.00,...
```

## Notes

- The current agent is rule-based and does not call the language model.
- `inference.py` initializes an OpenAI-compatible client pointing to Hugging Face Router, but the baseline flow currently classifies emails through the local `agent()` function.
- Because the environment data is fixed and ordered, this project is best understood as a simple evaluation/demo setup rather than a full RL benchmark.
