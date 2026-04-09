class EmailEnv:

    def __init__(self, task_id="task_easy_001"):
        all_data = [
            ("Your OTP is ", "IMPORTANT"),
            ("Win money now!!!", "SPAM"),
            ("Meeting at 10 AM", "WORK"),
            ("Your bank account update", "IMPORTANT"),
            ("You won a lottery!!!", "SPAM"),
            ("Project deadline tomorrow", "WORK"),
            ("Reset your password now", "IMPORTANT"),
            ("Click here for free gift", "SPAM"),
        ]

        task_slices = {
            "task_easy_001": 3,
            "task_medium_001": 5,
            "task_hard_001": 8,
        }

        self.task_id = task_id if task_id in task_slices else "task_easy_001"
        self.data = all_data[: task_slices[self.task_id]]
        self.index = 0

    def reset(self):
        self.index = 0
        email, _ = self.data[self.index]
        return email

    def step(self, action):
        email, correct = self.data[self.index]

        reward = 1 if action == correct else -1

        self.index += 1
        done = self.index >= len(self.data)

        next_email = None
        if not done:
            next_email, _ = self.data[self.index]

        return next_email, reward, done, {}
