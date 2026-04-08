def agent(email):
    email = email.lower()

    if "otp" in email or "bank" in email or "password" in email:
        return "IMPORTANT"

    elif "win" in email or "lottery" in email or "click" in email or "free" in email:
        return "SPAM"

    elif "meeting" in email or "deadline" in email or "project" in email:
        return "WORK"

    else:
        return "IMPORTANT"