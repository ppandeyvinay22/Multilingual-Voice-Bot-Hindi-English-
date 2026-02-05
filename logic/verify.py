import json
import re

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def _normalize_digit_words(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+|\\d+", text.lower())
    out = []
    for tok in tokens:
        if tok.isdigit():
            out.append(tok)
        elif tok in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[tok])
    return "".join(out)


def load_users(path="logic/users.json"):
    with open(path, "r") as f:
        return json.load(f)


def extract_mobile(text: str) -> str | None:
    digits = re.sub(r"\D", "", text)
    if len(digits) < 10:
        digits += _normalize_digit_words(text)
    if len(digits) >= 10:
        return digits[-10:]
    return None


def extract_last4(text: str) -> str | None:
    digits = re.sub(r"\D", "", text)
    if len(digits) < 4:
        digits += _normalize_digit_words(text)
    if len(digits) >= 4:
        return digits[-4:]
    return None


def extract_dob(text: str) -> str | None:
    # Accepts YYYY-MM-DD or DD-MM-YYYY with - or / separators.
    cleaned = text.strip()
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", cleaned)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", cleaned)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        # Assume DDMMYYYY
        day = digits[0:2]
        month = digits[2:4]
        year = digits[4:8]
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def extract_otp(text: str) -> str | None:
    digits = re.sub(r"\D", "", text)
    if len(digits) < 6:
        digits += _normalize_digit_words(text)
    if len(digits) >= 6:
        return digits[-6:]
    return None


def verify_user(users, mobile: str, last4: str | None = None, dob: str | None = None):
    for user in users:
        if user.get("mobile") != mobile:
            continue
        if last4 and user.get("last4") == last4:
            return user
        if dob and user.get("dob") == dob:
            return user
    return None
