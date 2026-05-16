# Telegram API and `.env` Setup

This explains how to get Telegram API credentials and add them to the `.env` file.
---

## 1. Get Telegram API credentials

Open this website in your browser:

```text
https://my.telegram.org
```

Log in with your phone number.

Example phone format:

```text
+316xxxxxxxxx
```

Telegram will send a verification code to your Telegram app.

Enter this code on the website.

After login, go to:

```text
API development tools
```

Create a new application here.

Example form:

```text
App title: Any name you want
Short name: Any name you want
URL: Leave empty
Platform: Desktop
Description: Anything you want
```

After you save the form, Telegram will give you these two values:

```text
api_id
api_hash
```

---

## 2. Create a `.env` file in the project root

Go to the project root directory and create a `.env` file if not.

---

## 3. Add the credentials to the `.env` file

Add the values you received from Telegram to the `.env` file with these variable names:

```env
TG_API_ID=WRITE_YOUR_OWN_API_ID_HERE
TG_API_HASH=WRITE_YOUR_OWN_API_HASH_HERE
TG_SESSION_NAME=agentic_telegram_session
TELEGRAM_KEYWORDS=ask them
```

---

## 4. Check the `.gitignore` file

In the project root directory, check the `.gitignore` file.

It must include these lines:

```gitignore
.env
*.session
*.session-journal
```

If these lines are missing, add them.

The `.env` and `.session` files may contain sensitive information, so they must not be pushed to GitHub.