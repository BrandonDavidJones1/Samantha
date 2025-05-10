# Agent Support Discord Bot - README

Hello Cory Grottola,

This document explains how to use the "Agent Support" Discord bot and how to keep it running. The bot is designed to answer common agent questions and provide guidance on call coding.

## How Agents Use the Bot

1.  **Find the Bot:** Agents will find a user named "Agent Support" (or the name we gave it) in the Discord server member list.
2.  **Send a Direct Message (DM):** They should click on the bot's name and send it a private message.
3.  **Ask Questions & Use Commands (in DMs):**
    -   **General Questions:** Type questions naturally.
        -   Example: `What time do we start tomorrow?`
        -   Example: `How long is our lunch break?`
    -   **Call Coding Questions:** Ask about specific call outcomes.
        -   Example: `How do I code a complete?`
        -   Example: `What's the code for a customer not interested?`
    -   **List All Codes:** Simply type `listcodes` to see all call coding procedures the bot knows. (No `!` prefix needed in DMs).
    -   **Say Hello:** Type `hello` and the bot will greet you.

## Managing the Bot's Knowledge (For Adam & You)

The bot learns from a file called `faq_data.json`. This is where all the questions and answers are stored. This file will be located in the bot's main project folder.

To Add or Change Answers:

1.  **Open the File:** You'll need a simple text editor (like Notepad on Windows, or TextEdit on Mac) to open the `faq_data.json` file.
2.  **Understanding the File:**
    -   `general_faqs`: This section is for common questions. Each question has:
        -   `keywords`: A list of words or phrases that will trigger the answer (e.g., "start time", "when do we begin").
        -   `answer`: The exact response the bot will give.
    -   `call_codes`: This section is for call coding. Each entry has:
        -   A "code name" (e.g., "sale completed").
        -   The detailed instructions for that code.
    -   `fallback_message`: What the bot says if it doesn't understand a question.
3.  **Make Changes:** Carefully edit the text. Make sure to keep the special characters like `{ } [ ] " ,` in the right places. It's best to copy an existing entry and modify it if you're unsure.
4.  **Save the File.**
5.  **Update the Bot:**
    -   The bot needs to re-read the `faq_data.json` file after changes.
    -   **Admin Command (in DM):** If you have "Administrator" permissions in Discord, you can DM the bot with the command: `reloadfaq` (no `!` prefix needed). This is the easiest way to update its knowledge without a full restart.
    -   Alternatively, the bot program can be restarted.

## Keeping the Bot Running (Hosting)

For the bot to work, the program (`bot.py`) needs to be running on a computer that's always on and connected to the internet. Here are the main ways to do this:

### Option 1: Run it on an Office Computer (Simpler, Good for Testing)

You can run the bot on an existing office computer, or a spare one, as long as it stays on during work hours (or 24/7 if needed).

Steps to set this up:

1.  **Python:** A free software called Python (version 3.8 or newer) needs to be installed on the computer. This can be downloaded from `python.org`.
2.  **Bot Files:** Get all the bot files (like `bot.py`, `faq_data.json`, etc.) onto that computer, into their own folder.
3.  **Install Bot "Add-ons":** Open a "Command Prompt" (Windows) or "Terminal" (Mac/Linux) window on that computer. Navigate to the bot's folder using commands like `cd path\to\folder`. Then, run the command: `pip install -r requirements.txt`. This installs necessary components for the bot.
4.  **Bot Token (Secret Password):**
    -   The bot needs a secret "token" (like a password) to log into Discord. This token comes from the Discord Developer Portal where the bot application was created.
    -   Create a file named `.env` (literally ".env", with a dot at the beginning) in the bot's main folder.
    -   Open this `.env` file with a text editor and add the following line, replacing `YOUR_ACTUAL_BOT_TOKEN_HERE` with the bot's unique token:
        ```
        DISCORD_TOKEN=YOUR_ACTUAL_BOT_TOKEN_HERE
        ```
    -   This file and the token inside it must be kept secret.
5.  **Start the Bot:** In the Command Prompt/Terminal, while in the bot's folder, run the command: `python bot.py`
    -   A window will stay open showing the bot is running and logging its activity. If you close this window, the bot stops. To keep it running, this window must remain open and the computer must stay on.

Pros:
-   Usually no extra cost if you have a spare PC.
-   Relatively simple to get started.

Cons:
-   The computer must stay on and connected to the internet. If it turns off, restarts, or loses internet, the bot will go offline.
-   Someone needs to remember to start it if the PC reboots and ensure the command window stays open.

### Option 2: Use an Online "Cloud" Service (More Reliable for 24/7)

This means renting a small, virtual computer on the internet that runs your bot 24/7. This is generally the best option if you want the bot to be always available.

How it generally works:

1.  **Sign Up:** A hosting provider is chosen (examples: Railway.app, DigitalOcean, Vultr). Some have free or very cheap plans (e.g., $5-10/month).
2.  **Setup:** The setup process on a cloud service generally involves:
    -   Uploading the bot files to the online server.
    -   Installing Python and the bot's "add-ons" (dependencies) there.
    -   Securely configuring the `DISCORD_TOKEN` as an environment variable through the service's dashboard.
    -   Setting up the bot to run continuously (e.g., using tools like `systemd`, `Docker`, or a `Procfile` depending on the platform) so it restarts automatically if there's an issue.

Pros:
-   Very reliable – the bot stays online 24/7.
-   Doesn't depend on an office computer being on.

Cons:
-   Usually has a small monthly cost.
-   Requires more technical setup initially.

Recommendation:
-   It might be easiest to start by running it on an office computer (Option 1) to test it out and see how useful it is.
-   If it becomes essential, it is highly recommended to move it to an online cloud service (Option 2) for better reliability. Technical assistance may be needed for this more advanced setup.

## VERY IMPORTANT: Bot Token Security

-   The `DISCORD_TOKEN` is the bot's password to access Discord.
-   NEVER share this token publicly. Don't email it unencrypted, don't post it in chats, don't put it in shared documents anyone can see.
-   If you think the token has been accidentally exposed, it needs to be invalidated and a new one generated immediately from the Discord Developer Portal. Ask for assistance with this if needed, as an exposed token compromises the bot.

---

If you have any questions about using the bot, changing its answers, or how it's running, please ask Brandon Jones.