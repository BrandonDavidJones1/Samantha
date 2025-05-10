Agent Support Discord Bot - README
This document explains how to use the "Agent Support" Discord bot and how to keep it running. The bot is designed to answer common agent questions and provide guidance on call coding.

How Agents Use the Bot

Find the Bot: Agents will find a user named "Agent Support" (or its configured name) in the Discord server member list.
Send a Direct Message (DM): Click on the bot's name and send it a private message.
Ask Questions & Use Keywords (in DMs): All interactions are via keywords in DMs.
General Questions: Type questions naturally.
Example: What time do we start tomorrow?
Example: How long is our lunch break?
Call Coding Questions: Ask about specific call outcomes.
Example: How do I code a complete?
Example: What's the code for a customer not interested?

Managing the Bot's Knowledge 

The bot learns from a file named faq_data.json, located in the bot's main project folder. This file stores all questions and answers.
To Add or Change Answers:
Open faq_data.json: Use a simple text editor (like Notepad or TextEdit).
Understanding the File Structure:
general_faqs: For common questions. Each has:
keywords: Phrases triggering the answer (e.g., "start time").
answer: The bot's response.
call_codes: For call coding. Each entry has:
A "code name" (e.g., "sale completed").
Detailed instructions for that code.
fallback_message: Bot's reply if it doesn't understand.
Make Changes: Carefully edit the text, preserving special characters ({ } [ ] " ,). Copying an existing entry to modify is recommended if unsure.
Save the File.
Update the Bot:
The bot needs to re-read faq_data.json after changes.
Admin Keyword (in DM): If you are an authorized admin (your Discord User ID is configured for the bot), DM the bot with the keyword: reloadfaq. This updates its knowledge without a full restart.
Alternatively, restart the bot program.

## Keeping the Bot Running (Hosting)

Keeping the Bot Running (Hosting)
The bot program (bot.py) must run on a computer that's always on and internet-connected. Alternatively you could use a cloud server (like AWS)

Recommendation:
-   It might be easiest to start by running it on an office computer (Option 1) to test it out and see how useful it is.
-   If it becomes essential, it is highly recommended to move it to an online cloud service (Option 2) for better reliability. Technical assistance may be needed for this more advanced setup.

## VERY IMPORTANT: Bot Token Security

-   The `DISCORD_TOKEN` is the bot's password to access Discord.
-   NEVER share this token publicly. Don't email it unencrypted, don't post it in chats, don't put it in shared documents anyone can see.
-   If you think the token has been accidentally exposed, it needs to be invalidated and a new one generated immediately from the Discord Developer Portal. Ask for assistance with this if needed, as an exposed token compromises the bot.

---
If you have any questions about using the bot, changing its answers, or how it's running, please ask Brandon Jones.