Agent Support Discord Bot - README
This document explains how to use the "Agent Support" Discord bot and how to keep it running. The bot is designed to answer common agent questions, provide guidance on call coding, and assist with word pronunciations.
How Agents Use the Bot
Find the Bot: Agents will find a user named "Agent Support" (or its configured name, e.g., the bot's actual Discord username) in the Discord server member list.
Send a Direct Message (DM): Click on the bot's name and send it a private message. All interactions with the bot occur in DMs.
Ask Questions & Use Commands (in DMs):
General Questions: Type questions naturally. The bot uses semantic search to understand the intent.
Example: What time do we start tomorrow?
Example: How should I handle an angry voter?
Example: Do I need to say every word in the script?
List Call Codes: To see all defined call disposition codes and their descriptions.
Command: list codes
Define Specific Call Code: To get the definition for a particular call code.
Command: define [code name] or what is [code name]
Example: define WRONG NUMBER
Example: what is DNC
Pronunciation Help: To get pronunciation resources for a word or short phrase.
Command: pronounce [word or phrase] or !pronounce [word or phrase]
Example: pronounce USGLC
Example: !pronounce Duquesne
Greetings & Common Phrases: The bot responds to simple greetings and acknowledgments.
Example: hello
Example: thank you
Managing the Bot's Knowledge
The bot's knowledge base is primarily stored in a JSON file named faq_data.json, located in the bot's main project folder.
To Add or Change Questions, Answers, or Call Codes:
Open faq_data.json: Use a plain text editor (like Notepad on Windows, TextEdit on Mac in plain text mode, VS Code, Sublime Text, etc.).
Understanding the File Structure:
greetings_and_pleasantries: Defines responses to common greetings and social phrases. Each entry has:
keywords: A list of words or phrases that trigger this greeting.
response_type: How the bot should reply (e.g., standard_greeting, specific_reply).
greeting_reply_template or reply_text: The bot's response.
general_faqs: For common questions and answers. Each entry has:
keywords: A list of representative keywords and phrases. The bot uses these for semantic matching. Well-chosen, diverse keywords are crucial for accuracy.
answer: The bot's detailed response to the question.
call_codes: Defines specific call disposition codes. Each entry is a key-value pair:
"CODE_NAME_IN_UPPERCASE": The call code itself (e.g., "WRONG NUMBER").
"Detailed instructions for that code.": The description/procedure.
fallback_message: The bot's default reply if it cannot find a relevant answer or understand the query.
Make Changes Carefully:
Preserve the JSON syntax (curly braces {}, square brackets [], double quotes "" for keys and string values, commas , between elements).
When adding new FAQs or keywords, think about how an agent would actually ask the question. Include variations.
If unsure, copy an existing entry within the same section and modify it.
Save the File.
Update the Bot's Knowledge:
The bot needs to re-read the faq_data.json file and rebuild its semantic understanding after changes.
The simplest and most reliable way to update the bot is to restart the bot program (bot.py).
(Note: An admin "reloadfaq" command is not currently implemented in this version of the bot. A restart is required.)
Keeping the Bot Running (Hosting)
The bot program (bot.py) must be continuously running on a computer that is always on and connected to the internet for agents to use it.
Local Computer (for testing/small scale):
You can run bot.py from a terminal or command prompt on an office computer.
Requirements: Python installed, all necessary libraries from requirements.txt (if applicable) installed (e.g., discord.py, python-dotenv, thefuzz, sentence-transformers, torch, requests).
Drawback: If the computer shuts down, restarts, or loses internet, the bot goes offline.
Cloud Server (Recommended for Reliability):
For continuous, reliable operation, host the bot on a cloud service provider like:
AWS (Amazon Web Services) - EC2 instance
Google Cloud Platform (GCP) - Compute Engine
Azure (Microsoft) - Virtual Machines
DigitalOcean Droplets
Heroku (might be suitable if resource needs are low, though may have limitations for always-on free tiers)
Railway.app, Render.com (Platform-as-a-Service options that can simplify deployment)
Benefit: These services are designed for 24/7 uptime.
Consideration: May involve a learning curve and potentially hosting costs. Technical assistance might be beneficial for initial setup.
Process Management: On a server, you'll want to use a process manager like systemd (Linux) or pm2 (Node.js, but can run Python scripts) to ensure the bot restarts automatically if it crashes and runs in the background.
Recommendation:
Start by running it on a local/office computer (Option 1) for initial testing and evaluation.
If the bot proves essential, migrating to a cloud service (Option 2) is highly recommended for stability and availability.
VERY IMPORTANT: Bot Token Security
The DISCORD_TOKEN (stored in your .env file and loaded by the bot) is the bot's unique password to connect to Discord. It grants full control over the bot's account.
NEVER share this token publicly.
Do not commit the .env file or the token itself to version control (e.g., Git, GitHub). Use a .gitignore file to exclude .env.
Do not email it unencrypted.
Do not post it in public or private chats.
Do not embed it directly in shared documents accessible by unauthorized individuals.
If you suspect the token has been accidentally exposed:
Go to the Discord Developer Portal.
Select your bot's application.
Navigate to the "Bot" page.
Click "Reset Token" and confirm. This will invalidate the old token.
A new token will be generated. Copy this new token immediately.
Update your bot's configuration (e.g., the .env file) with the new token.
Restart the bot.
Treat the token with the same level of security as you would any critical password. An exposed token compromises the bot and potentially its actions on your server.