# --- bot.py (MINIMAL TEST CODE) ---

import discord
import os
from dotenv import load_dotenv
import asyncio

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    # If you see these messages in your logs, the bot's environment is stable.
    print("--- MINIMAL TEST: Bot has connected successfully!")
    print(f"--- MINIMAL TEST: Bot name: {bot.user.name}")
    print("--- MINIMAL TEST: The on_ready event is firing correctly.")
    print("--- MINIMAL TEST: Waiting for 10 seconds...")
    await asyncio.sleep(10)
    print("--- MINIMAL TEST: Wait complete. Bot is stable and idle.")
    print("--- MINIMAL TEST: This confirms the issue is with loading the large models (RAM usage).")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return
    # Respond to any message to show the bot is live
    await message.channel.send("Minimal test bot is alive and responding!")

# --- Run the Bot ---
if __name__ == "__main__":
    if TOKEN:
        print("--- MINIMAL TEST: Token found. Attempting to run bot...")
        bot.run(TOKEN)
    else:
        print("--- MINIMAL TEST: ERROR - DISCORD_TOKEN not found.")