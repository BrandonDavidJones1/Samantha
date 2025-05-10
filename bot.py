import discord
from discord.ext import commands
import os
import json
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
TOKEN = os.getenv('DISCORD_TOKEN')
# BOT_PREFIX = "!" # We'll use a dynamic prefix now
FAQ_FILE = "faq_data.json"

# --- Helper function to determine prefix ---
def get_prefix(bot, message):
    if isinstance(message.channel, discord.DMChannel):
        return ""  # No prefix in DMs
    return commands.when_mentioned_or("!")(bot, message) # Use "!" or mention in guilds

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
intents.guild_messages = True # Keep this if you want guild commands with "!" or mention

bot = commands.Bot(command_prefix=get_prefix, intents=intents, help_command=None) # help_command=None if you want to make your own

# --- Global Data ---
faq_data = {}

# --- Helper Functions (load_faq_data, save_faq_data - remain the same) ---
def load_faq_data():
    global faq_data
    try:
        with open(FAQ_FILE, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        print(f"Successfully loaded data from {FAQ_FILE}")
    except FileNotFoundError:
        print(f"Error: {FAQ_FILE} not found. Creating a default structure.")
        faq_data = {
            "general_faqs": [],
            "call_codes": {},
            "fallback_message": "Sorry, I couldn't find an answer. Please ask Adam or your manager."
        }
        save_faq_data()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {FAQ_FILE}. Check its format.")
        faq_data = {
            "general_faqs": [],
            "call_codes": {},
            "fallback_message": "Sorry, I couldn't find an answer due to a data error. Please ask Adam or your manager."
        }

def save_faq_data():
    try:
        with open(FAQ_FILE, 'w', encoding='utf-8') as f:
            json.dump(faq_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved data to {FAQ_FILE}")
    except Exception as e:
        print(f"Error saving data to {FAQ_FILE}: {e}")

# --- Event Handlers ---
@bot.event
async def on_ready():
    print(f'{bot.user.name} (ID: {bot.user.id}) has connected to Discord!')
    print(f'Listening for DMs (no prefix for commands) and guild commands with "!" or mention.')
    load_faq_data()
    await bot.change_presence(activity=discord.Game(name="DM me for help!"))

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Let the commands framework attempt to process the message first.
    # This will use our `get_prefix` function.
    # If a command is found and executed, `process_commands` handles it.
    await bot.process_commands(message)

    # To prevent NLP from running if a command was already processed by `process_commands`:
    # We check if the message *could* have been a command.
    ctx = await bot.get_context(message)
    if ctx.valid and ctx.command: # ctx.valid checks if a command was found and could be invoked
        return # Command was handled, so don't proceed to NLP

    # If it's a DM and NO command was processed, then do NLP
    if isinstance(message.channel, discord.DMChannel):
        print(f"Received DM (NLP) from {message.author}: {message.content}") # For logging/debugging
        user_query = message.content.lower().strip()

        # 1. Check for "how to code" questions
        if "how to code" in user_query or "code for" in user_query:
            potential_code_key = user_query.replace("how to code", "").replace("code for", "").strip()
            
            found_code = False
            for code, description in faq_data.get("call_codes", {}).items():
                # More robust matching for call codes
                normalized_code = code.lower()
                normalized_key = potential_code_key # Already lower from user_query
                if normalized_key in normalized_code or normalized_code in normalized_key: # Check both directions
                    await message.channel.send(f"**How to code '{code.title()}':**\n{description}")
                    found_code = True
                    return 
            if not found_code and potential_code_key:
                 await message.channel.send(f"I don't have specific coding instructions for '{potential_code_key}'. "
                                           f"You can ask for `listcodes` or try a more general term. " # Update example here
                                           f"If this is a new code, ask Adam or your manager to add it to my knowledge base!")
                 return

        # 2. Check general FAQs
        for faq_item in faq_data.get("general_faqs", []):
            for keyword in faq_item.get("keywords", []):
                if keyword.lower() in user_query:
                    await message.channel.send(faq_item.get("answer", "Sorry, an answer is configured but missing."))
                    return

        # 3. If no match, send fallback (but only if it wasn't a command like "listcodes")
        #    The check `if ctx.valid and ctx.command:` above should prevent this for valid commands.
        fallback = faq_data.get("fallback_message", "I'm sorry, I don't have an answer for that right now.")
        await message.channel.send(fallback.replace("[Your Boss's Discord Name/Nickname]", "your manager"))
        return
    # No need for the `elif message.content.startswith(BOT_PREFIX):` for guild commands
    # as `await bot.process_commands(message)` at the top handles it with `get_prefix`.

# --- Bot Commands (remain largely the same, but now callable without "!" in DMs) ---
@bot.command(name='hello', help='Responds with a friendly greeting.')
async def hello(ctx):
    await ctx.send(f'Hello {ctx.author.mention}! How can I help you today?')

@bot.command(name='listcodes', aliases=['showcodes', 'codes'], help='Lists all known call codes and their descriptions.')
async def list_codes(ctx):
    codes = faq_data.get("call_codes", {})
    if not codes:
        await ctx.send("No call codes are currently defined.")
        return

    response_message = "**Available Call Codes:**\n"
    for code, description in codes.items():
        response_message += f"\n**{code.title()}**:\n_{description}_\n"
    
    if len(response_message) > 1900:
        await ctx.send("There are too many codes to list in one message. Please ask about specific codes or check with Adam/your manager.")
    else:
        await ctx.send(response_message)

@bot.command(name='reloadfaq', help='Reloads the FAQ data from the JSON file (Admin/Boss only).')
@commands.has_permissions(administrator=True)
async def reload_faq(ctx):
    load_faq_data()
    await ctx.send("FAQ data has been reloaded successfully!")

@reload_faq.error
async def reloadfaq_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("Sorry, you don't have permission to use that command.")
    else:
        print(f"Error in reloadfaq: {error}") # Log other errors
        await ctx.send(f"An error occurred with reloadfaq command.")


# --- Run the Bot ---
if __name__ == "__main__":
    if TOKEN:
        bot.run(TOKEN)
    else:
        print("Error: DISCORD_TOKEN not found in .env file or environment variables. Bot cannot start.")