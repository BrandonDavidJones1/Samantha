import discord
from discord.ext import commands
import os
import json
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
TOKEN = os.getenv('DISCORD_TOKEN')
BOT_PREFIX = "!" # Or any prefix you prefer for commands, though DMs won't strictly need it
FAQ_FILE = "faq_data.json"

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True # REQUIRED for reading message content
intents.dm_messages = True     # REQUIRED for DMs
intents.guild_messages = True  # If you want it to respond in channels too (optional)

bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)

# --- Global Data ---
faq_data = {}

# --- Helper Functions ---
def load_faq_data():
    global faq_data
    try:
        with open(FAQ_FILE, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        print(f"Successfully loaded data from {FAQ_FILE}")
    except FileNotFoundError:
        print(f"Error: {FAQ_FILE} not found. Creating a default structure.")
        # Create a default structure if file doesn't exist (or handle as error)
        faq_data = {
            "general_faqs": [],
            "call_codes": {},
            "fallback_message": "Sorry, I couldn't find an answer. Please ask Adam or your manager."
        }
        save_faq_data() # Save the default structure
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {FAQ_FILE}. Check its format.")
        # Potentially load a backup or exit
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
    print(f'Listening for DMs and commands with prefix: {BOT_PREFIX}')
    load_faq_data()
    # You can set a custom status
    await bot.change_presence(activity=discord.Game(name="DM me for help!"))


@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Primary focus: DMs
    if isinstance(message.channel, discord.DMChannel):
        print(f"Received DM from {message.author}: {message.content}") # For logging/debugging
        user_query = message.content.lower().strip()

        # Check for command prefix first (even in DMs)
        if user_query.startswith(BOT_PREFIX):
            # Let the command processor handle it, but remove prefix for consistency
            # This is useful if you decide to add commands for DMs specifically
            # message.content = message.content[len(BOT_PREFIX):] # Strip prefix for command processing
            await bot.process_commands(message)
            return # Command handled (or not found by process_commands)

        # 1. Check for "how to code" questions
        if "how to code" in user_query or "code for" in user_query:
            # Try to extract the call type
            # This is a simple extraction, could be improved with more NLP/regex
            potential_code_key = user_query.replace("how to code", "").replace("code for", "").strip()
            
            found_code = False
            # More robust matching for call codes
            for code, description in faq_data.get("call_codes", {}).items():
                if potential_code_key in code.lower() or code.lower() in potential_code_key:
                    await message.channel.send(f"**How to code '{code.title()}':**\n{description}")
                    found_code = True
                    return
            if not found_code and potential_code_key: # if we tried to extract something but failed
                 await message.channel.send(f"I don't have specific coding instructions for '{potential_code_key}'. "
                                           f"You can ask for `!listcodes` or try a more general term. "
                                           f"If this is a new code, ask Adam or your manager to add it to my knowledge base!")
                 return
            # If no specific code query was identified, fall through to general FAQs

        # 2. Check general FAQs
        for faq_item in faq_data.get("general_faqs", []):
            for keyword in faq_item.get("keywords", []):
                if keyword.lower() in user_query:
                    await message.channel.send(faq_item.get("answer", "Sorry, an answer is configured but missing."))
                    return # Found an answer

        # 3. If no match, send fallback
        fallback = faq_data.get("fallback_message", "I'm sorry, I don't have an answer for that right now.")
        await message.channel.send(fallback.replace("[Your Boss's Discord Name/Nickname]", "your manager")) # Personalize fallback
        return

    # If you want the bot to respond to mentions or commands in server channels:
    # elif bot.user.mentioned_in(message) or message.content.startswith(BOT_PREFIX):
    #     await bot.process_commands(message)
    
    # For now, explicitly process commands if they start with prefix, even outside DMs
    if message.content.startswith(BOT_PREFIX):
         await bot.process_commands(message)


# --- Bot Commands ---
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
    
    # Discord has a message character limit (2000)
    if len(response_message) > 1900: # Leave some room
        await ctx.send("There are too many codes to list in one message. Please ask about specific codes or check with Adam/your manager.")
        # Alternative: paginate or send in chunks
    else:
        await ctx.send(response_message)

@bot.command(name='reloadfaq', help='Reloads the FAQ data from the JSON file (Admin/Boss only).')
@commands.has_permissions(administrator=True) # Example: only admins can run this
async def reload_faq(ctx):
    load_faq_data()
    await ctx.send("FAQ data has been reloaded successfully!")

@reload_faq.error
async def reloadfaq_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("Sorry, you don't have permission to use that command.")
    else:
        await ctx.send(f"An error occurred: {error}")


# --- Run the Bot ---
if __name__ == "__main__":
    if TOKEN:
        bot.run(TOKEN)
    else:
        print("Error: DISCORD_TOKEN not found in .env file or environment variables. Bot cannot start.")