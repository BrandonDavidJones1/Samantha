import discord
import os
import json
from dotenv import load_dotenv
import string
from thefuzz import fuzz # For fuzzy matching
# from thefuzz import process # Could be used for more advanced matching later

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
FAQ_FILE = "faq_data.json"
FUZZY_MATCH_THRESHOLD_GREETINGS = 75
FUZZY_MATCH_THRESHOLD_KEYWORDS = 80
FUZZY_MATCH_THRESHOLD_CALLCODES_STRICT = 80

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = discord.Client(intents=intents)

# --- Global Data ---
faq_data = {}

# --- Helper Functions ---
def load_faq_data():
    global faq_data
    try:
        with open(FAQ_FILE, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        faq_data.setdefault("greetings_and_pleasantries", [])
        faq_data.setdefault("general_faqs", [])
        faq_data.setdefault("call_codes", {})
        faq_data.setdefault("fallback_message", "Sorry, I couldn't find an answer. Please ask your manager for assistance.")
        print(f"Successfully loaded data from {FAQ_FILE}")
    except FileNotFoundError:
        print(f"Error: {FAQ_FILE} not found. Creating a default structure.")
        faq_data = {
            "greetings_and_pleasantries": [],
            "general_faqs": [],
            "call_codes": {},
            "fallback_message": "Sorry, I couldn't find an answer. Please ask your manager for assistance."
        }
        save_faq_data()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {FAQ_FILE}. Check its format.")
        faq_data = {
            "greetings_and_pleasantries": [],
            "general_faqs": [],
            "call_codes": {},
            "fallback_message": "Sorry, I couldn't find an answer due to a data error. Please ask your manager for assistance."
        }

def save_faq_data():
    try:
        with open(FAQ_FILE, 'w', encoding='utf-8') as f:
            json.dump(faq_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved data to {FAQ_FILE}")
    except Exception as e:
        print(f"Error saving data to {FAQ_FILE}: {e}")

def is_text_empty_or_punctuation_only(text):
    if not text:
        return True
    return all(char in string.punctuation or char.isspace() for char in text)

def get_best_fuzzy_match(query, choices_list, threshold):
    best_match = None
    highest_score = 0
    for choice in choices_list:
        score = fuzz.token_sort_ratio(query, choice)
        if score > highest_score:
            highest_score = score
            best_match = choice

    if highest_score >= threshold:
        return best_match, highest_score
    return None, 0

# --- Event Handlers ---
@bot.event
async def on_ready():
    print(f'{bot.user.name} (ID: {bot.user.id}) has connected to Discord!')
    print(f'Listening for DMs. All interactions are handled as direct messages.')
    load_faq_data()
    await bot.change_presence(activity=discord.Game(name="DM me for help!"))

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if not isinstance(message.channel, discord.DMChannel):
        return

    user_query_lower = message.content.lower().strip()
    original_message_content = message.content.strip()

    if user_query_lower in ["listcodes", "showcodes", "codes"]:
        codes = faq_data.get("call_codes", {})
        if not codes:
            await message.channel.send("No call codes are currently defined.")
            return
        response_message = "**Available Call Codes:**\n"
        for code, description in codes.items():
            response_message += f"\n**{code.title()}**:\n_{description}_\n"
        if len(response_message) > 1900:
            response_message = response_message[:1900] + "\n\n... (message too long, truncated)"
        await message.channel.send(response_message)
        return

    greetings_config = faq_data.get("greetings_and_pleasantries", [])
    greeting_handled_or_skipped = False

    for item in greetings_config:
        if greeting_handled_or_skipped:
            break

        user_typed_greeting_text_for_item = ""
        best_score_for_item_greeting_match = 0

        for kw_json in item.get("keywords", []):
            for l_offset in range(-2, 3):
                user_prefix_len = len(kw_json) + l_offset
                if user_prefix_len <= 0 or user_prefix_len > len(user_query_lower):
                    continue

                user_prefix_to_check = user_query_lower[:user_prefix_len]
                score = fuzz.token_set_ratio(kw_json, user_prefix_to_check)

                if score > best_score_for_item_greeting_match and score >= FUZZY_MATCH_THRESHOLD_GREETINGS:
                    best_score_for_item_greeting_match = score
                    user_typed_greeting_text_for_item = user_prefix_to_check

        if user_typed_greeting_text_for_item:
            remaining_text_after_greeting = user_query_lower[len(user_typed_greeting_text_for_item):].strip()
            is_standalone = is_text_empty_or_punctuation_only(remaining_text_after_greeting)
            response_type = item.get("response_type")

            if response_type == "specific_reply":
                if is_standalone:
                    reply_text = item.get("reply_text", "Okay.")
                    await message.channel.send(reply_text)
                    return
            elif response_type == "standard_greeting":
                if is_standalone:
                    template = item.get("greeting_reply_template", "Hello {user_mention}! How can I help?")
                    actual_greeting_cased = original_message_content[:len(user_typed_greeting_text_for_item)].strip()
                    reply = template.replace("{actual_greeting_cased}", actual_greeting_cased) \
                                    .replace("{user_mention}", message.author.mention)
                    await message.channel.send(reply)
                    return
                else:
                    greeting_handled_or_skipped = True

        if greeting_handled_or_skipped:
             break

    extracted_code_name = ""
    call_code_intent_detected = False

    trigger_phrases_exact = {
        "how to code": None, "how to click": None, "how to press": None, "how to choose": None,
        "code for": None, "click for": None, "press for": None, "choose for": None,
        "what is the code for": None, "what is the click for": None,
        "what is the press for": None, "what is the choose for": None,
        "coding for": None, "clicking for": None, "pressing for": None, "choosing for": None,
        "how do i code": None, "how do i click": None, "how do i press": None, "how do i choose": None,
        "code": None, "click": None, "press": None, "choose": None,
    }

    for phrase in trigger_phrases_exact.keys():
        if phrase in user_query_lower:
            parts = user_query_lower.split(phrase, 1)
            if len(parts) > 1:
                idx_phrase_start = user_query_lower.find(phrase) # More reliable way to get start index
                if idx_phrase_start == 0 or user_query_lower[idx_phrase_start-1].isspace():
                    potential_code_name = parts[1].strip().lstrip(string.punctuation).strip()
                    if potential_code_name: # Ensure we extracted something
                        extracted_code_name = potential_code_name
                        call_code_intent_detected = True
                        break

    if call_code_intent_detected and extracted_code_name:
        temp_key_for_stripping = extracted_code_name
        stripped_leading_greeting = False
        for item_greet in greetings_config:
            if item_greet.get("response_type") == "standard_greeting":
                for keyword_json_greet in item_greet.get("keywords", []):
                    if temp_key_for_stripping.startswith(keyword_json_greet.lower()): # ensure keyword is lower
                        part_after_greeting = temp_key_for_stripping[len(keyword_json_greet):]
                        if not is_text_empty_or_punctuation_only(part_after_greeting) or not part_after_greeting:
                            temp_key_for_stripping = part_after_greeting.lstrip(string.punctuation + ' ').strip()
                            stripped_leading_greeting = True
                            break
            if stripped_leading_greeting:
                extracted_code_name = temp_key_for_stripping
                break

        if extracted_code_name:
            all_known_call_codes_json = faq_data.get("call_codes", {})
            call_codes_map_lower_to_original = {key.lower(): key for key in all_known_call_codes_json.keys()}

            matched_code_key_lower, code_score = get_best_fuzzy_match(
                extracted_code_name,
                list(call_codes_map_lower_to_original.keys()),
                FUZZY_MATCH_THRESHOLD_CALLCODES_STRICT
            )

            if matched_code_key_lower:
                original_cased_code_key = call_codes_map_lower_to_original[matched_code_key_lower]
                description = all_known_call_codes_json.get(original_cased_code_key)
                await message.channel.send(f"**How to code '{original_cased_code_key.title()}':**\n{description}")
                return
            else:
                await message.channel.send(f"I don't have specific coding instructions for '{extracted_code_name}'. "
                                           f"It's not a code I recognize. You can type `listcodes` to see known codes.")
                return
        else:
            pass


    for faq_item in faq_data.get("general_faqs", []):
        for keyword_from_json in faq_item.get("keywords", []):
            if len(keyword_from_json.split()) == 1: 
                for user_word in user_query_lower.split():
                    user_word_cleaned = user_word.strip(string.punctuation)
                    score = fuzz.ratio(keyword_from_json, user_word_cleaned)
                    if score >= FUZZY_MATCH_THRESHOLD_KEYWORDS + 2:
                        await message.channel.send(faq_item.get("answer", "Sorry, an answer is configured but missing."))
                        return

            else: 
                score = fuzz.token_set_ratio(keyword_from_json, user_query_lower)
                if score >= FUZZY_MATCH_THRESHOLD_KEYWORDS:
                    await message.channel.send(faq_item.get("answer", "Sorry, an answer is configured but missing."))
                    return

    fallback_message_template = faq_data.get("fallback_message", "I'm sorry, I don't have an answer for that right now. Please ask your manager.")
    await message.channel.send(fallback_message_template)
    return


# --- Run the Bot ---
if __name__ == "__main__":
    if TOKEN:
        try:
            bot.run(TOKEN)
        except discord.PrivilegedIntentsRequired:
            print("Error: Privileged Intents (Message Content) are not enabled for the bot in the Discord Developer Portal.")
            print("Please go to your bot's application page on the Discord Developer Portal and enable the 'Message Content Intent'.")
        except Exception as e:
            print(f"An error occurred while trying to run the bot: {e}")
    else:
        print("Error: DISCORD_TOKEN not found in .env file or environment variables. Bot cannot start.")