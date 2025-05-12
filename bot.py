import discord
import os
import json
from dotenv import load_dotenv
import string
from thefuzz import fuzz
import logging

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
FAQ_FILE = "faq_data.json"
FUZZY_MATCH_THRESHOLD_GREETINGS = 75
FUZZY_MATCH_THRESHOLD_KEYWORDS = 80  # For multi-word FAQ keywords
FUZZY_MATCH_THRESHOLD_KEYWORDS_SINGLE_WORD = 88 # Stricter for single-word FAQ keywords
FUZZY_MATCH_THRESHOLD_CALLCODES_STRICT = 80
LOG_FILE = "unanswered_queries.log"

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = discord.Client(intents=intents)

# --- Global Data ---
faq_data = {}

# --- Logging Setup ---
logger = logging.getLogger('discord_faq_bot')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.INFO)
# This formatter expects 'user_id', 'username', and 'extra_info' to be in the 'extra' dict for log calls.
# The main log message will be %(message)s.
formatter = logging.Formatter('%(asctime)s - %(levelname)s - User: %(user_id)s (%(username)s) - Query: %(message)s - %(extra_info)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

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
        save_faq_data() # Call save_faq_data if file not found
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
    best_match_str = None
    highest_score = 0
    if not query or not choices_list:
        return None, 0
        
    for choice in choices_list:
        score = fuzz.token_sort_ratio(query.lower(), choice.lower())
        if score > highest_score:
            highest_score = score
            best_match_str = choice

    if highest_score >= threshold:
        return best_match_str, highest_score
    return None, 0

async def send_long_message(channel, text_content):
    MAX_LEN = 1980
    if len(text_content) <= MAX_LEN:
        await channel.send(text_content)
    else:
        parts = []
        while len(text_content) > 0:
            if len(text_content) > MAX_LEN:
                split_at = text_content.rfind('\n', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 2:
                    split_at = text_content.rfind(' ', 0, MAX_LEN)
                    if split_at == -1 or split_at < MAX_LEN / 2:
                         split_at = MAX_LEN
                parts.append(text_content[:split_at])
                text_content = text_content[split_at:].lstrip()
            else:
                parts.append(text_content)
                break
        for part in parts:
            if part.strip():
                await channel.send(part)

# --- Event Handlers ---
@bot.event
async def on_ready():
    print(f'{bot.user.name} (ID: {bot.user.id}) has connected to Discord!')
    print(f'Listening for DMs. All interactions are handled as direct messages.')
    load_faq_data()
    await bot.change_presence(activity=discord.Game(name="DM me for help!"))
    logger.info("Bot started and ready.", extra={'user_id': 'SYSTEM', 'username': 'SYSTEM', 'extra_info': 'Bot Ready'})

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if not isinstance(message.channel, discord.DMChannel):
        return

    user_query_lower = message.content.lower().strip()
    original_message_content = message.content.strip()
    log_user_info = {'user_id': message.author.id, 'username': str(message.author)}

    if is_text_empty_or_punctuation_only(original_message_content) and not user_query_lower:
        logger.info(f"Ignoring empty or punctuation-only query: '{original_message_content}'", extra={**log_user_info, 'extra_info': 'Empty query ignored'})
        return

    # --- Stage 1: Handle 'listcodes' command ---
    if user_query_lower in ["listcodes", "list codes", "showcodes", "codes", "list code"]:
        codes = faq_data.get("call_codes", {})
        if not codes:
            await message.channel.send("No call codes are currently defined.")
            return
        response_message = "**Available Call Codes:**\n"
        for code, description in codes.items():
            response_message += f"\n**{code.title()}**:\n_{description}_\n"
        await send_long_message(message.channel, response_message)
        logger.info(f"Command 'listcodes' executed by user.", extra={**log_user_info, 'extra_info': 'List codes displayed'})
        return

    # --- Stage 2: Process Greetings ---
    greetings_config = faq_data.get("greetings_and_pleasantries", [])
    query_after_greeting_processing = user_query_lower
    greeting_was_standalone_and_replied = False

    for item in greetings_config:
        user_typed_greeting_text_for_item = ""
        best_score_for_item_greeting_match = 0
        for kw_json in item.get("keywords", []):
            kw_json_lower = kw_json.lower()
            for l_offset in range(-2, 3):
                user_prefix_len = len(kw_json_lower) + l_offset
                if user_prefix_len <= 0 or user_prefix_len > len(user_query_lower):
                    continue
                user_prefix_to_check = user_query_lower[:user_prefix_len]
                score = fuzz.token_set_ratio(kw_json_lower, user_prefix_to_check)

                if score > best_score_for_item_greeting_match and score >= FUZZY_MATCH_THRESHOLD_GREETINGS:
                    best_score_for_item_greeting_match = score
                    user_typed_greeting_text_for_item = user_prefix_to_check
        
        if user_typed_greeting_text_for_item:
            remaining_text_after_greeting_match = user_query_lower[len(user_typed_greeting_text_for_item):].strip()
            is_standalone_greeting_scenario = is_text_empty_or_punctuation_only(remaining_text_after_greeting_match)
            response_type = item.get("response_type")

            if response_type == "specific_reply":
                if is_standalone_greeting_scenario:
                    reply_text = item.get("reply_text", "Okay.")
                    await message.channel.send(reply_text)
                    log_msg = f"Greeting (specific_reply) matched: '{user_typed_greeting_text_for_item}'. Reply: '{reply_text}'. Original query: '{original_message_content}'"
                    logger.info(log_msg, extra={**log_user_info, 'extra_info': 'Specific greeting reply'})
                    greeting_was_standalone_and_replied = True
                    break 
            elif response_type == "standard_greeting":
                actual_greeting_cased = original_message_content[:len(user_typed_greeting_text_for_item)].strip()
                template = item.get("greeting_reply_template", "Hello {user_mention}! How can I help?")
                reply = template.replace("{actual_greeting_cased}", actual_greeting_cased) \
                                .replace("{user_mention}", message.author.mention)
                await message.channel.send(reply)
                log_msg = f"Greeting (standard_greeting) matched: '{user_typed_greeting_text_for_item}'. Reply: '{reply}'. Original query: '{original_message_content}'"
                logger.info(log_msg, extra={**log_user_info, 'extra_info': 'Standard greeting reply'})
                if is_standalone_greeting_scenario:
                    greeting_was_standalone_and_replied = True
                else:
                    query_after_greeting_processing = remaining_text_after_greeting_match
                break 
            
    if greeting_was_standalone_and_replied:
        return

    user_query_lower = query_after_greeting_processing.strip()

    if is_text_empty_or_punctuation_only(user_query_lower):
        logger.info(f"Query became empty/punctuation after greeting processing. Original: '{original_message_content}'.", extra={**log_user_info, 'extra_info': 'Query empty post-greeting'})
        return

    # --- Stage 3: Call Code Intent Detection ---
    extracted_code_name = ""
    call_code_intent_detected = False
    trigger_phrases_exact = {
        "how to code": True, "how to click": True, "how to press": True, "how to choose": True,
        "code for": True, "click for": True, "press for": True, "choose for": True,
        "what is the code for": True, "what is the click for": True,
        "what is the press for": True, "what is the choose for": True,
        "coding for": True, "clicking for": True, "pressing for": True, "choosing for": True,
        "how do i code": True, "how do i click": True, "how do i press": True, "how do i choose": True,
        "code": True, "click": True, "press": True, "choose": True,
    }
    sorted_trigger_phrases = sorted(trigger_phrases_exact.keys(), key=len, reverse=True)
    current_search_query_for_codes = user_query_lower 

    if current_search_query_for_codes:
        for phrase in sorted_trigger_phrases:
            if current_search_query_for_codes.startswith(phrase):
                if len(current_search_query_for_codes) == len(phrase) or \
                   (len(current_search_query_for_codes) > len(phrase) and \
                    (current_search_query_for_codes[len(phrase)].isspace() or current_search_query_for_codes[len(phrase)] in string.punctuation)):
                    
                    potential_code_name = current_search_query_for_codes[len(phrase):].strip().lstrip(string.punctuation).strip()
                    if potential_code_name:
                        extracted_code_name = potential_code_name
                        call_code_intent_detected = True
                        break
    
    if call_code_intent_detected and extracted_code_name:
        current_extracted_code_val = extracted_code_name
        while True:
            key_before_stripping_iteration = current_extracted_code_val
            stripped_in_this_iteration = False
            for item_greet in greetings_config:
                if item_greet.get("response_type") == "standard_greeting":
                    for kw_greet_json in item_greet.get("keywords", []):
                        kw_greet = kw_greet_json.lower()
                        if current_extracted_code_val.startswith(kw_greet):
                            if len(current_extracted_code_val) == len(kw_greet) or \
                               (len(current_extracted_code_val) > len(kw_greet) and \
                                (current_extracted_code_val[len(kw_greet)].isspace() or \
                                 current_extracted_code_val[len(kw_greet)] in string.punctuation)):
                                current_extracted_code_val = current_extracted_code_val[len(kw_greet):].lstrip(string.punctuation + ' ').strip()
                                stripped_in_this_iteration = True
                                break
                if stripped_in_this_iteration:
                    break
            if not stripped_in_this_iteration or current_extracted_code_val == key_before_stripping_iteration:
                break
        extracted_code_name = current_extracted_code_val

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
                response = f"**How to code '{original_cased_code_key.title()}':**\n{description}"
                await send_long_message(message.channel, response)
                log_msg = f"Call Code Matched. Extracted='{extracted_code_name}', Matched Code='{original_cased_code_key}' (Score: {code_score}). Original query: '{original_message_content}'"
                logger.info(log_msg, extra={**log_user_info, 'extra_info': 'Call code info provided'})
                return
            else:
                response = (f"I don't have specific coding instructions for '{extracted_code_name}'. "
                            f"It's not a code I recognize. You can type `listcodes` to see known codes.")
                await message.channel.send(response)
                log_msg = f"Call Code Not Found. Extracted='{extracted_code_name}', No match. Original query: '{original_message_content}'"
                logger.info(log_msg, extra={**log_user_info, 'extra_info': 'Call code not recognized'})
                return

    # --- Stage 4: General FAQs ---
    if user_query_lower:
        for faq_item in faq_data.get("general_faqs", []):
            faq_matched_in_this_item = False
            best_score_for_this_faq_item = 0 # Not currently used to decide between FAQs, but good for logging
            matched_keyword_for_log = ""

            for keyword_from_json_raw in faq_item.get("keywords", []):
                keyword_from_json = keyword_from_json_raw.lower().strip()
                if not keyword_from_json: continue

                current_keyword_score = 0
                if ' ' not in keyword_from_json:
                    for user_word in user_query_lower.split():
                        user_word_cleaned = user_word.strip(string.punctuation)
                        if not user_word_cleaned: continue
                        
                        score = fuzz.ratio(keyword_from_json, user_word_cleaned)
                        if score >= FUZZY_MATCH_THRESHOLD_KEYWORDS_SINGLE_WORD:
                            faq_matched_in_this_item = True
                            current_keyword_score = score
                            matched_keyword_for_log = keyword_from_json_raw
                            break 
                else:
                    score = fuzz.token_set_ratio(keyword_from_json, user_query_lower)
                    if score >= FUZZY_MATCH_THRESHOLD_KEYWORDS:
                        faq_matched_in_this_item = True
                        current_keyword_score = score
                        matched_keyword_for_log = keyword_from_json_raw
                
                if faq_matched_in_this_item:
                    if current_keyword_score > best_score_for_this_faq_item:
                        best_score_for_this_faq_item = current_keyword_score
                    
                    answer = faq_item.get("answer", "Sorry, an answer is configured but missing.")
                    await send_long_message(message.channel, answer)
                    log_msg = f"FAQ Matched. Keyword='{matched_keyword_for_log}' (Score: {current_keyword_score}), Answer Preview='{answer[:50]}...'. Original query: '{original_message_content}'"
                    logger.info(log_msg, extra={**log_user_info, 'extra_info': 'FAQ answered'})
                    return 
        
    # --- Stage 5: Fallback ---
    # This is the corrected logging for fallbacks:
    log_message_for_fallback = f"Unanswered query. Fallback triggered for: '{original_message_content}'"
    logger.info(log_message_for_fallback, extra={**log_user_info, 'extra_info': 'Fallback - No answer found'})

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
            logger.critical("Bot failed to run: Privileged Intents Required.", extra={'user_id': 'SYSTEM', 'username': 'SYSTEM', 'extra_info': 'Privileged Intents Error'})
        except Exception as e:
            print(f"An error occurred while trying to run the bot: {e}")
            logger.exception("Bot failed to run due to an unhandled exception.", extra={'user_id': 'SYSTEM', 'username': 'SYSTEM', 'extra_info': str(e)})
    else:
        print("Error: DISCORD_TOKEN not found in .env file or environment variables. Bot cannot start.")
        logger.critical("DISCORD_TOKEN not found. Bot cannot start.", extra={'user_id': 'SYSTEM', 'username': 'SYSTEM', 'extra_info': 'Token Missing'})