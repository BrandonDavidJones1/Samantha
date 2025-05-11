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
        # Using token_sort_ratio as it's generally good for matching phrases where word order might vary slightly.
        # For single keywords or exact code names, ratio or partial_ratio might also be considered,
        # but token_sort_ratio is a good general choice here.
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

    # --- Stage 1: Handle 'listcodes' command ---
    if user_query_lower in ["listcodes", "showcodes", "codes"]:
        codes = faq_data.get("call_codes", {})
        if not codes:
            await message.channel.send("No call codes are currently defined.")
            return
        response_message = "**Available Call Codes:**\n"
        for code, description in codes.items():
            response_message += f"\n**{code.title()}**:\n_{description}_\n"
        if len(response_message) > 1900: # Discord message limit is 2000
            response_message = response_message[:1900] + "\n\n... (message too long, truncated)"
        await message.channel.send(response_message)
        return

    # --- Stage 2: Process Greetings ---
    greetings_config = faq_data.get("greetings_and_pleasantries", [])
    query_after_greeting = user_query_lower
    # Use original_message_content for extracting cased greeting text
    # Use user_query_lower (as original_query_for_greeting_processing) for matching logic

    greeting_was_standalone_and_replied = False

    for item in greetings_config:
        user_typed_greeting_text_for_item = "" # Actual text matched
        best_score_for_item_greeting_match = 0

        for kw_json in item.get("keywords", []): # kw_json is assumed lowercase from FAQ_FILE
            # Fuzzy match kw_json against the beginning of the current user_query_lower
            # This loop allows for slight variations in the typed greeting length vs keyword length
            for l_offset in range(-2, 3):
                user_prefix_len = len(kw_json) + l_offset
                if user_prefix_len <= 0 or user_prefix_len > len(user_query_lower): # current user_query_lower
                    continue
                
                user_prefix_to_check = user_query_lower[:user_prefix_len] # from current user_query_lower
                score = fuzz.token_set_ratio(kw_json, user_prefix_to_check)

                if score > best_score_for_item_greeting_match and score >= FUZZY_MATCH_THRESHOLD_GREETINGS:
                    best_score_for_item_greeting_match = score
                    user_typed_greeting_text_for_item = user_prefix_to_check
        
        if user_typed_greeting_text_for_item: # A greeting keyword was matched at the start
            _remaining_text = user_query_lower[len(user_typed_greeting_text_for_item):].strip()
            _is_standalone_greeting = is_text_empty_or_punctuation_only(_remaining_text)
            response_type = item.get("response_type")

            if response_type == "specific_reply":
                if _is_standalone_greeting:
                    reply_text = item.get("reply_text", "Okay.")
                    await message.channel.send(reply_text)
                    greeting_was_standalone_and_replied = True
                    break # from greetings_config loop, effectively ends greeting processing for this message
            elif response_type == "standard_greeting":
                actual_greeting_cased = original_message_content[:len(user_typed_greeting_text_for_item)].strip()
                template = item.get("greeting_reply_template", "Hello {user_mention}! How can I help?")
                reply = template.replace("{actual_greeting_cased}", actual_greeting_cased) \
                                .replace("{user_mention}", message.author.mention)
                
                await message.channel.send(reply) # Send the greeting reply

                if _is_standalone_greeting:
                    greeting_was_standalone_and_replied = True
                    # Break from loop, reply sent, interaction ends after this stage
                else:
                    # Non-standalone standard greeting: reply sent, now update query for further processing
                    query_after_greeting = _remaining_text
                break # We've handled/responded to the initial greeting, break from greetings_config loop
            
    if greeting_was_standalone_and_replied:
        return # Interaction ended with a standalone greeting reply

    # Update user_query_lower for all subsequent processing stages based on greeting processing
    user_query_lower = query_after_greeting

    # If the query is now empty after non-standalone greeting processing, it will likely hit fallback or do nothing.
    if not user_query_lower.strip() and not greeting_was_standalone_and_replied :
        # If a greeting was sent (even non-standalone) and query is now empty,
        # it implies the greeting was the main/only actionable part.
        # If no greeting was sent and query is empty (e.g. user sent "   "), just fallback.
        # The current logic will naturally fall through to fallback if user_query_lower is empty.
        pass


    # --- Stage 3: Call Code Intent Detection ---
    # Uses the user_query_lower (which might have had an initial greeting removed)
    extracted_code_name = ""
    call_code_intent_detected = False

    # Trigger phrases: value `True` implies it expects a code name after it.
    trigger_phrases_exact = {
        "how to code": True, "how to click": True, "how to press": True, "how to choose": True,
        "code for": True, "click for": True, "press for": True, "choose for": True,
        "what is the code for": True, "what is the click for": True,
        "what is the press for": True, "what is the choose for": True,
        "coding for": True, "clicking for": True, "pressing for": True, "choosing for": True,
        "how do i code": True, "how do i click": True, "how do i press": True, "how do i choose": True,
        "code": True, "click": True, "press": True, "choose": True,
    }
    # Sort by length, descending, to match longer phrases first
    sorted_trigger_phrases = sorted(trigger_phrases_exact.keys(), key=len, reverse=True)

    if user_query_lower.strip(): # Only attempt if there's a query left
        for phrase in sorted_trigger_phrases:
            if user_query_lower.startswith(phrase):
                # Ensure it's a whole word/phrase match or followed by a clear separator
                if len(user_query_lower) == len(phrase) or \
                   (len(user_query_lower) > len(phrase) and \
                    (user_query_lower[len(phrase)].isspace() or user_query_lower[len(phrase)] in string.punctuation)):
                    
                    potential_code_name = user_query_lower[len(phrase):].strip().lstrip(string.punctuation).strip()
                    if potential_code_name: # Must be something *after* the trigger phrase for it to be a code name
                        extracted_code_name = potential_code_name
                        call_code_intent_detected = True
                        break # Found trigger and potential code name

    if call_code_intent_detected and extracted_code_name:
        # Iteratively strip known "standard greeting" keywords from the beginning of extracted_code_name
        current_extracted_code_val = extracted_code_name
        while True:
            key_before_stripping_iteration = current_extracted_code_val
            stripped_in_this_iteration = False
            for item_greet in greetings_config:
                if item_greet.get("response_type") == "standard_greeting": # Focus on general greetings
                    for kw_greet_json in item_greet.get("keywords", []):
                        kw_greet = kw_greet_json.lower() # ensure lowercase for comparison
                        if current_extracted_code_val.startswith(kw_greet):
                            # Check for whole word/phrase match at the start of current_extracted_code_val
                            if len(current_extracted_code_val) == len(kw_greet) or \
                               (len(current_extracted_code_val) > len(kw_greet) and \
                                (current_extracted_code_val[len(kw_greet)].isspace() or \
                                 current_extracted_code_val[len(kw_greet)] in string.punctuation)):
                                
                                current_extracted_code_val = current_extracted_code_val[len(kw_greet):].lstrip(string.punctuation + ' ').strip()
                                stripped_in_this_iteration = True
                                break # from kw_greet_json loop
                if stripped_in_this_iteration:
                    break # from item_greet loop to restart while loop with modified current_extracted_code_val
            
            if not stripped_in_this_iteration or current_extracted_code_val == key_before_stripping_iteration:
                break # No change or no greeting found, exit stripping loop
        
        extracted_code_name = current_extracted_code_val

        if extracted_code_name: # If something remains after stripping
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
        # If extracted_code_name became empty after stripping, fall through to general FAQs / fallback

    # --- Stage 4: General FAQs ---
    # Uses user_query_lower (which might have had initial greeting removed)
    if user_query_lower.strip(): # Only process if there's actual query text left
        for faq_item in faq_data.get("general_faqs", []):
            for keyword_from_json in faq_item.get("keywords", []): # keyword_from_json assumed lowercase
                # Check for single-word keywords vs multi-word phrases
                if len(keyword_from_json.split()) == 1: 
                    # For single keywords, check against individual words in the query
                    for user_word in user_query_lower.split():
                        user_word_cleaned = user_word.strip(string.punctuation)
                        score = fuzz.ratio(keyword_from_json, user_word_cleaned)
                        # Using a slightly higher threshold for single word matches to reduce false positives
                        if score >= FUZZY_MATCH_THRESHOLD_KEYWORDS + 2: # Kept original +2 logic
                            await message.channel.send(faq_item.get("answer", "Sorry, an answer is configured but missing."))
                            return
                else: 
                    # For multi-word keywords, match against the whole query
                    score = fuzz.token_set_ratio(keyword_from_json, user_query_lower)
                    if score >= FUZZY_MATCH_THRESHOLD_KEYWORDS:
                        await message.channel.send(faq_item.get("answer", "Sorry, an answer is configured but missing."))
                        return

    # --- Stage 5: Fallback ---
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