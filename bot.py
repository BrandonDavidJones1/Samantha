import discord
import os
import json
from dotenv import load_dotenv
import string
from thefuzz import process, fuzz
import logging
from sentence_transformers import SentenceTransformer, util
import torch
import re
import urllib.parse

# --- New Imports for Pronunciation Feature ---
import requests 
import asyncio
from discord import ui # For Buttons and Views

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
FAQ_FILE = "faq_data.json"
FUZZY_MATCH_THRESHOLD_GREETINGS = 75
LOG_FILE = "unanswered_queries.log"
SEMANTIC_SEARCH_THRESHOLD = 0.65 # Adjusted slightly higher after embedding all keywords
SUGGESTION_THRESHOLD = 0.45    # Adjusted slightly higher

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = discord.Client(intents=intents)

# --- Global Data ---
faq_data = {}
model = None
faq_embeddings = []
faq_questions = [] # This will store ALL individual keywords
faq_original_indices = [] # This will map each keyword back to its original FAQ item index

# --- Logging Setup ---
logger = logging.getLogger('discord_faq_bot')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.INFO)
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
            "greetings_and_pleasantries": [], "general_faqs": [], "call_codes": {},
            "fallback_message": "Sorry, I couldn't find an answer. Please ask your manager for assistance."
        }
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {FAQ_FILE}. Check its format.")
        faq_data = {
            "greetings_and_pleasantries": [], "general_faqs": [], "call_codes": {},
            "fallback_message": "Sorry, I couldn't find an answer. Please ask your manager for assistance."
        }
    build_semantic_embeddings()


def build_semantic_embeddings():
    global faq_embeddings, faq_questions, model, faq_original_indices # Add faq_original_indices
    try:
        if model is None:
            logger.info("Loading sentence transformer model ('all-MiniLM-L6-v2')...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded.")
        
        general_faqs_list = faq_data.get("general_faqs", [])
        if not isinstance(general_faqs_list, list):
            logger.error(f"general_faqs in {FAQ_FILE} is not a list. Semantic search may not work correctly.")
            general_faqs_list = []

        current_flat_faq_questions = []
        current_flat_faq_original_indices = []

        for original_idx, item in enumerate(general_faqs_list): # Use enumerate to get original index
            if isinstance(item, dict) and item.get("keywords"):
                keywords_data = item["keywords"]
                if isinstance(keywords_data, list):
                    for keyword_entry in keywords_data: # Iterate through all keywords
                        if isinstance(keyword_entry, str) and keyword_entry.strip():
                            current_flat_faq_questions.append(keyword_entry.strip().lower()) # Store lowercase for consistency
                            current_flat_faq_original_indices.append(original_idx)
                elif isinstance(keywords_data, str) and keywords_data.strip():
                    current_flat_faq_questions.append(keywords_data.strip().lower()) # Store lowercase for consistency
                    current_flat_faq_original_indices.append(original_idx)
        
        faq_questions = current_flat_faq_questions # This now holds all individual keywords
        faq_original_indices = current_flat_faq_original_indices

        if not model:
            logger.error("Model is not loaded. Cannot build embeddings.")
            faq_embeddings = [] 
            return

        if faq_questions: # If we have any keywords to embed
            logger.info(f"Encoding {len(faq_questions)} individual FAQ keywords/questions...")
            # Embeddings are created based on the (now lowercase) keywords
            faq_embeddings = model.encode(faq_questions, convert_to_tensor=True) 
            logger.info(f"FAQ embeddings created with shape: {faq_embeddings.shape}")
        else:
            logger.info("No FAQ keywords/questions to encode. Creating empty embeddings tensor.")
            embedding_dim = model.get_sentence_embedding_dimension()
            device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            faq_embeddings = torch.empty((0, embedding_dim), dtype=torch.float, device=device)
            logger.info(f"Empty FAQ embeddings created with shape: {faq_embeddings.shape}")

    except Exception as e:
        logger.exception("Error building semantic embeddings:")
        faq_questions = [] 
        faq_original_indices = []
        if model:
            try:
                embedding_dim = model.get_sentence_embedding_dimension()
                device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
                faq_embeddings = torch.empty((0, embedding_dim), dtype=torch.float, device=device)
            except Exception: faq_embeddings = [] 
        else: faq_embeddings = []

def is_text_empty_or_punctuation_only(text):
    return not text or all(char in string.punctuation or char.isspace() for char in text)

async def get_pronunciation_audio_url(word_or_phrase: str) -> str | None:
    """
    Fetches an audio pronunciation URL from the Free Dictionary API.
    Attempts to find US English audio first, then UK, then any.
    For multi-word phrases, it queries the first word.
    Returns the audio URL (str) or None if not found or an error occurs.
    """
    audio_url = None
    response_obj = None 
    try:
        # DictionaryAPI works best with single words. For phrases, try the first word.
        query_term = word_or_phrase.split(" ")[0].lower() 
        encoded_query = urllib.parse.quote_plus(query_term)
        api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{encoded_query}"
        
        loop = asyncio.get_event_loop()
        response_obj = await loop.run_in_executor(None, lambda: requests.get(api_url, timeout=7))
        response_obj.raise_for_status()
        data = response_obj.json()

        found_audio_urls = []
        if data and isinstance(data, list):
            for entry in data:
                if "phonetics" in entry and isinstance(entry["phonetics"], list):
                    for phonetic_info in entry["phonetics"]:
                        if "audio" in phonetic_info and isinstance(phonetic_info["audio"], str) and phonetic_info["audio"].startswith("http"):
                            found_audio_urls.append(phonetic_info["audio"])
        
        # Prioritize audio: US > UK > any other
        for url in found_audio_urls:
            if "us.mp3" in url.lower() or "en-us" in url.lower(): # Common US patterns
                audio_url = url
                break
        if not audio_url:
            for url in found_audio_urls:
                if "uk.mp3" in url.lower() or "en-gb" in url.lower(): # Common UK patterns
                    audio_url = url
                    break
        if not audio_url and found_audio_urls:
            audio_url = found_audio_urls[0] # Take the first available if no regional preference matched

        return audio_url
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            logger.info(f"Pronunciation API: Word '{word_or_phrase}' (query: '{query_term}') not found (404).")
        else:
            logger.error(f"Pronunciation API: HTTP error for '{word_or_phrase}': {http_err}")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Pronunciation API: Request timed out for '{word_or_phrase}'.")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Pronunciation API: Request exception for '{word_or_phrase}': {req_err}")
        return None
    except json.JSONDecodeError:
        resp_text = response_obj.text[:200] if response_obj and hasattr(response_obj, 'text') else "N/A"
        logger.error(f"Pronunciation API: JSON decode error for '{word_or_phrase}'. Response: {resp_text}")
        return None
    except Exception:
        logger.exception(f"Pronunciation API: Unexpected error fetching audio for '{word_or_phrase}':")
        return None

def semantic_search_best_match(query): # Query should be lowercased before passing here
    if model is None:
        logger.warning("Semantic search: Model not ready.")
        return None, 0.0
    
    if not isinstance(faq_embeddings, torch.Tensor) or faq_embeddings.shape[0] == 0:
        if isinstance(faq_embeddings, list) and not faq_embeddings:
             logger.warning("Semantic search: FAQ embeddings list is empty.")
        elif isinstance(faq_embeddings, torch.Tensor) and faq_embeddings.shape[0] == 0:
             logger.warning("Semantic search: FAQ embeddings tensor is empty.")
        else:
            logger.error(f"Semantic search: FAQ embeddings not a valid tensor or is empty. Type: {type(faq_embeddings)}")
        return None, 0.0
        
    query_embedding = model.encode(query, convert_to_tensor=True) # Query is already lowercased
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)

    current_faq_embeddings = faq_embeddings 

    if current_faq_embeddings.ndim == 1: 
        logger.warning(f"Semantic search: Global FAQ embeddings (shape {current_faq_embeddings.shape}) are 1D. Reshaping locally.")
        if model and hasattr(model, 'get_sentence_embedding_dimension'):
            emb_dim = model.get_sentence_embedding_dimension()
            if emb_dim > 0 and current_faq_embeddings.shape[0] == emb_dim:
                current_faq_embeddings = current_faq_embeddings.unsqueeze(0)
            else:
                logger.error(f"Semantic search: Unreshapable 1D FAQ embeddings ({current_faq_embeddings.shape}) vs model_dim {emb_dim}.")
                return None, 0.0
        else:
            logger.error("Semantic search: Cannot reshape 1D FAQ embeddings; model/dim info unavailable.")
            return None, 0.0
    
    if query_embedding.device != current_faq_embeddings.device:
        logger.warning(f"Semantic search: Device mismatch. Query: {query_embedding.device}, FAQs: {current_faq_embeddings.device}. Moving FAQs.")
        try:
            current_faq_embeddings = current_faq_embeddings.to(query_embedding.device)
        except Exception as e:
            logger.error(f"Semantic search: Failed to move FAQ embeddings to device {query_embedding.device}: {e}")
            return None, 0.0

    cosine_scores = util.pytorch_cos_sim(query_embedding, current_faq_embeddings)[0]
    
    if cosine_scores.numel() == 0:
        return None, 0.0

    best_score = float(torch.max(cosine_scores))
    best_idx = int(torch.argmax(cosine_scores))
    
    return best_idx, best_score


async def send_long_message(channel, text_content):
    MAX_LEN = 1980 
    if len(text_content) <= MAX_LEN:
        await channel.send(text_content)
    else:
        parts = []
        while len(text_content) > 0:
            if len(text_content) > MAX_LEN:
                split_at = text_content.rfind('\n', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 2 : 
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
    await bot.change_presence( activity = discord.Activity(
        name="DM me for HELP!",
        type=discord.ActivityType.custom,
        state="DM me for HELP!" # The 'state' field is often what shows for custom statuses
    ))
    logger.info("Bot started and ready.", extra={'user_id': 'SYSTEM', 'username': 'SYSTEM', 'extra_info': 'Bot Ready'})


@bot.event
async def on_message(message):
    if message.author == bot.user or not isinstance(message.channel, discord.DMChannel):
        return

    user_query_lower = message.content.lower().strip() # Lowercase once here
    original_message_content = message.content.strip() 
    log_user_info = {'user_id': message.author.id, 'username': str(message.author)}

    if is_text_empty_or_punctuation_only(original_message_content):
        logger.info(f"Ignoring empty or punctuation-only query: '{original_message_content}'", extra={**log_user_info, 'extra_info': 'Empty query ignored'})
        return

    # --- "!pronounce [word]" Command Handler ---
    pronounce_prefix = "!pronounce "
    pronounce_prefix_no_bang = "pronounce " # Already lowercase
    
    word_to_pronounce_input = None
    # user_query_lower is already lowercase
    if user_query_lower.startswith(pronounce_prefix):
        word_to_pronounce_input = original_message_content[len(pronounce_prefix):].strip()
    elif user_query_lower.startswith(pronounce_prefix_no_bang):
        temp_word = original_message_content[len(pronounce_prefix_no_bang):].strip()
        if temp_word and len(user_query_lower.split()) < 5: 
             word_to_pronounce_input = temp_word

    if word_to_pronounce_input:
        if not word_to_pronounce_input: 
            await message.channel.send("Please tell me what word or phrase you want to pronounce. Usage: `pronounce [word or phrase]`")
        else:
            async with message.channel.typing():
                audio_url = await get_pronunciation_audio_url(word_to_pronounce_input) # API handles casing if needed
            
            encoded_word_for_google = urllib.parse.quote_plus(word_to_pronounce_input)
            google_link = f"https://www.google.com/search?q=how+to+pronounce+{encoded_word_for_google}"
            youtube_link = f"https://www.youtube.com/playlist?list=PLvJSE3hDJAyN2a-i1GXZPXOpDPQZcerDc"

            view = discord.ui.View() 
            response_message_lines = [f"Pronunciation resources for \"**{word_to_pronounce_input}**\":"]
            
            log_audio_status = "Audio not found from API."
            if audio_url:
                play_button = discord.ui.Button(
                    label="Play Sound",
                    style=discord.ButtonStyle.link,
                    url=audio_url,
                    emoji="🔊" 
                )
                view.add_item(play_button)
                # response_message_lines.append(f"• Click the button to hear it from an API.") # Button is self-explanatory
                log_audio_status = f"Audio found from API: {audio_url}"
            else:
                response_message_lines.append(f"• Sorry, I couldn't find a direct audio pronunciation for \"{word_to_pronounce_input}\" from an API.")

            google_button = discord.ui.Button(
                label="Search on Google",
                style=discord.ButtonStyle.link,
                url=google_link
            )
            youtube_button = discord.ui.Button(
                label="Check LTS youtube Playlist",
                style=discord.ButtonStyle.link,
                url=youtube_link
            )
            view.add_item(google_button)
            view.add_item(youtube_button)
            
            # Simplified message:
            if not audio_url: # If only Google button is present, add a bit more context
                 response_message_lines.append(f"• You can check Google for pronunciation and other resources.")
            
            final_message_content = "\n".join(response_message_lines)

            if view.children: 
                 await message.channel.send(final_message_content, view=view)
            else: 
                 await message.channel.send(final_message_content) # Should not happen

            logger.info(f"Pronunciation requested for '{word_to_pronounce_input}'. {log_audio_status}. Original: '{original_message_content}'", extra=log_user_info)
        return 

    # --- "List Codes" Command Handler ---
    if user_query_lower == "list codes": # user_query_lower is already lowercase
        call_codes_data = faq_data.get("call_codes", {})
        if not call_codes_data:
            await message.channel.send("I don't have any call codes defined at the moment.")
            logger.info(f"Command 'list codes' processed. No codes found. Query: '{original_message_content}'", 
                        extra={**log_user_info, 'extra_info': 'List codes - no data'})
            return

        embed = discord.Embed(title="☎️ Call Disposition Codes", color=discord.Color.blue())
        current_field_value = ""
        field_count = 0
        MAX_FIELD_VALUE_LEN = 1020 
        MAX_FIELDS = 24 

        for code, description in call_codes_data.items():
            entry_text = f"**{code.upper()}**: {description}\n\n"
            if len(current_field_value) + len(entry_text) > MAX_FIELD_VALUE_LEN and field_count < MAX_FIELDS:
                embed.add_field(name=f"Codes (Part {field_count + 1})" if field_count > 0 else "Codes", value=current_field_value, inline=False)
                current_field_value = ""
                field_count += 1
            
            if field_count >= MAX_FIELDS:
                 logger.warning(f"List codes: Exceeded max embed fields ({MAX_FIELDS}). Some codes might be truncated.")
                 await message.channel.send("The list of codes is very long! Displaying a partial list. Please ask your supervisor for the complete list if needed.")
                 break 

            current_field_value += entry_text
        
        if current_field_value and field_count < MAX_FIELDS: 
            embed.add_field(name=f"Codes (Part {field_count + 1})" if field_count > 0 or not embed.fields else "Codes", value=current_field_value, inline=False)
        elif not embed.fields and not current_field_value: 
             await message.channel.send("No call codes could be formatted. Please check the data.")
             logger.info(f"Command 'list codes' processed. No codes formatted. Query: '{original_message_content}'", 
                        extra={**log_user_info, 'extra_info': 'List codes - formatting issue'})
             return

        if not embed.fields:
             await message.channel.send("I found call codes, but couldn't display them. Please try again or ask a manager.")
        else:
            await message.channel.send(embed=embed)
        
        logger.info(f"Command 'list codes' processed. Query: '{original_message_content}'", 
                    extra={**log_user_info, 'extra_info': 'List codes command executed'})
        return

    # --- Specific Call Code Definition Command ---
    # user_query_lower is already lowercase
    match_define_code = re.match(r"^(?:what is|define|explain)\s+([\w\s-]+)\??$", user_query_lower)
    if match_define_code:
        code_name_query_original_case = match_define_code.group(1).strip() # Keep original case for display if needed
        code_name_query_upper = code_name_query_original_case.upper() # For matching keys
        call_codes_data = faq_data.get("call_codes", {})
        
        # Direct match (case-insensitive keys in call_codes_data is assumed, but we search by UPPER)
        found_code_key = None
        for key_in_dict in call_codes_data.keys():
            if key_in_dict.upper() == code_name_query_upper:
                found_code_key = key_in_dict
                break
        
        if found_code_key:
            description = call_codes_data[found_code_key]
            embed = discord.Embed(title=f"Definition: {found_code_key.upper()}", description=description, color=discord.Color.purple())
            await message.channel.send(embed=embed)
            logger.info(f"Defined code '{found_code_key.upper()}' (exact). Query: '{original_message_content}'", extra=log_user_info)
            return
        else:
            # Fuzzy match on the original cased query against original cased keys for better fuzz results
            best_match_code, score = process.extractOne(code_name_query_original_case, call_codes_data.keys(), scorer=fuzz.token_set_ratio)
            if score > 80: 
                description = call_codes_data[best_match_code]
                # Display the matched key from the dictionary (which has original casing)
                embed = discord.Embed(title=f"Definition (for '{code_name_query_original_case}'): {best_match_code.upper()}", description=description, color=discord.Color.purple())
                await message.channel.send(embed=embed)
                logger.info(f"Defined code '{best_match_code.upper()}' (fuzzy, score {score}). Query: '{original_message_content}'", extra=log_user_info)
                return
    # --- Greetings Handler ---
    greetings_data = faq_data.get("greetings_and_pleasantries", [])
    for greeting_entry in greetings_data:
        keywords = greeting_entry.get("keywords", [])
        if not keywords: continue
        # For greetings, fuzzy matching on lowercased user query against original cased keywords is fine
        match_result = process.extractOne(user_query_lower, keywords, scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_MATCH_THRESHOLD_GREETINGS)
        if match_result:
            matched_keyword_from_fuzz, score = match_result 
            response_type = greeting_entry.get("response_type")
            if response_type == "standard_greeting":
                reply_template = greeting_entry.get("greeting_reply_template", "Hello there, {user_mention}!")
                # Find the original casing of the matched keyword for the reply
                actual_greeting_cased = matched_keyword_from_fuzz 
                for kw_original in keywords:
                    if kw_original.lower() == matched_keyword_from_fuzz.lower():
                        actual_greeting_cased = kw_original
                        break
                reply = reply_template.format(actual_greeting_cased=actual_greeting_cased.capitalize(), user_mention=message.author.mention)
                await message.channel.send(reply)
            elif response_type == "specific_reply":
                await message.channel.send(greeting_entry.get("reply_text", "I acknowledge that."))
            else: await message.channel.send(f"Hello {message.author.mention}!")
            logger.info(f"Greeting matched. Keyword: '{matched_keyword_from_fuzz}' (Score: {score}). Query: '{original_message_content}'", extra={**log_user_info, 'extra_info': 'Greeting answered'})
            return
    # --- Semantic Matching (as fallback) ---
    faq_items_original_list = faq_data.get("general_faqs", []) 
    if not isinstance(faq_items_original_list, list): 
        logger.error("general_faqs is not a list. Cannot perform semantic search.")
        faq_items_original_list = [] 

    # Semantic search uses user_query_lower (already lowercased)
    # faq_questions (global) is also already lowercased
    semantic_idx, semantic_score = semantic_search_best_match(user_query_lower)

    if semantic_idx is not None and 0 <= semantic_idx < len(faq_questions) and 0 <= semantic_idx < len(faq_original_indices):
        original_faq_item_index = faq_original_indices[semantic_idx] 

        if 0 <= original_faq_item_index < len(faq_items_original_list):
            matched_original_item = faq_items_original_list[original_faq_item_index] 
            answer = matched_original_item.get("answer", "Sorry, the answer for this item is missing.")
            
            # The primary keyword for the title is the actual keyword that matched from the flat list (which is lowercase)
            # We can capitalize its first letter for display.
            matched_keyword_for_title = faq_questions[semantic_idx].capitalize()

            log_extra_info = ""
            response_message_prefix = ""
            embed_title_prefix = "" 

            if semantic_score >= SEMANTIC_SEARCH_THRESHOLD:
                log_extra_info = f"Semantic FAQ Direct. Score: {semantic_score:.2f}"
                embed_title_prefix = "💡" 
            elif semantic_score >= SUGGESTION_THRESHOLD:
                log_extra_info = f"Semantic FAQ Suggestion. Score: {semantic_score:.2f}"
                response_message_prefix = (
                    f"I'm not sure I have an exact answer for that, but perhaps this is related to **'{matched_keyword_for_title}'**?\n\n"
                )
                embed_title_prefix = "🤔 Related to:" 
            
            if log_extra_info: 
                embed_title = f"{embed_title_prefix} {matched_keyword_for_title}"
                faq_embed = discord.Embed(title=embed_title, description=answer, color=discord.Color.green())
                
                if response_message_prefix: 
                     await message.channel.send(response_message_prefix)
                
                await message.channel.send(embed=faq_embed)
                
                logger.info(f"{log_extra_info}. Matched Keyword: '{faq_questions[semantic_idx]}'. User Query: '{original_message_content}'",
                            extra={**log_user_info, 'extra_info': 'Semantic match processed'})
                return
        else:
            logger.error(f"Semantic match error: original_faq_item_index {original_faq_item_index} out of bounds for faq_items (len {len(faq_items_original_list)}). semantic_idx: {semantic_idx}")


    # --- Fallback if nothing else matched or score too low for suggestion ---
    fallback_message_template = faq_data.get("fallback_message", "I'm sorry, I couldn't find an answer for that right now. Please ask your manager.")
    await message.channel.send(fallback_message_template)
    score_info_for_fallback = f"Last semantic score: {semantic_score:.2f}" if semantic_idx is not None else "Semantic search not applicable or failed pre-check"
    logger.info(f"Unanswered query. Fallback triggered. {score_info_for_fallback}. Query: '{original_message_content}'", extra={**log_user_info, 'extra_info': 'Fallback - No answer/suggestion'})

# --- Run the Bot ---
if __name__ == "__main__":
    if TOKEN:
        try: bot.run(TOKEN)
        except discord.PrivilegedIntentsRequired:
            print("Error: Privileged Intents (Message Content) are not enabled for the bot in the Discord Developer Portal.")
            logger.critical("Bot failed to run: Privileged Intents Required.", extra={'user_id': 'SYSTEM', 'username': 'SYSTEM', 'extra_info': 'Privileged Intents Error'})
        except Exception as e:
            print(f"An error occurred while trying to run the bot: {e}")
            logger.exception("Bot failed to run due to an unhandled exception.", extra={'user_id': 'SYSTEM', 'username': 'SYSTEM', 'extra_info': str(e)})
    else:
        print("Error: DISCORD_TOKEN not found in .env file or environment variables. Bot cannot start.")
        logger.critical("DISCORD_TOKEN not found. Bot cannot start.", extra={'user_id': 'SYSTEM', 'username': 'SYSTEM', 'extra_info': 'Token Missing'})