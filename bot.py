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
SEMANTIC_SEARCH_THRESHOLD = 0.65
SUGGESTION_THRESHOLD = 0.45
MAX_SUGGESTIONS_TO_SHOW = 2 # Max number of suggestions in mid-range

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
# Define a context filter to ensure custom log attributes have defaults
class ContextFilter(logging.Filter):
    def __init__(self, default_user_id='SYSTEM', default_username='SYSTEM', default_original_query='N/A', default_details=''):
        super().__init__()
        self.default_user_id = default_user_id
        self.default_username = default_username
        self.default_original_query = default_original_query
        self.default_details = default_details

    def filter(self, record):
        record.user_id = getattr(record, 'user_id', self.default_user_id)
        record.username = getattr(record, 'username', self.default_username)
        record.original_query_text = getattr(record, 'original_query_text', self.default_original_query)
        record.details = getattr(record, 'details', self.default_details)
        return True

logger = logging.getLogger('discord_faq_bot')
logger.setLevel(logging.INFO)
context_filter = ContextFilter()
logger.addFilter(context_filter) # Add filter to the logger instance

fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - User: %(user_id)s (%(username)s) - Query: %(original_query_text)s - Log: %(message)s - Details: %(details)s'
)
fh.setFormatter(formatter)
logger.addHandler(fh)

# --- Helper Functions ---
def load_faq_data():
    global faq_data # Ensure faq_data is global if modified
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
    global faq_embeddings, faq_questions, model, faq_original_indices
    logger.info("Attempting to build semantic embeddings...")
    try:
        if model is None:
            logger.info("Loading sentence transformer model ('all-MiniLM-L6-v2')...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded.")

        general_faqs_list = faq_data.get("general_faqs", [])
        logger.info(f"Raw general_faqs_list type: {type(general_faqs_list)}, length: {len(general_faqs_list) if isinstance(general_faqs_list, list) else 'N/A'}")

        if not isinstance(general_faqs_list, list):
            logger.error(f"CRITICAL: general_faqs in {FAQ_FILE} is not a list. Semantic search will FAIL.")
            general_faqs_list = []

        current_flat_faq_questions = []
        current_flat_faq_original_indices = []

        for original_idx, item in enumerate(general_faqs_list):
            if isinstance(item, dict) and item.get("keywords"):
                keywords_data = item["keywords"]
                if isinstance(keywords_data, list):
                    for keyword_entry in keywords_data:
                        if isinstance(keyword_entry, str) and keyword_entry.strip():
                            current_flat_faq_questions.append(keyword_entry.strip().lower())
                            current_flat_faq_original_indices.append(original_idx)
                elif isinstance(keywords_data, str) and keywords_data.strip():
                    current_flat_faq_questions.append(keywords_data.strip().lower())
                    current_flat_faq_original_indices.append(original_idx)

        logger.info(f"Total keywords extracted for embedding: {len(current_flat_faq_questions)}")
        if current_flat_faq_questions:
            logger.info(f"First few extracted keywords: {current_flat_faq_questions[:5]}")

        faq_questions = current_flat_faq_questions
        faq_original_indices = current_flat_faq_original_indices

        if not model:
            logger.error("CRITICAL: Model is not loaded. Cannot build embeddings.")
            faq_embeddings = []
            return

        if faq_questions:
            logger.info(f"Encoding {len(faq_questions)} individual FAQ keywords/questions...")
            faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
            logger.info(f"FAQ embeddings created with shape: {faq_embeddings.shape}")
        else:
            logger.warning("No FAQ keywords/questions to encode. Creating empty embeddings tensor.")
            embedding_dim = model.get_sentence_embedding_dimension()
            device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            faq_embeddings = torch.empty((0, embedding_dim), dtype=torch.float, device=device)
            logger.warning(f"Empty FAQ embeddings created with shape: {faq_embeddings.shape}")

    except Exception as e:
        logger.exception("CRITICAL Error building semantic embeddings:")
        faq_questions = []
        faq_original_indices = []
        if model:
            try:
                embedding_dim = model.get_sentence_embedding_dimension()
                device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
                faq_embeddings = torch.empty((0, embedding_dim), dtype=torch.float, device=device)
            except Exception: faq_embeddings = []
        else: faq_embeddings = []
    logger.info("Finished build_semantic_embeddings attempt.")


def is_text_empty_or_punctuation_only(text):
    return not text or all(char in string.punctuation or char.isspace() for char in text)

async def get_pronunciation_audio_url(word_or_phrase: str) -> str | None:
    audio_url = None
    response_obj = None
    try:
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
        for url in found_audio_urls:
            if "us.mp3" in url.lower() or "en-us" in url.lower():
                audio_url = url
                break
        if not audio_url:
            for url in found_audio_urls:
                if "uk.mp3" in url.lower() or "en-gb" in url.lower():
                    audio_url = url
                    break
        if not audio_url and found_audio_urls:
            audio_url = found_audio_urls[0]
        return audio_url
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            logger.info(f"Pronunciation API: Word '{word_or_phrase}' (query: '{query_term}') not found (404).")
        else:
            logger.error(f"Pronunciation API: HTTP error for '{word_or_phrase}': {http_err}")
    except requests.exceptions.Timeout:
        logger.error(f"Pronunciation API: Request timed out for '{word_or_phrase}'.")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Pronunciation API: Request exception for '{word_or_phrase}': {req_err}")
    except json.JSONDecodeError:
        resp_text = response_obj.text[:200] if response_obj and hasattr(response_obj, 'text') else "N/A"
        logger.error(f"Pronunciation API: JSON decode error for '{word_or_phrase}'. Response: {resp_text}")
    except Exception:
        logger.exception(f"Pronunciation API: Unexpected error fetching audio for '{word_or_phrase}':")
    return None

def semantic_search_top_n_matches(query, n=2):
    """Performs semantic search and returns top N matches (indices and scores)."""
    if model is None:
        logger.warning("Semantic search: Model not ready.")
        return [], []
    if not isinstance(faq_embeddings, torch.Tensor) or faq_embeddings.shape[0] == 0:
        logger.warning("Semantic search: FAQ embeddings tensor is empty or not a tensor.")
        return [], []

    query_embedding = model.encode(query, convert_to_tensor=True)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)

    current_faq_embeddings = faq_embeddings
    if current_faq_embeddings.ndim == 1: # Should not happen if build_semantic_embeddings is correct
        logger.error("Semantic search: FAQ embeddings are 1D, which is unexpected. Re-check build_semantic_embeddings.")
        return [], []
    if query_embedding.device != current_faq_embeddings.device:
        try:
            current_faq_embeddings = current_faq_embeddings.to(query_embedding.device)
        except Exception as e:
            logger.error(f"Semantic search: Failed to move FAQ embeddings to device {query_embedding.device}: {e}")
            return [], []

    cosine_scores = util.pytorch_cos_sim(query_embedding, current_faq_embeddings)[0]
    if cosine_scores.numel() == 0:
        return [], []

    actual_n = min(n, cosine_scores.numel())
    if actual_n == 0 : return [], []

    top_scores, top_indices = torch.topk(cosine_scores, actual_n)
    return top_indices.tolist(), top_scores.tolist()


async def send_long_message(channel, text_content, view=None):
    MAX_LEN = 1980
    messages_sent = []
    if len(text_content) <= MAX_LEN:
        msg = await channel.send(text_content, view=view)
        messages_sent.append(msg)
    else:
        parts = []
        temp_content = text_content
        while len(temp_content) > 0:
            if len(temp_content) > MAX_LEN:
                split_at = temp_content.rfind('\n\n', 0, MAX_LEN) # Prefer splitting at double newlines
                if split_at == -1 or split_at < MAX_LEN / 3: # If no double newline or too short, try single
                    split_at = temp_content.rfind('\n', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 3: # If no newline or too short, try space
                    split_at = temp_content.rfind(' ', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 3: # Force split if no good break found
                    split_at = MAX_LEN
                parts.append(temp_content[:split_at])
                temp_content = temp_content[split_at:].lstrip()
            else:
                parts.append(temp_content)
                break
        for i, part in enumerate(parts):
            if part.strip():
                # Only add the view to the last part
                current_view = view if i == len(parts) - 1 else None
                msg = await channel.send(part, view=current_view)
                messages_sent.append(msg)
    return messages_sent # Return list of message objects sent


# --- UI Classes for Interactive Buttons ---
class SuggestionButton(discord.ui.Button):
    def __init__(self, label, style, custom_id, original_query, matched_keyword_text, faq_item_index, helpful):
        super().__init__(label=label, style=style, custom_id=custom_id)
        self.original_query = original_query
        self.matched_keyword_text = matched_keyword_text
        self.faq_item_index = faq_item_index # Store the original FAQ item index if needed later
        self.helpful = helpful

    async def callback(self, interaction: discord.Interaction):
        feedback_type = "Helpful" if self.helpful else "Not Helpful"

        original_item_idx_from_flat_str = "N/A"
        if 0 <= self.faq_item_index < len(faq_original_indices):
            original_item_idx_from_flat_str = str(faq_original_indices[self.faq_item_index])

        log_message = f"Suggestion Feedback: User clicked '{feedback_type}'."
        log_extra = {
            'user_id': interaction.user.id,
            'username': str(interaction.user),
            'original_query_text': self.original_query,
            'details': (f"Feedback given: {feedback_type}. "
                        f"For matched keyword: '{self.matched_keyword_text}'. "
                        f"FAQ Item Index (flat): {self.faq_item_index}. "
                        f"Original FAQ Item Index: {original_item_idx_from_flat_str}.")
        }
        logger.info(
            log_message, extra=log_extra
        )

        await interaction.response.send_message(f"Thanks for your feedback on the suggestion about '{self.matched_keyword_text}'!", ephemeral=True)

        # Disable all buttons in the original message's view
        if self.view:
            for item in self.view.children:
                if isinstance(item, discord.ui.Button):
                    item.disabled = True
            try:
                await interaction.message.edit(view=self.view)
            except discord.NotFound:
                logger.warning(f"Could not find message {interaction.message.id} to disable buttons for suggestion feedback.")
            except discord.Forbidden:
                 logger.error(f"Missing permissions to edit message {interaction.message.id} for suggestion feedback.")


# --- Event Handlers ---
@bot.event
async def on_ready():
    print(f'{bot.user.name} (ID: {bot.user.id}) has connected to Discord!')
    print(f'Listening for DMs. All interactions are handled as direct messages.')
    load_faq_data()
    activity = discord.Activity(
        name="DM for HELP!", # Changed from "DM me for HELP!" to fit better
        type=discord.ActivityType.custom,
        state="DM for HELP!"
    )
    await bot.change_presence(activity=activity)
    logger.info("Bot started and ready.",
                extra={'details': 'Bot presence set, FAQ loaded, listening for DMs.'})


@bot.event
async def on_message(message: discord.Message): # Added type hint
    if message.author == bot.user or not isinstance(message.channel, discord.DMChannel):
        return

    user_query_lower = message.content.lower().strip()
    original_message_content = message.content.strip()

    if is_text_empty_or_punctuation_only(original_message_content):
        logger.info("Ignoring empty or punctuation-only query.",
                    extra={'user_id': message.author.id,
                           'username': str(message.author),
                           'original_query_text': original_message_content,
                           'details': 'Query was empty or only punctuation.'})
        return

    # --- Command Handlers (Order of Precedence) ---

    # 1. !pronounce Command
    pronounce_prefix = "!pronounce "
    pronounce_prefix_no_bang = "pronounce "
    word_to_pronounce_input = None
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
                audio_url = await get_pronunciation_audio_url(word_to_pronounce_input)
            encoded_word_for_google = urllib.parse.quote_plus(word_to_pronounce_input)
            google_link = f"https://www.google.com/search?q=how+to+pronounce+{encoded_word_for_google}"
            youtube_link = f"https://www.youtube.com/playlist?list=PLvJSE3hDJAyN2a-i1GXZPXOpDPQZcerDc" # Example playlist
            view = discord.ui.View()
            response_message_lines = [f"Pronunciation resources for \"**{word_to_pronounce_input}**\":"]
            log_audio_status = "Audio not found from API."
            if audio_url:
                view.add_item(discord.ui.Button(label="Play Sound", style=discord.ButtonStyle.link, url=audio_url, emoji="🔊"))
                log_audio_status = f"Audio found from API: {audio_url}"
            else:
                response_message_lines.append(f"• Sorry, I couldn't find a direct audio pronunciation for \"{word_to_pronounce_input}\" from an API.")
            view.add_item(discord.ui.Button(label="Search on Google", style=discord.ButtonStyle.link, url=google_link))
            view.add_item(discord.ui.Button(label="Check LTS YouTube Playlist", style=discord.ButtonStyle.link, url=youtube_link)) # Added example
            if not audio_url:
                 response_message_lines.append(f"• You can check Google/YouTube for pronunciation resources.")
            final_message_content = "\n".join(response_message_lines)
            if view.children:
                 await message.channel.send(final_message_content, view=view)
            else:
                 await message.channel.send(final_message_content)
            logger.info(f"Pronunciation request processed for '{word_to_pronounce_input}'.",
                        extra={'user_id': message.author.id,
                               'username': str(message.author),
                               'original_query_text': original_message_content,
                               'details': f"Audio from API: {log_audio_status}. Links provided."})
        return

    # 2. "list codes" Command
    if user_query_lower == "list codes":
        call_codes_data = faq_data.get("call_codes", {})
        if not call_codes_data:
            await message.channel.send("I don't have any call codes defined at the moment.")
            logger.info("Command 'list codes' - no codes found.",
                        extra={'user_id': message.author.id,
                               'username': str(message.author),
                               'original_query_text': original_message_content})
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
                 logger.warning(f"List codes: Exceeded max embed fields ({MAX_FIELDS}).")
                 await message.channel.send("List of codes too long, displaying partial. Ask supervisor for full list.")
                 break
            current_field_value += entry_text
        if current_field_value and field_count < MAX_FIELDS:
            embed.add_field(name=f"Codes (Part {field_count + 1})" if field_count > 0 or not embed.fields else "Codes", value=current_field_value, inline=False)
        elif not embed.fields and not current_field_value:
             await message.channel.send("No call codes formatted. Check data.")
             logger.info("Command 'list codes' - no codes formatted.",
                        extra={'user_id': message.author.id,
                               'username': str(message.author),
                               'original_query_text': original_message_content})
             return
        if not embed.fields: await message.channel.send("Found codes, but couldn't display. Try again/ask manager.")
        else: await message.channel.send(embed=embed)
        logger.info("Command 'list codes' processed.",
                    extra={'user_id': message.author.id, 'username': str(message.author), 'original_query_text': original_message_content, 'details': f"Displayed {len(call_codes_data) if call_codes_data else 0} codes."})
        return

    # 3. "define [code]" Command
    match_define_code = re.match(r"^(?:what is|define|explain)\s+([\w\s-]+)\??$", user_query_lower)
    if match_define_code:
        code_name_query_original_case = match_define_code.group(1).strip()
        code_name_query_upper = code_name_query_original_case.upper()
        call_codes_data = faq_data.get("call_codes", {})
        found_code_key = next((k for k in call_codes_data if k.upper() == code_name_query_upper), None)
        if found_code_key:
            embed = discord.Embed(title=f"Definition: {found_code_key.upper()}", description=call_codes_data[found_code_key], color=discord.Color.purple())
            await message.channel.send(embed=embed)
            logger.info(f"Defined code '{found_code_key.upper()}' (exact match).",
                        extra={'user_id': message.author.id,
                               'username': str(message.author),
                               'original_query_text': original_message_content,
                               'details': f"Query for code: '{code_name_query_original_case}'"})
            return
        else:
            best_match_code, score = process.extractOne(code_name_query_original_case, call_codes_data.keys(), scorer=fuzz.token_set_ratio)
            if score > 80:
                embed = discord.Embed(title=f"Definition (for '{code_name_query_original_case}'): {best_match_code.upper()}", description=call_codes_data[best_match_code], color=discord.Color.purple())
                await message.channel.send(embed=embed)
                logger.info(f"Defined code '{best_match_code.upper()}' (fuzzy match).",
                            extra={'user_id': message.author.id,
                                   'username': str(message.author),
                                   'original_query_text': original_message_content,
                                   'details': f"Query for code: '{code_name_query_original_case}', Score: {score}."})
                return

    # 4. Greetings Handler
    greetings_data = faq_data.get("greetings_and_pleasantries", [])
    for greeting_entry in greetings_data:
        keywords = greeting_entry.get("keywords", [])
        if not keywords: continue
        match_result = process.extractOne(user_query_lower, keywords, scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_MATCH_THRESHOLD_GREETINGS)
        if match_result:
            matched_keyword_from_fuzz, score = match_result
            response_type = greeting_entry.get("response_type")
            if response_type == "standard_greeting":
                reply_template = greeting_entry.get("greeting_reply_template", "Hello there, {user_mention}!")
                actual_greeting_cased = next((kw for kw in keywords if kw.lower() == matched_keyword_from_fuzz.lower()), matched_keyword_from_fuzz)
                reply = reply_template.format(actual_greeting_cased=actual_greeting_cased.capitalize(), user_mention=message.author.mention)
                await message.channel.send(reply)
            elif response_type == "specific_reply":
                await message.channel.send(greeting_entry.get("reply_text", "I acknowledge that."))
            else: await message.channel.send(f"Hello {message.author.mention}!")
            logger.info(f"Greeting matched.",
                        extra={'user_id': message.author.id,
                               'username': str(message.author),
                               'original_query_text': original_message_content,
                               'details': f"Matched keyword: '{matched_keyword_from_fuzz}', Score: {score}. Response type: {response_type}."})
            return

    # 5. Semantic Matching
    faq_items_original_list = faq_data.get("general_faqs", [])
    if not isinstance(faq_items_original_list, list):
        logger.error("general_faqs is not a list. Semantic search will fail.")
        faq_items_original_list = []

    top_indices, top_scores = semantic_search_top_n_matches(user_query_lower, n=MAX_SUGGESTIONS_TO_SHOW)

    if not top_indices:
        await message.channel.send(faq_data.get("fallback_message", "I'm sorry, I couldn't find an answer for that right now."))
        logger.info("Unanswered query, fallback sent.",
                    extra={'user_id': message.author.id,
                           'username': str(message.author),
                           'original_query_text': original_message_content,
                           'details': "No semantic hits found."})

        return

    # Primary match
    primary_faq_flat_idx = top_indices[0]
    primary_score = top_scores[0]

    if not (0 <= primary_faq_flat_idx < len(faq_questions) and 0 <= primary_faq_flat_idx < len(faq_original_indices)):
        logger.error(f"Semantic primary match index {primary_faq_flat_idx} out of bounds. Query: '{original_message_content}'")
        await message.channel.send(faq_data.get("fallback_message", "Sorry, an error occurred finding an answer."))
        return

    original_faq_item_idx_primary = faq_original_indices[primary_faq_flat_idx]
    if not (0 <= original_faq_item_idx_primary < len(faq_items_original_list)):
        logger.error(f"Semantic primary original item index {original_faq_item_idx_primary} out of bounds. Query: '{original_message_content}'")
        await message.channel.send(faq_data.get("fallback_message", "Sorry, an error occurred retrieving an answer."))
        return

    matched_item_primary = faq_items_original_list[original_faq_item_idx_primary]
    answer_primary = matched_item_primary.get("answer", "Answer not available.")
    matched_keyword_primary_text = faq_questions[primary_faq_flat_idx].capitalize()


    if primary_score >= SEMANTIC_SEARCH_THRESHOLD:
        embed = discord.Embed(title=f"💡 {matched_keyword_primary_text}", description=answer_primary, color=discord.Color.green())
        await send_long_message(message.channel, embed.description, view=None) # Embeds can't be sent via send_long_message directly
        logger.info("Semantic FAQ Direct Match.",
                    extra={'user_id': message.author.id,
                           'username': str(message.author),
                           'original_query_text': original_message_content,
                           'details': f"Score: {primary_score:.2f}. Matched keyword: '{matched_keyword_primary_text}'. FAQ item original index: {original_faq_item_idx_primary}."})


    elif primary_score >= SUGGESTION_THRESHOLD:
        suggestion_view = discord.ui.View(timeout=300) # Timeout for buttons in seconds (5 minutes)
        intro_message = f"I'm not sure I have an exact answer for '{original_message_content}', but perhaps one of these is helpful?\n"
        suggestion_messages_sent = await send_long_message(message.channel, intro_message) # Send intro first

        # Suggestion 1
        embed1 = discord.Embed(title=f"🤔 Suggestion 1: Related to '{matched_keyword_primary_text}'", description=answer_primary, color=discord.Color.gold())
        # Add buttons for the first suggestion
        btn_yes1 = SuggestionButton(label="✅ Helpful!", style=discord.ButtonStyle.success, custom_id=f"sugg_yes_{primary_faq_flat_idx}", original_query=original_message_content, matched_keyword_text=matched_keyword_primary_text, faq_item_index=primary_faq_flat_idx, helpful=True)
        btn_no1 = SuggestionButton(label="❌ Not quite", style=discord.ButtonStyle.danger, custom_id=f"sugg_no_{primary_faq_flat_idx}", original_query=original_message_content, matched_keyword_text=matched_keyword_primary_text, faq_item_index=primary_faq_flat_idx, helpful=False)
        current_view1 = discord.ui.View(timeout=300)
        current_view1.add_item(btn_yes1)
        current_view1.add_item(btn_no1)
        await message.channel.send(embed=embed1, view=current_view1)
        logger.info("Semantic Suggestion 1 offered.",
                    extra={'user_id': message.author.id,
                           'username': str(message.author),
                           'original_query_text': original_message_content,
                           'details': f"Score: {primary_score:.2f}. Matched keyword: '{matched_keyword_primary_text}'. FAQ item flat index: {primary_faq_flat_idx}."})

        # Suggestion 2 (if exists and meets threshold)
        if len(top_indices) > 1:
            secondary_faq_flat_idx = top_indices[1]
            secondary_score = top_scores[1]
            if secondary_score >= SUGGESTION_THRESHOLD:
                if not (0 <= secondary_faq_flat_idx < len(faq_questions) and 0 <= secondary_faq_flat_idx < len(faq_original_indices)):
                    logger.error(f"Semantic secondary match index {secondary_faq_flat_idx} out of bounds. Query: '{original_message_content}'")
                else:
                    original_faq_item_idx_secondary = faq_original_indices[secondary_faq_flat_idx]
                    if not (0 <= original_faq_item_idx_secondary < len(faq_items_original_list)):
                         logger.error(f"Semantic secondary original item index {original_faq_item_idx_secondary} out of bounds. Query: '{original_message_content}'")
                    else:
                        matched_item_secondary = faq_items_original_list[original_faq_item_idx_secondary]
                        answer_secondary = matched_item_secondary.get("answer", "Answer not available.")
                        matched_keyword_secondary_text = faq_questions[secondary_faq_flat_idx].capitalize()

                        embed2 = discord.Embed(title=f"🤔 Suggestion 2: Related to '{matched_keyword_secondary_text}'", description=answer_secondary, color=discord.Color.gold())
                        btn_yes2 = SuggestionButton(label="✅ Helpful!", style=discord.ButtonStyle.success, custom_id=f"sugg_yes_{secondary_faq_flat_idx}", original_query=original_message_content, matched_keyword_text=matched_keyword_secondary_text, faq_item_index=secondary_faq_flat_idx, helpful=True)
                        btn_no2 = SuggestionButton(label="❌ Not quite", style=discord.ButtonStyle.danger, custom_id=f"sugg_no_{secondary_faq_flat_idx}", original_query=original_message_content, matched_keyword_text=matched_keyword_secondary_text, faq_item_index=secondary_faq_flat_idx, helpful=False)
                        current_view2 = discord.ui.View(timeout=300)
                        current_view2.add_item(btn_yes2)
                        current_view2.add_item(btn_no2)
                        await message.channel.send(embed=embed2, view=current_view2)
                        logger.info("Semantic Suggestion 2 offered.",
                                    extra={'user_id': message.author.id,
                                           'username': str(message.author),
                                           'original_query_text': original_message_content,
                                           'details': f"Score: {secondary_score:.2f}. Matched keyword: '{matched_keyword_secondary_text}'. FAQ item flat index: {secondary_faq_flat_idx}."})
    else:
        # Fallback if score is below SUGGESTION_THRESHOLD
        details_str = f"Primary score {primary_score:.2f} below suggestion threshold {SUGGESTION_THRESHOLD}."
        await message.channel.send(faq_data.get("fallback_message", "I'm sorry, I couldn't find a clear answer for that."))
        logger.info("Unanswered query, fallback sent.",
                    extra={'user_id': message.author.id,
                           'username': str(message.author),
                           'original_query_text': original_message_content,
                           'details': details_str})


# --- Run the Bot ---
if __name__ == "__main__":
    if TOKEN:
        try:
            bot.run(TOKEN)
        except discord.PrivilegedIntentsRequired:
            print("Error: Privileged Intents (Message Content) are not enabled for the bot in the Discord Developer Portal.")
            logger.critical("Bot failed to run: Privileged Intents Required.", extra={'details': 'Privileged Intents Error'})
        except Exception as e:
            print(f"An error occurred while trying to run the bot: {e}")
            logger.exception("Bot failed to run due to an unhandled exception.", extra={'details': str(e)})
    else:
        print("Error: DISCORD_TOKEN not found in .env file or environment variables. Bot cannot start.")
        logger.critical("DISCORD_TOKEN not found. Bot cannot start.", extra={'details': 'Token Missing'})