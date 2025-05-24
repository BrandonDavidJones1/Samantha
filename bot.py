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
import requests # Added for fetching from URL
import asyncio
from discord import ui # For Buttons and Views
import collections # Added for deque
import spacy # Added for pronoun resolution

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
FAQ_URL = "https://raw.githubusercontent.com/BrandonDavidJones1/Samantha/main/faq_data.json"
FUZZY_MATCH_THRESHOLD_GREETINGS = 75
LOG_FILE = "unanswered_queries.log"
SEMANTIC_SEARCH_THRESHOLD = 0.65
SUGGESTION_THRESHOLD = 0.45
MAX_SUGGESTIONS_TO_SHOW = 2
CANDIDATES_TO_FETCH_FOR_SUGGESTIONS = 5

DYNAMIC_THRESHOLD_ENABLED = True
CONFIDENCE_GAP_FOR_DIRECT_ANSWER = 0.15
SIMILAR_SCORE_CLUSTER_THRESHOLD = 0.07
ABSOLUTE_HIGH_SCORE_OVERRIDE = 0.95

CONTEXT_WINDOW_SIZE = 3
CONTEXT_QUERY_LENGTH_THRESHOLD = 6 # Max words for a query to be considered "short" for context
CONTEXT_PRONOUNS = {
    "it", "that", "this", "those", "them", "they", "he", "she", "him", "her", "its", # Subject/Object
    "one" # Can sometimes be used contextually e.g. "how about that one?"
}
# For pronoun replacement, map pronoun to a generic "thing" if we can't find a better noun
PRONOUN_MAP_SUBJECT = {"it", "that", "this", "he", "she", "they"} # Pronouns that can act as subjects
PRONOUN_MAP_OBJECT = {"it", "that", "this", "him", "her", "them", "one"} # Pronouns that can act as objects

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = discord.Client(intents=intents)

# --- Global Data ---
faq_data = {}
model = None
faq_embeddings = []
faq_questions = []
faq_original_indices = []
user_context_history = {} # Stores deque for each user: (user_query_text, bot_answer_text_or_object)
nlp = None # spaCy model

# --- Admin and Log Forwarding Globals ---
ADMIN_USER_IDS = {
    1342311589298311230, # Example IDs, replace with actual integer IDs
    770409748922368000,
    1011068427189887037,
}
admin_log_activation_status = {} # Stores {admin_id: bool}

# --- Logging Setup ---
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
logger.addFilter(context_filter)

fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - User: %(user_id)s (%(username)s) - Query: %(original_query_text)s - Log: %(message)s - Details: %(details)s'
)
fh.setFormatter(formatter)
logger.addHandler(fh)

# --- Helper Functions ---
def load_spacy_model():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
            print("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
            nlp = None # Ensure it's None if loading failed
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading spaCy model: {e}")
            nlp = None

def get_likely_referent_from_previous_query(text: str) -> str | None:
    if not nlp or not text:
        return None
    doc = nlp(text)
    # Prioritize noun chunks first, then individual nouns/proper nouns
    # Take the last one as a simple heuristic for recency
    
    # Noun Chunks
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks]
    if noun_chunks:
        # Filter out very short/common noun chunks that might be pronouns themselves
        filtered_chunks = [nc for nc in noun_chunks if len(nc.split()) > 1 or (len(nc.split()) == 1 and nc.lower() not in CONTEXT_PRONOUNS)]
        if filtered_chunks:
            logger.debug(f"Pronoun Resolution: Found noun chunks in prev_query '{text}': {filtered_chunks}. Using last: '{filtered_chunks[-1]}'")
            return filtered_chunks[-1]
        elif noun_chunks: # If all chunks were pronouns, maybe pick the last one if it's not too generic
            if noun_chunks[-1].lower() not in ["it", "this", "that"]: # Avoid replacing a pronoun with another generic pronoun
                logger.debug(f"Pronoun Resolution: Found only pronoun-like noun chunks. Using last non-generic: '{noun_chunks[-1]}'")
                return noun_chunks[-1]

    # Individual Nouns/Proper Nouns
    potential_referents = []
    for token in reversed(doc): # Iterate backwards to get the last occurring ones first
        if token.pos_ in ["NOUN", "PROPN"]:
            # Avoid using pronouns themselves as referents from the previous query
            if token.text.lower() not in CONTEXT_PRONOUNS:
                potential_referents.append(token.text)
    
    if potential_referents:
        logger.debug(f"Pronoun Resolution: Found nouns/propns in prev_query '{text}': {potential_referents}. Using most recent: '{potential_referents[0]}'")
        return potential_referents[0] # First one from reversed list is the last occurring

    logger.debug(f"Pronoun Resolution: No clear noun/propn referent found in prev_query '{text}'.")
    return None

def load_faq_data_from_url():
    global faq_data
    default_faq_structure = {
        "greetings_and_pleasantries": [],
        "general_faqs": [],
        "call_codes": {},
        "fallback_message": "Sorry, I couldn't find an answer. Critical configuration error: FAQ data unavailable."
    }
    try:
        print(f"Attempting to download FAQ data from: {FAQ_URL}")
        response = requests.get(FAQ_URL, timeout=15)
        response.raise_for_status()
        faq_data = response.json()
        print(f"Successfully downloaded and parsed FAQ data from {FAQ_URL}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching FAQ data: {http_err} from {FAQ_URL}")
        logger.error(f"HTTP error occurred while fetching FAQ data from {FAQ_URL}: {http_err}")
        faq_data = default_faq_structure
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred while fetching FAQ data: {conn_err} from {FAQ_URL}")
        logger.error(f"Connection error occurred while fetching FAQ data from {FAQ_URL}: {conn_err}")
        faq_data = default_faq_structure
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout occurred while fetching FAQ data: {timeout_err} from {FAQ_URL}")
        logger.error(f"Timeout occurred while fetching FAQ data from {FAQ_URL}: {timeout_err}")
        faq_data = default_faq_structure
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request for FAQ data: {req_err} from {FAQ_URL}")
        logger.error(f"An error occurred during the request for FAQ data from {FAQ_URL}: {req_err}")
        faq_data = default_faq_structure
    except json.JSONDecodeError as json_err:
        print(f"Error: Could not decode JSON from fetched FAQ data from {FAQ_URL}. Error: {json_err}")
        response_text_snippet = ""
        if 'response' in locals() and hasattr(response, 'text'):
            response_text_snippet = response.text[:500] if response.text else "Response text was empty."
        logger.error(f"Could not decode JSON from {FAQ_URL}. Response text snippet: {response_text_snippet}")
        faq_data = default_faq_structure
    except Exception as e:
        print(f"An unexpected error occurred while loading FAQ data from URL {FAQ_URL}: {e}")
        logger.exception(f"Unexpected error loading FAQ data from {FAQ_URL}")
        faq_data = default_faq_structure

    faq_data.setdefault("greetings_and_pleasantries", [])
    faq_data.setdefault("general_faqs", [])
    faq_data.setdefault("call_codes", {})
    faq_data.setdefault("fallback_message", default_faq_structure["fallback_message"])

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
        if not isinstance(general_faqs_list, list):
            logger.error(f"CRITICAL: general_faqs in {FAQ_URL} result is not a list. Semantic search will FAIL.")
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

        faq_questions = current_flat_faq_questions
        faq_original_indices = current_flat_faq_original_indices

        if not model:
            logger.error("CRITICAL: Model is not loaded. Cannot build embeddings.")
            faq_embeddings = []
            return

        if faq_questions:
            faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
            logger.info(f"FAQ embeddings created with shape: {faq_embeddings.shape}")
        else:
            logger.warning("No FAQ keywords/questions to encode. Creating empty embeddings tensor.")
            embedding_dim = model.get_sentence_embedding_dimension()
            device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            faq_embeddings = torch.empty((0, embedding_dim), dtype=torch.float, device=device)

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
        # Try to get the first word for pronunciation, as APIs are often word-based
        query_term = word_or_phrase.split(" ")[0].lower() # Get first word
        encoded_query = urllib.parse.quote_plus(query_term)
        api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{encoded_query}"
        
        # Run requests in an executor to avoid blocking asyncio event loop
        loop = asyncio.get_event_loop()
        response_obj = await loop.run_in_executor(None, lambda: requests.get(api_url, timeout=7))
        response_obj.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        data = response_obj.json()

        found_audio_urls = []
        if data and isinstance(data, list): # API returns a list of entries
            for entry in data:
                if "phonetics" in entry and isinstance(entry["phonetics"], list):
                    for phonetic_info in entry["phonetics"]:
                        if "audio" in phonetic_info and isinstance(phonetic_info["audio"], str) and phonetic_info["audio"].startswith("http"):
                            found_audio_urls.append(phonetic_info["audio"])
        
        # Prioritize US, then UK, then any other audio
        for url in found_audio_urls:
            if "us.mp3" in url.lower() or "en-us" in url.lower(): # Prioritize US English
                audio_url = url
                break
        if not audio_url:
            for url in found_audio_urls:
                if "uk.mp3" in url.lower() or "en-gb" in url.lower(): # Then UK English
                    audio_url = url
                    break
        if not audio_url and found_audio_urls: # Fallback to the first audio found
            audio_url = found_audio_urls[0]
            
        return audio_url
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            logger.info(f"Pronunciation API: Word '{word_or_phrase}' (query: '{query_term}') not found (404).")
        else:
            logger.error(f"Pronunciation API: HTTP error for '{word_or_phrase}': {http_err}")
    except requests.exceptions.Timeout:
        logger.error(f"Pronunciation API: Request timed out for '{word_or_phrase}'.")
    except requests.exceptions.RequestException as req_err: # Catch other request-related errors
        logger.error(f"Pronunciation API: Request exception for '{word_or_phrase}': {req_err}")
    except json.JSONDecodeError:
        resp_text = response_obj.text[:200] if response_obj and hasattr(response_obj, 'text') else "N/A"
        logger.error(f"Pronunciation API: JSON decode error for '{word_or_phrase}'. Response: {resp_text}")
    except Exception: # Catch-all for other unexpected errors
        logger.exception(f"Pronunciation API: Unexpected error fetching audio for '{word_or_phrase}':")
    return None


def semantic_search_top_n_matches(query, n=2):
    if model is None:
        logger.warning("Semantic search: Model not ready.")
        return [], []
    if not isinstance(faq_embeddings, torch.Tensor) or faq_embeddings.shape[0] == 0:
        logger.warning("Semantic search: FAQ embeddings tensor is empty or not a tensor.")
        return [], []

    query_embedding = model.encode(query, convert_to_tensor=True)
    if query_embedding.ndim == 1: # Ensure query_embedding is 2D
        query_embedding = query_embedding.unsqueeze(0)

    current_faq_embeddings = faq_embeddings # Use a local var for clarity
    if current_faq_embeddings.ndim == 1: # Should ideally be 2D [num_faqs, embedding_dim]
        logger.error("Semantic search: FAQ embeddings are 1D, which is unexpected. Re-check build_semantic_embeddings.")
        return [], [] # Cannot compare if embeddings are malformed

    # Ensure embeddings are on the same device as the model/query_embedding
    if query_embedding.device != current_faq_embeddings.device:
        try:
            current_faq_embeddings = current_faq_embeddings.to(query_embedding.device)
        except Exception as e:
            logger.error(f"Semantic search: Failed to move FAQ embeddings to device {query_embedding.device}: {e}")
            return [], []


    cosine_scores = util.pytorch_cos_sim(query_embedding, current_faq_embeddings)[0] # Get 1D tensor of scores

    # Check if cosine_scores is empty (e.g., if faq_embeddings was [0, dim])
    if cosine_scores.numel() == 0:
        return [], []

    actual_n = min(n, cosine_scores.numel()) # Don't request more items than available
    if actual_n == 0 : return [], []


    top_scores, top_indices = torch.topk(cosine_scores, actual_n)
    return top_indices.tolist(), top_scores.tolist()


async def send_long_message(channel, text_content, view=None):
    MAX_LEN = 1980 # Discord's message length limit is 2000, 1980 gives buffer
    messages_sent = []
    if len(text_content) <= MAX_LEN:
        msg = await channel.send(text_content, view=view)
        messages_sent.append(msg)
    else:
        parts = []
        temp_content = text_content
        while len(temp_content) > 0:
            if len(temp_content) > MAX_LEN:
                # Try to split at double newline, then single, then space, then hard split
                split_at = temp_content.rfind('\n\n', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 3: # Heuristic: prefer double newline if it's not too early
                    split_at = temp_content.rfind('\n', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 3: # Prefer single newline if reasonable
                    split_at = temp_content.rfind(' ', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 3: # Fallback to hard split
                    split_at = MAX_LEN
                
                parts.append(temp_content[:split_at])
                temp_content = temp_content[split_at:].lstrip() # Remove leading whitespace from next part
            else:
                parts.append(temp_content)
                break
        
        for i, part in enumerate(parts):
            if part.strip(): # Ensure part is not just whitespace
                current_view = view if i == len(parts) - 1 else None # Only add view to the last part
                msg = await channel.send(part, view=current_view)
                messages_sent.append(msg)
    return messages_sent

# We'll need a small helper to parse ordinals/cardinals if we go detailed
def parse_ordinal_or_number_to_index(text: str) -> int | None:
    """
    Parses text like "first", "1st", "one", "2", "last" into a 0-based index.
    "last" is returned as -1.
    Assumes `text` is a single word/token as extracted by prior regex.
    """
    text_lower = text.lower().strip()
    # More comprehensive mapping
    mapping = {
        "first": 0, "1st": 0, "one": 0, "1": 0,
        "second": 1, "2nd": 1, "two": 1, "2": 1,
        "third": 2, "3rd": 2, "three": 2, "3": 2,
        "fourth": 3, "4th": 3, "four": 3, "4": 3,
        "fifth": 4, "5th": 4, "five": 4, "5": 4,
        # Add more as needed
        "last": -1
    }
    # Check for whole word matches from the mapping
    # Since `text` is expected to be a single token like "first" or "1",
    # we can directly check if `text_lower` is a key in the mapping.
    if text_lower in mapping:
        return mapping[text_lower]

    # If no keyword match, try to parse as a number (already handled if "1"-"5" are in mapping)
    # This handles numbers like "6", "7", etc.
    try:
        num = int(text_lower)
        if num > 0:
            return num - 1 # Convert 1-based to 0-based index
    except ValueError:
        pass # Not a valid number

    return None


async def forward_qa_to_admins(user_query: str, bot_answer_obj: any, original_author: discord.User):
    bot_answer_str = ""
    if isinstance(bot_answer_obj, dict) and "display_text_for_log" in bot_answer_obj:
        bot_answer_str = bot_answer_obj["display_text_for_log"]
    elif isinstance(bot_answer_obj, str):
        bot_answer_str = bot_answer_obj
    elif bot_answer_obj is None: # Can happen if a command doesn't generate a direct reply string for logging
        return 
    else: # Fallback for other types
        bot_answer_str = str(bot_answer_obj)

    if not bot_answer_str or not bot_answer_str.strip(): # If after processing, it's an empty or whitespace-only string
        return

    # --- Rest of the function from the base code ---
    safe_user_query = user_query.replace("`", "'") # Basic sanitization for markdown
    safe_bot_answer = bot_answer_str.replace("`", "'")

    MAX_QUERY_LOG_LEN = 600
    MAX_ANSWER_LOG_LEN = 1200 # Total message length is ~2000

    log_query_part = safe_user_query
    if len(log_query_part) > MAX_QUERY_LOG_LEN:
        log_query_part = log_query_part[:MAX_QUERY_LOG_LEN - 3] + "..."

    log_answer_part = safe_bot_answer
    if len(log_answer_part) > MAX_ANSWER_LOG_LEN:
        log_answer_part = log_answer_part[:MAX_ANSWER_LOG_LEN - 3] + "..."
    
    forward_message_base = (
        f"**User:** {original_author.name}#{original_author.discriminator} (ID: {original_author.id})\n"
        f"**Asked:** ```\n{log_query_part}\n```\n"
        f"**Bot Answered:** ```\n{log_answer_part}\n```"
    )
    forward_message = forward_message_base

    if len(forward_message) > 1990: # Max Discord message length approx 2000
        error_notice = "... (Log message content was too long to display fully)"
        # Further truncate query and answer to fit the error notice
        max_query_fallback_len = max(50, MAX_QUERY_LOG_LEN // 3) # Ensure at least some query context
        log_query_part_fallback = safe_user_query[:max_query_fallback_len] + "..."

        base_len_for_fallback = len(
            f"**User:** {original_author.name}#{original_author.discriminator} (ID: {original_author.id})\n"
            f"**Asked:** ```\n{log_query_part_fallback}\n```\n"
            f"**Bot Answered:** ```\n\n```" # Placeholder for answer part
        )
        max_answer_fallback_len = 1990 - base_len_for_fallback - len(error_notice) - 10 # -10 for safety/newlines
        max_answer_fallback_len = max(50, max_answer_fallback_len) # Ensure some answer context

        log_answer_part_fallback = safe_bot_answer[:max_answer_fallback_len - 3] + "..." + f"\n{error_notice}"

        forward_message = (
            f"**User:** {original_author.name}#{original_author.discriminator} (ID: {original_author.id})\n"
            f"**Asked:** ```\n{log_query_part_fallback}\n```\n"
            f"**Bot Answered:** ```\n{log_answer_part_fallback}\n```"
        )
        if len(forward_message) > 1990: # Final fallback if still too long
            forward_message = (
                f"**User:** {original_author.name}#{original_author.discriminator} (ID: {original_author.id})\n"
                f"**Asked:** Query received (log too long to display fully).\n"
                f"**Bot Answered:** Answer provided (log too long to display fully)."
            )
    
    active_admin_ids_to_notify = [admin_id for admin_id, is_active in admin_log_activation_status.items() if is_active]

    if not active_admin_ids_to_notify:
        return

    for admin_id in active_admin_ids_to_notify:
        admin_user = None
        try:
            admin_user = await bot.fetch_user(admin_id)
            if admin_user:
                await admin_user.send(forward_message)
        except discord.NotFound:
            logger.warning(f"Could not find admin user with ID {admin_id} to forward logs.")
        except discord.Forbidden: # Bot might be blocked or not share a server
            logger.warning(f"Bot is blocked by or cannot DM admin user {admin_id}. Disabling logs for them.")
            admin_log_activation_status[admin_id] = False # Auto-mute for this admin
        except discord.HTTPException as e:
            if e.status == 400 and e.code == 50035: # Message too long (Invalid Form Body)
                logger.error(f"Failed to send log to admin {admin_id} due to message length or formatting. Original Message Length: {len(forward_message)}. Query: '{user_query[:50]}...', Answer: '{bot_answer_str[:50]}...'. Code: {e.code}, Status: {e.status}")
                try:
                    if admin_user: # Try sending a very short notice
                         await admin_user.send(f"Failed to send a full Q&A log due to message length. User: {original_author.name}, Query: '{user_query[:100]}...'")
                except Exception as inner_e:
                    logger.error(f"Failed to send even the short error notification to admin {admin_id}: {inner_e}")
            else:
                logger.error(f"HTTPException forwarding Q&A to admin {admin_id}: {e}")
        except Exception as e: # Catch any other exceptions
            logger.error(f"Error forwarding Q&A to admin {admin_id}: {e}")


class SuggestionButton(discord.ui.Button):
    def __init__(self, label, style, custom_id, original_query, matched_keyword_text, faq_item_index, original_faq_item_idx, helpful):
        super().__init__(label=label, style=style, custom_id=custom_id)
        self.original_query = original_query
        self.matched_keyword_text = matched_keyword_text
        self.faq_item_index = faq_item_index # This is the index in the flattened faq_questions list
        self.original_faq_item_idx = original_faq_item_idx # This is the index in the original general_faqs list
        self.helpful = helpful

    async def callback(self, interaction: discord.Interaction):
        # Log the feedback
        logger.info(f"SuggestionButton callback initiated by user {interaction.user.id} for custom_id {self.custom_id}. Helpful: {self.helpful}. Original FAQ Idx: {self.original_faq_item_idx}")

        try:
            feedback_type = "Helpful" if self.helpful else "Not Helpful"
            original_item_idx_str = str(self.original_faq_item_idx) if self.original_faq_item_idx is not None else "N/A"

            log_message = f"Suggestion Feedback: User clicked '{feedback_type}'."
            log_extra = {
                'user_id': interaction.user.id,
                'username': str(interaction.user),
                'original_query_text': self.original_query, # Query that led to suggestions
                'details': (f"Feedback given: {feedback_type}. "
                            f"For matched keyword: '{self.matched_keyword_text}'. "
                            f"FAQ Item Index (flat keyword): {self.faq_item_index}. "
                            f"Original FAQ Item Index (of answer): {original_item_idx_str}.")
            }
            logger.info(log_message, extra=log_extra)

            # Acknowledge interaction and disable buttons
            try:
                if interaction.response.is_done():
                    logger.warning(f"SuggestionButton callback: Interaction for custom_id {self.custom_id} already responded to.")
                else:
                    await interaction.response.send_message(f"Thanks for your feedback on the suggestion about '{self.matched_keyword_text}'!", ephemeral=True)
                    logger.info(f"SuggestionButton callback: Successfully sent ephemeral response for custom_id {self.custom_id}.")
            except discord.errors.InteractionResponded:
                logger.warning(f"SuggestionButton callback: InteractionResponded error for custom_id {self.custom_id}.") # Already responded
            except Exception as e_resp:
                logger.error(f"SuggestionButton callback: ERROR sending ephemeral response for custom_id {self.custom_id}. Error: {e_resp}", exc_info=True)


            if self.view:
                logger.info(f"SuggestionButton callback: View found for custom_id {self.custom_id}. Attempting to disable buttons.")
                for item in self.view.children:
                    if isinstance(item, discord.ui.Button):
                        item.disabled = True
                try:
                    if interaction.message: # Make sure message still exists
                        await interaction.message.edit(view=self.view)
                        logger.info(f"SuggestionButton callback: Successfully edited message {interaction.message.id} to disable buttons.")
                    else: 
                        logger.error(f"SuggestionButton callback: interaction.message is None for custom_id {self.custom_id}. Cannot disable buttons.")
                except discord.NotFound: # Message might have been deleted
                    logger.warning(f"SuggestionButton callback: Could not find message to edit for custom_id {self.custom_id}.")
                except discord.Forbidden: # Missing permissions
                     logger.error(f"SuggestionButton callback: Missing permissions to edit message for custom_id {self.custom_id}.")
                except Exception as e_edit:
                    logger.error(f"SuggestionButton callback: ERROR editing message to disable buttons for custom_id {self.custom_id}. Error: {e_edit}", exc_info=True)
            else:
                logger.warning(f"SuggestionButton callback: self.view is None for custom_id {self.custom_id}. Cannot disable buttons.")

        except Exception as e_outer: # Catch any other unexpected error during callback processing
            logger.error(f"SuggestionButton callback: UNHANDLED EXCEPTION in callback for custom_id {self.custom_id}. Error: {e_outer}", exc_info=True)
            try: # Try to send a generic error if no response has been sent yet
                if not interaction.response.is_done():
                    await interaction.response.send_message("Sorry, there was an error processing your feedback.", ephemeral=True)
            except discord.errors.InteractionResponded: pass # Already responded
            except Exception: logger.error(f"SuggestionButton callback: CRITICAL - Failed to even send a generic error message after outer exception.", exc_info=True)


@bot.event
async def on_ready():
    global nlp
    print(f'{bot.user.name} (ID: {bot.user.id}) has connected to Discord!')
    print(f'Listening for DMs. All interactions are handled as direct messages.')
    
    load_spacy_model() # Load spaCy model here
    load_faq_data_from_url() # FAQ data which also builds embeddings

    global admin_log_activation_status
    activated_admins_count = 0
    if ADMIN_USER_IDS:
        for admin_id in ADMIN_USER_IDS:
            admin_log_activation_status[admin_id] = True # Default to active
            activated_admins_count +=1
        logger.info(f"Default log forwarding activated for {activated_admins_count} admin(s). Admins can use 'mute logs' to opt-out.")
        print(f"Default log forwarding activated for {activated_admins_count} admin(s).")
    else:
        logger.warning("No ADMIN_USER_IDS defined. Log forwarding will not be active by default for anyone.")
        print("WARNING: No ADMIN_USER_IDS defined. Log forwarding will not be active by default for anyone.")


    activity = discord.Activity(
        name="DM for HELP!",  # Text that appears after "Playing"
        type=discord.ActivityType.custom, # Use custom for more control if needed
        state="DM for HELP!" # This is the main text for custom status
    )
    await bot.change_presence(activity=activity)
    logger.info("Bot started and ready.",
                extra={'details': 'Bot presence set, FAQ loaded, spaCy model loaded, listening for DMs.'})


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or not isinstance(message.channel, discord.DMChannel):
        return

    initial_user_message_content_for_forwarding = message.content.strip()
    user_query_lower_for_processing = message.content.lower().strip()
    original_message_content_for_processing = message.content.strip() # User's verbatim input this turn

    author_id = message.author.id
    author_name = str(message.author)
    log_extra_base = {'user_id': author_id, 'username': author_name, 'original_query_text': original_message_content_for_processing}
    
    context_was_applied_this_turn = False
    context_application_method = "none" # "rewritten", "prepended", "list_item_resolved", "none"
    faq_answer_part_text = None # Will be set by semantic search or fallback
    greeting_matched_this_interaction = False # Will be set by greeting handler


    if author_id in ADMIN_USER_IDS:
        if user_query_lower_for_processing == "activate logs":
            admin_log_activation_status[author_id] = True
            await message.channel.send("✅ Real-time Q&A log forwarding **activated** for you.")
            logger.info("Admin command: activate logs.", extra={**log_extra_base, 'details': f"Admin {author_name} activated logs."})
            return
        elif user_query_lower_for_processing == "mute logs":
            admin_log_activation_status[author_id] = False
            await message.channel.send("🔇 Real-time Q&A log forwarding **muted** for you.")
            logger.info("Admin command: mute logs.", extra={**log_extra_base, 'details': f"Admin {author_name} muted logs."})
            return

    if is_text_empty_or_punctuation_only(original_message_content_for_processing):
        logger.info("Ignoring empty or punctuation-only query.", extra={**log_extra_base, 'details': 'Query was empty or only punctuation.'})
        return

    bot_reply_parts_for_forwarding = []

    current_query_for_faq_lower = user_query_lower_for_processing
    current_original_content_for_faq = original_message_content_for_processing

    # --- Contextual Query Enhancement ---
    # This now includes the "list codes" follow-up logic
    if author_id in user_context_history and user_context_history[author_id]:
        previous_user_query_context_text, previous_bot_answer_context_obj = user_context_history[author_id][-1]
        
        cleaned_query_for_word_count = re.sub(r'[^\w\s]', '', user_query_lower_for_processing)
        query_words = cleaned_query_for_word_count.split()

        # 1. Special check for "list codes" follow-up
        is_short_query_for_list_follow_up = len(query_words) <= CONTEXT_QUERY_LENGTH_THRESHOLD + 2 # Slightly more lenient for "tell me about the first one"
        
        if is_short_query_for_list_follow_up and \
           isinstance(previous_bot_answer_context_obj, dict) and \
           previous_bot_answer_context_obj.get("type") == "code_list_sent":
            
            codes_ordered = previous_bot_answer_context_obj.get("codes_ordered")
            if codes_ordered:
                # Try to parse "first", "second", "1st", "2nd", "last" etc. from the current user query
                potential_ordinals = re.findall(r'\b(first|second|third|fourth|fifth|last|\d+)\b', user_query_lower_for_processing)
                
                target_code_name = None
                found_ordinal_str = None

                for p_ord in potential_ordinals:
                    index = parse_ordinal_or_number_to_index(p_ord)
                    if index is not None:
                        actual_index = index
                        if index == -1: # 'last'
                            if not codes_ordered: continue # Skip if list is empty
                            actual_index = len(codes_ordered) - 1
                        
                        if 0 <= actual_index < len(codes_ordered):
                            target_code_name = codes_ordered[actual_index]
                            found_ordinal_str = p_ord
                            break # Take the first valid one found
                
                if target_code_name:
                    # Rewrite the query to be a "define [code]" query
                    current_original_content_for_faq = f"define {target_code_name}"
                    current_query_for_faq_lower = current_original_content_for_faq.lower()
                    context_application_method = "list_item_resolved"
                    logger.info(
                        "Context Used (List Item Resolved).",
                        extra={
                            'user_id': author_id, 'username': author_name,
                            'original_query_text': initial_user_message_content_for_forwarding, 
                            'details': (f"Follow-up to 'list codes'. Original: '{original_message_content_for_processing}'. "
                                        f"Matched '{found_ordinal_str}' to code '{target_code_name}'. "
                                        f"New query: '{current_original_content_for_faq}'")
                        }
                    )
                    log_extra_base['original_query_text'] = current_original_content_for_faq # Log with the rewritten query
                    context_was_applied_this_turn = True
        
        # 2. General Pronoun Resolution (only if list item not resolved)
        if not context_was_applied_this_turn and nlp:
            found_pronoun_in_current_query = None
            for word_idx, word in enumerate(query_words):
                if word in CONTEXT_PRONOUNS:
                    found_pronoun_in_current_query = word
                    break
            
            if len(query_words) <= CONTEXT_QUERY_LENGTH_THRESHOLD and found_pronoun_in_current_query:
                referent = None
                if isinstance(previous_user_query_context_text, str): # Ensure it's text before passing
                    referent = get_likely_referent_from_previous_query(previous_user_query_context_text)

                if referent:
                    pronoun_pattern = r'\b' + re.escape(found_pronoun_in_current_query) + r'\b'
                    rewritten_original_content, num_replacements = re.subn(
                        pronoun_pattern, referent, original_message_content_for_processing,
                        count=1, flags=re.IGNORECASE
                    )
                    if num_replacements > 0:
                        current_original_content_for_faq = rewritten_original_content
                        current_query_for_faq_lower = rewritten_original_content.lower()
                        context_application_method = "rewritten"
                        logger.info(
                            "Context Used (Pronoun Rewritten).",
                            extra={
                                'user_id': author_id, 'username': author_name,
                                'original_query_text': initial_user_message_content_for_forwarding, 
                                'details': f"Short query: '{original_message_content_for_processing}'. Pronoun '{found_pronoun_in_current_query}' replaced with referent '{referent}' from prev_Q_text: '{previous_user_query_context_text}'. Rewritten: '{current_original_content_for_faq}'"
                            }
                        )
                    else: # Fallback to prepending if pronoun rewrite failed
                        if isinstance(previous_user_query_context_text, str):
                            current_original_content_for_faq = f"{previous_user_query_context_text} {original_message_content_for_processing}"
                            current_query_for_faq_lower = f"{previous_user_query_context_text.lower()} {user_query_lower_for_processing}"
                            context_application_method = "prepended (rewrite failed)"
                            logger.info("Context Used (Prepended - Pronoun Rewrite Failed)...", extra={'user_id':author_id, 'username':author_name, 'original_query_text': initial_user_message_content_for_forwarding, 'details': f"Failed to replace '{found_pronoun_in_current_query}' with '{referent}'. Enhanced: '{current_original_content_for_faq}'."})

                elif isinstance(previous_user_query_context_text, str): # No referent, but previous query exists - prepend
                    current_original_content_for_faq = f"{previous_user_query_context_text} {original_message_content_for_processing}"
                    current_query_for_faq_lower = f"{previous_user_query_context_text.lower()} {user_query_lower_for_processing}"
                    context_application_method = "prepended (no referent)"
                    logger.info("Context Used (Prepended - No Referent Found)...", extra={'user_id':author_id, 'username':author_name, 'original_query_text': initial_user_message_content_for_forwarding, 'details': f"Enhanced: '{current_original_content_for_faq}'."})

                if context_application_method in ["rewritten", "prepended (rewrite failed)", "prepended (no referent)"]:
                    log_extra_base['original_query_text'] = current_original_content_for_faq
                    context_was_applied_this_turn = True
            elif author_id in user_context_history and user_context_history[author_id]: # Context history exists, but conditions for application not met
                reason = []
                if not nlp: reason.append("spaCy (nlp) not loaded")
                if not (len(query_words) <= CONTEXT_QUERY_LENGTH_THRESHOLD):
                    reason.append(f"Query too long (len {len(query_words)} > {CONTEXT_QUERY_LENGTH_THRESHOLD})")
                if not found_pronoun_in_current_query and (len(query_words) <= CONTEXT_QUERY_LENGTH_THRESHOLD): # only log no pronoun if it was short enough
                    reason.append("No qualifying pronoun found in short query")
                if reason:
                    logger.info(
                        f"Context NOT Used: Conditions not met for pronoun resolution. {' & '.join(reason)}.",
                        extra=log_extra_base
                    )
    elif not nlp and author_id in user_context_history and user_context_history[author_id]: # History exists but NLP is off
         logger.warning("Context NOT Used: spaCy model (nlp) not loaded. Pronoun resolution disabled.", extra=log_extra_base)


    # --- Pronunciation Handling ---
    # (Copied from base code's on_message)
    pronounce_prefix = "!pronounce "
    pronounce_prefix_no_bang = "pronounce "
    word_to_pronounce_input = None
    # Use original_message_content_for_processing for command checking to avoid context-modified query
    if original_message_content_for_processing.lower().startswith(pronounce_prefix):
        word_to_pronounce_input = original_message_content_for_processing[len(pronounce_prefix):].strip()
    elif original_message_content_for_processing.lower().startswith(pronounce_prefix_no_bang):
        temp_word = original_message_content_for_processing[len(pronounce_prefix_no_bang):].strip()
        # Heuristic: if "pronounce" is followed by a word and query is short, assume it's a command
        if temp_word and len(original_message_content_for_processing.split()) < 5: # e.g. "pronounce word" or "pronounce word please"
             word_to_pronounce_input = temp_word

    if word_to_pronounce_input:
        response_to_user = ""
        pronunciation_view = discord.ui.View()
        if not word_to_pronounce_input: # Should be caught by above, but defensive
            response_to_user = "Please tell me what word or phrase you want to pronounce. Usage: `pronounce [word or phrase]`"
            await message.channel.send(response_to_user)
        else:
            async with message.channel.typing():
                audio_url = await get_pronunciation_audio_url(word_to_pronounce_input)
            
            encoded_word_for_google = urllib.parse.quote_plus(word_to_pronounce_input)
            google_link = f"https://www.google.com/search?q=how+to+pronounce+{encoded_word_for_google}"
            # A generic YouTube playlist link - replace if you have a specific one
            youtube_link = f"https://www.youtube.com/results?search_query=how+to+pronounce+{encoded_word_for_google}" 


            response_message_lines = [f"Pronunciation resources for \"**{word_to_pronounce_input}**\":"]
            log_audio_status = "Audio not found from API."
            if audio_url:
                pronunciation_view.add_item(discord.ui.Button(label="Play Sound", style=discord.ButtonStyle.link, url=audio_url, emoji="🔊"))
                log_audio_status = f"Audio found from API: {audio_url}"
            else:
                response_message_lines.append(f"• Sorry, I couldn't find a direct audio pronunciation for \"{word_to_pronounce_input}\" from an API.")
            
            pronunciation_view.add_item(discord.ui.Button(label="Search on Google", style=discord.ButtonStyle.link, url=google_link))
            pronunciation_view.add_item(discord.ui.Button(label="Search on YouTube", style=discord.ButtonStyle.link, url=youtube_link))
            
            if not audio_url: # Add this line only if no direct audio was found
                 response_message_lines.append(f"• You can check Google/YouTube for pronunciation resources.")
            
            response_to_user = "\n".join(response_message_lines)

            if pronunciation_view.children: # Only send view if it has items
                 await message.channel.send(response_to_user, view=pronunciation_view)
            else: # Should not happen if Google/YT links are always added
                 await message.channel.send(response_to_user)
        
        bot_reply_parts_for_forwarding.append(response_to_user + (" (View with buttons also sent)" if pronunciation_view.children else ""))
        cmd_log_extra = {'user_id': author_id, 'username': author_name, 'original_query_text': initial_user_message_content_for_forwarding}
        logger.info(f"Pronunciation request processed for '{word_to_pronounce_input}'.", extra={**cmd_log_extra, 'details': f"Audio from API: {log_audio_status}. Links provided."})
        
        # Pronunciation is a terminal command, forward and return
        await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
        return

    # --- List Codes Handling ---
    # Check original raw query (user_query_lower_for_processing), ensure not already context-rewritten to something else
    if user_query_lower_for_processing == "list codes" and not context_application_method == "list_item_resolved": 
        call_codes_data = faq_data.get("call_codes", {})
        cmd_log_extra = {'user_id': author_id, 'username': author_name, 'original_query_text': initial_user_message_content_for_forwarding}
        faq_answer_part_for_context_storage = None # For context storage specifically

        if not call_codes_data:
            reply_text = "I don't have any call codes defined at the moment."
            await message.channel.send(reply_text)
            bot_reply_parts_for_forwarding.append(reply_text)
            faq_answer_part_for_context_storage = reply_text # Store simple text for context
            logger.info("Command 'list codes' - no codes found.", extra=cmd_log_extra)
        else:
            embed = discord.Embed(title="☎️ Call Disposition Codes", color=discord.Color.blue())
            current_field_value = ""
            field_count = 0
            MAX_FIELD_VALUE_LEN = 1020 # Max length for an embed field value
            MAX_FIELDS = 24 # Max fields in an embed (Discord limit is 25, keeping one for safety/title)
            temp_forward_text_parts_for_log = ["Call Codes Sent:"]
            ordered_code_keys_for_context = list(call_codes_data.keys()) # Get ordered keys

            for code_key in ordered_code_keys_for_context: # Iterate using ordered keys
                description = call_codes_data[code_key]
                entry_text = f"**{code_key.upper()}**: {description}\n\n"
                temp_forward_text_parts_for_log.append(f"{code_key.upper()}: {description}")
                
                if len(current_field_value) + len(entry_text) > MAX_FIELD_VALUE_LEN and field_count < MAX_FIELDS:
                    embed.add_field(name=f"Codes (Part {field_count + 1})" if field_count > 0 else "Codes", value=current_field_value, inline=False)
                    current_field_value = ""
                    field_count += 1
                
                if field_count >= MAX_FIELDS:
                    logger.warning(f"List codes: Exceeded max embed fields ({MAX_FIELDS}). Truncating.")
                    await message.channel.send("The list of codes is very long and might be truncated.")
                    break
                current_field_value += entry_text
            
            if current_field_value and field_count < MAX_FIELDS: # Add any remaining text to a field
                embed.add_field(name=f"Codes (Part {field_count + 1})" if field_count > 0 or not embed.fields else "Codes", value=current_field_value, inline=False)
            
            if not embed.fields and not current_field_value: # Should not happen if codes_data is not empty
                reply_text = "No call codes formatted. Check data."
                await message.channel.send(reply_text)
                bot_reply_parts_for_forwarding.append(reply_text)
                faq_answer_part_for_context_storage = reply_text
            elif not embed.fields: # Also unlikely
                 reply_text = "Found codes, but couldn't display them. Try again/ask manager."
                 await message.channel.send(reply_text)
                 bot_reply_parts_for_forwarding.append(reply_text)
                 faq_answer_part_for_context_storage = reply_text
            else:
                await message.channel.send(embed=embed)
                bot_reply_text_for_log = "[Bot sent 'list codes' as an embed.]\n" + "\n".join(temp_forward_text_parts_for_log)
                bot_reply_parts_for_forwarding.append(bot_reply_text_for_log)
                # THIS IS WHERE WE STORE THE SPECIAL CONTEXT
                faq_answer_part_for_context_storage = {
                    "type": "code_list_sent",
                    "codes_ordered": ordered_code_keys_for_context, # Store the ordered keys
                    "display_text_for_log": bot_reply_text_for_log # For admin forwarding if needed
                }
            logger.info("Command 'list codes' processed.", extra={**cmd_log_extra, 'details': f"Displayed {len(call_codes_data) if call_codes_data else 0} codes."})
        
        # Context Storage for "list codes" itself
        if author_id not in user_context_history:
            user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
        # Store the original query "list codes" and our special answer object (or simple text if no codes)
        user_context_history[author_id].append((initial_user_message_content_for_forwarding, faq_answer_part_for_context_storage))
        
        log_details_ctx_list_q = initial_user_message_content_for_forwarding[:100]
        log_details_ctx_list_a = str(faq_answer_part_for_context_storage)[:100] if not isinstance(faq_answer_part_for_context_storage, dict) else "special code_list_sent object"
        logger.info(f"Context Stored after 'list codes' for user {author_id}.", extra={**cmd_log_extra, 'details': f"Stored Q: '{log_details_ctx_list_q}...'. Stored A: '{log_details_ctx_list_a}...'"})
        
        await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
        return # End processing for "list codes"

    # --- Define Code Handling ---
    # The `current_original_content_for_faq` might now be "define [resolved_code_name]" from context
    define_pattern = r"^(?:what is|define|explain)\s+([\w\s-]+)\??$" # \w includes numbers, \s for spaces, - for hyphens
    match_define_code = re.match(define_pattern, current_query_for_faq_lower) # Use potentially context-modified query
    
    if match_define_code:
        # If context resolved it, current_original_content_for_faq is already "define ACTUAL_CODE_NAME"
        # If it's a fresh "define query", code_name_query_original_case is from user's input.
        
        # Get the code name part from `current_original_content_for_faq` (which could be context-rewritten or original)
        # Example: current_original_content_for_faq = "define WRONG NUMBER" -> we need "WRONG NUMBER"
        # Example: current_original_content_for_faq = "what is WRONG NUMBER" -> we need "WRONG NUMBER"
        code_name_from_query_to_lookup = match_define_code.group(1).strip() # This is from current_query_for_faq_lower

        code_name_query_upper = code_name_from_query_to_lookup.upper() # Use the (potentially resolved) name
        call_codes_data = faq_data.get("call_codes", {})
        found_code_key = next((k for k in call_codes_data if k.upper() == code_name_query_upper), None)
        
        defined_code_embed = None
        temp_bot_reply_for_define = None # Specific reply for define command
        
        # Use initial_user_message_content_for_forwarding for logging original intent
        # but current_original_content_for_faq for what was processed.
        cmd_log_extra_define = {'user_id': author_id, 'username': author_name, 'original_query_text': initial_user_message_content_for_forwarding}

        if found_code_key:
            defined_code_embed = discord.Embed(title=f"Definition: {found_code_key.upper()}", description=call_codes_data[found_code_key], color=discord.Color.purple())
            temp_bot_reply_for_define = f"Definition: {found_code_key.upper()}\n{call_codes_data[found_code_key]}"
            log_detail_msg = f"Defined code '{found_code_key.upper()}' (exact match - from query '{current_original_content_for_faq}')."
            if context_was_applied_this_turn : log_detail_msg += f" Context method: {context_application_method}."
            logger.info(log_detail_msg, extra=cmd_log_extra_define)
        elif not context_was_applied_this_turn: # Only do fuzzy if not context-rewritten to a specific define
            # `code_name_from_query_to_lookup` here would be from the user's raw "define xyz"
            # We should use the original casing for fuzzy matching if available, or `code_name_from_query_to_lookup`
            original_define_term_for_fuzzy = code_name_from_query_to_lookup # Default to what was parsed
            
            # If 'define' was original command, original_message_content_for_processing contains the full "define xyz"
            # We need to extract "xyz" from it with original casing for best fuzzy match
            original_define_match_for_casing = re.match(define_pattern, original_message_content_for_processing, re.IGNORECASE)
            if original_define_match_for_casing:
                original_define_term_for_fuzzy = original_define_match_for_casing.group(1).strip()

            best_match_code, score = process.extractOne(original_define_term_for_fuzzy, list(call_codes_data.keys()), scorer=fuzz.token_set_ratio)
            if score > 80 and best_match_code: # Threshold for fuzzy define
                defined_code_embed = discord.Embed(title=f"Definition (for '{original_define_term_for_fuzzy}'): {best_match_code.upper()}", description=call_codes_data[best_match_code], color=discord.Color.purple())
                temp_bot_reply_for_define = f"Definition (for '{original_define_term_for_fuzzy}'): {best_match_code.upper()}\n{call_codes_data[best_match_code]}"
                logger.info(f"Defined code '{best_match_code.upper()}' (fuzzy match on '{original_define_term_for_fuzzy}').", extra={**cmd_log_extra_define, 'details': f"Score: {score}."})
        
        if defined_code_embed:
            await message.channel.send(embed=defined_code_embed)
            if temp_bot_reply_for_define: bot_reply_parts_for_forwarding.append(temp_bot_reply_for_define)
            
            # Context Storage for "define code"
            if author_id not in user_context_history:
                user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
            # Store current_original_content_for_faq (which could be "define XYZ" from context or user)
            # And the bot's reply.
            user_context_history[author_id].append((current_original_content_for_faq, temp_bot_reply_for_define))
            log_details_ctx_define_q = current_original_content_for_faq[:100]
            log_details_ctx_define_a = temp_bot_reply_for_define[:100] if temp_bot_reply_for_define else "N/A"
            logger.info(f"Context Stored after 'define code' for user {author_id}.", extra={**cmd_log_extra_define, 'details': f"Stored Q: '{log_details_ctx_define_q}...'. Stored A: '{log_details_ctx_define_a}...'"})

            await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
            return
        # If "define" command doesn't find a code (and wasn't a context rewrite to a valid define that failed),
        # let it fall through to general semantic search with the *potentially context-enhanced* query
        # (current_original_content_for_faq).
        # If it *was* a context rewrite (list_item_resolved) and it didn't find a code, that's an issue.
        # The current logic will let it fall through. This might be okay, semantic search might find something related.
        # Consider if "define X" from context should have a specific "X not found" message if it fails.
        # For now, falling through is the behavior.

    # --- Greeting Handling ---
    # (Copied from base code's on_message, adapted to use current_query_for_faq_lower and current_original_content_for_faq)
    greetings_data = faq_data.get("greetings_and_pleasantries", [])
    # greeting_matched_this_interaction = False # Already initialized above
    best_greeting_match_entry = None
    len_of_matched_greeting_keyword = 0
    matched_greeting_keyword_original_casing = None

    greeting_log_extra = {**log_extra_base} # Use a copy, log_extra_base might be updated if greeting is stripped

    for greeting_entry in greetings_data:
        keywords = greeting_entry.get("keywords", [])
        if not keywords: continue
        for kw in keywords:
            kw_lower = kw.lower()
            # Check if the current_query_for_faq_lower starts with a greeting keyword
            if current_query_for_faq_lower.startswith(kw_lower):
                if len(kw_lower) > len_of_matched_greeting_keyword: # Find the longest matching keyword
                    len_of_matched_greeting_keyword = len(kw_lower)
                    best_greeting_match_entry = greeting_entry
                    # Get the original casing from current_original_content_for_faq
                    matched_greeting_keyword_original_casing = current_original_content_for_faq[:len(kw_lower)]

    if best_greeting_match_entry:
        greeting_matched_this_interaction = True
        response_type = best_greeting_match_entry.get("response_type")
        greeting_reply_text = f"Hello {message.author.mention}!" # Default reply

        if response_type == "standard_greeting":
            reply_template = best_greeting_match_entry.get("greeting_reply_template", "Hello there, {user_mention}!")
            # Use the matched keyword with its original casing if available
            actual_greeting_cased = matched_greeting_keyword_original_casing.capitalize() if matched_greeting_keyword_original_casing else "Hello"
            greeting_reply_text = reply_template.format(actual_greeting_cased=actual_greeting_cased, user_mention=message.author.mention)
        elif response_type == "specific_reply":
            greeting_reply_text = best_greeting_match_entry.get("reply_text", "I acknowledge that.")

        await message.channel.send(greeting_reply_text)
        bot_reply_parts_for_forwarding.append(greeting_reply_text)

        # Remove the matched greeting from the query to process the remainder
        remainder_after_greeting = current_original_content_for_faq[len_of_matched_greeting_keyword:].strip()
        # Also remove leading punctuation from remainder if any
        if remainder_after_greeting and remainder_after_greeting[0] in string.punctuation:
            remainder_after_greeting = remainder_after_greeting[1:].strip()
        
        logger.info(f"Greeting matched (prefix).", extra={**greeting_log_extra, 'details': f"Matched: '{matched_greeting_keyword_original_casing}'. Remainder for FAQ: '{remainder_after_greeting}'."})

        if not is_text_empty_or_punctuation_only(remainder_after_greeting) and len(remainder_after_greeting) > 3: # Heuristic for meaningful remainder
            current_query_for_faq_lower = remainder_after_greeting.lower()
            current_original_content_for_faq = remainder_after_greeting 
            # Update log_extra_base if query is modified for semantic search
            log_extra_base['original_query_text'] = current_original_content_for_faq
            logger.info(f"Processing remainder of query after greeting: '{current_original_content_for_faq}'", extra=log_extra_base)
        else: # Only greeting, or remainder is too short/empty
            # Store context for the greeting itself if it's terminal
            if author_id not in user_context_history:
                user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
            # Store the part of the query that was matched as greeting, and its reply
            user_context_history[author_id].append((matched_greeting_keyword_original_casing if matched_greeting_keyword_original_casing else current_original_content_for_faq, greeting_reply_text))
            logger.info(f"Context Stored after terminal greeting.", extra={**greeting_log_extra, 'details': f"Stored Q: '{matched_greeting_keyword_original_casing}'. Stored A: '{greeting_reply_text[:100]}...'"})

            await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
            return # Greeting was handled, and no significant remainder
            
    elif len(current_query_for_faq_lower) < 30: # Only do fuzzy match for short queries if no prefix match
        for greeting_entry in greetings_data:
            keywords = greeting_entry.get("keywords", [])
            if not keywords: continue
            # Fuzzy match the whole current_query_for_faq_lower against greeting keywords
            match_result = process.extractOne(current_query_for_faq_lower, keywords, scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_MATCH_THRESHOLD_GREETINGS)
            if match_result:
                matched_keyword_from_fuzz, score = match_result
                greeting_matched_this_interaction = True
                response_type = greeting_entry.get("response_type")
                reply_text = f"Hello {message.author.mention}!" # Default
                if response_type == "standard_greeting":
                    reply_template = greeting_entry.get("greeting_reply_template", "Hello there, {user_mention}!")
                    # Find original casing of the fuzzy matched keyword for a nicer reply
                    actual_greeting_cased = next((kw for kw in keywords if kw.lower() == matched_keyword_from_fuzz.lower()), matched_keyword_from_fuzz)
                    reply_text = reply_template.format(actual_greeting_cased=actual_greeting_cased.capitalize(), user_mention=message.author.mention)
                elif response_type == "specific_reply":
                    reply_text = greeting_entry.get("reply_text", "I acknowledge that.")
                
                await message.channel.send(reply_text)
                bot_reply_parts_for_forwarding.append(reply_text)
                logger.info(f"Greeting matched (fuzzy, short query).", extra={**greeting_log_extra, 'details': f"Matched '{current_query_for_faq_lower}' with '{matched_keyword_from_fuzz}', Score: {score}."})
                
                # Store context for the fuzzy greeting if it's terminal
                if author_id not in user_context_history:
                    user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
                user_context_history[author_id].append((current_original_content_for_faq, reply_text)) # Store the full query that matched
                logger.info(f"Context Stored after terminal fuzzy greeting.", extra={**greeting_log_extra, 'details': f"Stored Q: '{current_original_content_for_faq}'. Stored A: '{reply_text[:100]}...'"})

                await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
                return # Fuzzy greeting matched the whole (short) query

    # --- Semantic Search and Dynamic Threshold Logic ---
    # (Copied from base code's on_message, uses current_query_for_faq_lower)
    # log_extra_base['original_query_text'] now reflects the query being sent to semantic search
    # (original, context-enhanced/rewritten, or remainder after greeting)
    faq_items_original_list = faq_data.get("general_faqs", [])
    top_indices, top_scores = semantic_search_top_n_matches(current_query_for_faq_lower, n=CANDIDATES_TO_FETCH_FOR_SUGGESTIONS)

    # faq_answer_part_text = None # Initialized at the top of on_message
    action = "fallback" # Default action
    dynamic_decision_details = "N/A"

    if context_was_applied_this_turn: # Log semantic search details if context was used
        top_matches_log_details = []
        if top_indices:
            for i_log in range(min(len(top_indices), 3)): # Log top 3 for brevity
                idx_log = top_indices[i_log]
                score_log = top_scores[i_log]
                if 0 <= idx_log < len(faq_original_indices) and 0 <= idx_log < len(faq_questions): # Check bounds
                    original_faq_idx_log = faq_original_indices[idx_log]
                    keyword_text_log = faq_questions[idx_log]
                    top_matches_log_details.append(f"(Score: {score_log:.3f}, OrigFAQIdx: {original_faq_idx_log}, MatchedKW: '{keyword_text_log[:60]}...')")
                else:
                    top_matches_log_details.append(f"(Score: {score_log:.3f}, FlatIdx: {idx_log} - out of bounds)")
        
        logger.info(
            f"Semantic search on CONTEXT-APPLIED query ({context_application_method}). Top matches: {'; '.join(top_matches_log_details) if top_matches_log_details else 'No semantic hits.'}",
            extra={**log_extra_base, 'details': f"Applied query: '{current_original_content_for_faq}'. Raw short query (if applicable): '{original_message_content_for_processing if context_application_method != 'none' else 'N/A'}'."}
        )


    if not top_indices: # No semantic hits at all
        if not greeting_matched_this_interaction: # Only send fallback if no greeting was sent
            faq_answer_part_text = faq_data.get("fallback_message", "I'm sorry, I couldn't find an answer right now.")
            await message.channel.send(faq_answer_part_text)
            logger.info("Unanswered query, fallback sent (no semantic hits).", extra=log_extra_base)
    else:
        primary_score = top_scores[0]
        num_candidates = len(top_scores)

        if DYNAMIC_THRESHOLD_ENABLED:
            secondary_score = top_scores[1] if num_candidates > 1 else 0.0
            gap = primary_score - secondary_score

            if primary_score >= ABSOLUTE_HIGH_SCORE_OVERRIDE:
                action = "direct_answer"
                dynamic_decision_details = f"Dynamic: Direct (High Score Override). P={primary_score:.2f} >= {ABSOLUTE_HIGH_SCORE_OVERRIDE}."
            elif primary_score >= SEMANTIC_SEARCH_THRESHOLD:
                is_dynamically_confirmed_direct = (num_candidates == 1 or gap >= CONFIDENCE_GAP_FOR_DIRECT_ANSWER)
                is_ambiguous_cluster_for_suggestions = (num_candidates > 1 and
                                                       gap < SIMILAR_SCORE_CLUSTER_THRESHOLD and # Small gap
                                                       secondary_score >= SUGGESTION_THRESHOLD) # Secondary is decent
                if is_dynamically_confirmed_direct:
                    action = "direct_answer"
                    dynamic_decision_details = f"Dynamic: Direct. P={primary_score:.2f}, S={secondary_score:.2f}. Gap {gap:.2f} >= {CONFIDENCE_GAP_FOR_DIRECT_ANSWER} or num_cand=1."
                elif is_ambiguous_cluster_for_suggestions:
                    action = "suggestions"
                    dynamic_decision_details = f"Dynamic: Suggestions (ambiguous cluster). P={primary_score:.2f}, S={secondary_score:.2f}. Gap {gap:.2f} < {SIMILAR_SCORE_CLUSTER_THRESHOLD} & S >= {SUGGESTION_THRESHOLD}."
                else: # Above semantic threshold, but not a clear direct answer or strong cluster for suggestions
                    # Default to suggestions if primary is good enough, or direct if no good secondary
                    if primary_score >= SUGGESTION_THRESHOLD : # and (num_candidates == 1 or secondary_score < SUGGESTION_THRESHOLD):
                         action = "direct_answer" # Treat as direct if only one candidate or others are too weak
                         dynamic_decision_details = f"Dynamic: Direct (P >= SemThresh, not cluster, weak/no secondary). P={primary_score:.2f}, S={secondary_score:.2f}."
                    else: # Should be rare if P >= SEMANTIC_SEARCH_THRESHOLD
                         action = "suggestions"
                         dynamic_decision_details = f"Dynamic: Suggestions (P >= SemThresh but not dyn_direct/cluster, fallback to sugg). P={primary_score:.2f}, S={secondary_score:.2f}. Gap {gap:.2f}."

            elif primary_score >= SUGGESTION_THRESHOLD: # Primary score is between suggestion and semantic thresholds
                action = "suggestions"
                dynamic_decision_details = f"Dynamic: Suggestions (P < SemThresh but P >= SugThresh). P={primary_score:.2f}."
            # else: action remains "fallback"
            if action == "fallback" and dynamic_decision_details == "N/A": # If no other rule set it
                 dynamic_decision_details = f"Dynamic: Fallback. P={primary_score:.2f} < {SUGGESTION_THRESHOLD} (or other rule miss)."
        else: # Static Threshold Logic (DYNAMIC_THRESHOLD_ENABLED is False)
            dynamic_decision_details = "Static Thresholds Used."
            if primary_score >= SEMANTIC_SEARCH_THRESHOLD:
                action = "direct_answer"
            elif primary_score >= SUGGESTION_THRESHOLD:
                action = "suggestions"
            # else: action remains "fallback"
        
        # --- Execute Action (Direct Answer, Suggestions, Fallback) ---
        if action == "direct_answer":
            primary_faq_flat_idx = top_indices[0] # Index in faq_questions/faq_original_indices
            # Ensure indices are valid before accessing lists
            if not (0 <= primary_faq_flat_idx < len(faq_questions) and 0 <= primary_faq_flat_idx < len(faq_original_indices)):
                logger.error(f"Semantic primary match index {primary_faq_flat_idx} out of bounds for faq_questions/faq_original_indices. {dynamic_decision_details}", extra=log_extra_base)
                faq_answer_part_text = faq_data.get("fallback_message", "Sorry, an error occurred processing your request.")
                await message.channel.send(faq_answer_part_text)
            else:
                original_faq_item_idx_primary = faq_original_indices[primary_faq_flat_idx]
                if not (0 <= original_faq_item_idx_primary < len(faq_items_original_list)):
                    logger.error(f"Semantic primary original item index {original_faq_item_idx_primary} out of bounds for faq_items_original_list. {dynamic_decision_details}", extra=log_extra_base)
                    faq_answer_part_text = faq_data.get("fallback_message", "Sorry, an error occurred retrieving the answer.")
                    await message.channel.send(faq_answer_part_text)
                else:
                    matched_item_primary = faq_items_original_list[original_faq_item_idx_primary]
                    faq_answer_part_text = matched_item_primary.get("answer", "Answer not available for this topic.")
                    await send_long_message(message.channel, faq_answer_part_text, view=None) # No buttons for direct answer
                    logger.info("Semantic FAQ Direct Match.", extra={**log_extra_base, 'details': f"Score: {primary_score:.2f}. Matched Keyword Flat Idx: '{primary_faq_flat_idx}', Original FAQ Idx: {original_faq_item_idx_primary}. Keyword: '{faq_questions[primary_faq_flat_idx]}'. Decision: {dynamic_decision_details}"})

        elif action == "suggestions":
            suggestions_to_display = []
            shown_original_faq_indices = set() # To avoid showing same answer for different keywords

            for i in range(len(top_indices)): # Iterate through all candidates fetched
                current_flat_idx = top_indices[i]
                current_score = top_scores[i]

                if current_score < SUGGESTION_THRESHOLD: break # Stop if score too low

                if not (0 <= current_flat_idx < len(faq_original_indices) and 0 <= current_flat_idx < len(faq_questions)):
                    logger.warning(f"Suggestion selection: flat_idx {current_flat_idx} out of bounds. Decision: {dynamic_decision_details}")
                    continue
                
                current_original_faq_idx = faq_original_indices[current_flat_idx]
                if current_original_faq_idx not in shown_original_faq_indices:
                    if not (0 <= current_original_faq_idx < len(faq_items_original_list)):
                        logger.warning(f"Suggestion selection: original_faq_idx {current_original_faq_idx} out of bounds. Decision: {dynamic_decision_details}")
                        continue
                    
                    matched_item = faq_items_original_list[current_original_faq_idx]
                    answer_text = matched_item.get("answer", "Answer not available.")
                    matched_keyword_text = faq_questions[current_flat_idx] # Get the actual keyword text
                    if matched_keyword_text: matched_keyword_text = matched_keyword_text[0].upper() + matched_keyword_text[1:] # Capitalize

                    suggestions_to_display.append({
                        "score": current_score, "flat_idx": current_flat_idx, "original_idx": current_original_faq_idx,
                        "answer": answer_text, "keyword": matched_keyword_text
                    })
                    shown_original_faq_indices.add(current_original_faq_idx)
                
                if len(suggestions_to_display) >= MAX_SUGGESTIONS_TO_SHOW: break
            
            if suggestions_to_display:
                user_verbatim_input_this_turn = initial_user_message_content_for_forwarding # Use initial for this message
                intro_message_text = f"I'm not sure I have an exact answer for '{user_verbatim_input_this_turn}', but perhaps one of these is helpful?\n"
                
                await send_long_message(message.channel, intro_message_text) # Send intro first
                temp_suggestion_forward_texts = [intro_message_text.strip()] # For logging
                logger_details_base_sugg = f"Decision: {dynamic_decision_details}. Primary Score: {primary_score:.2f}."

                for i, sugg_data in enumerate(suggestions_to_display):
                    embed_title = f"🤔 Suggestion {i+1}: Related to '{sugg_data['keyword']}'"
                    if len(embed_title) > 256: embed_title = embed_title[:253] + "..." # Embed title limit
                    
                    embed = discord.Embed(title=embed_title, description=sugg_data['answer'], color=discord.Color.gold())
                    
                    # original_query should be what was actually used for semantic search (current_original_content_for_faq)
                    btn_yes = SuggestionButton(label="✅ Helpful!", style=discord.ButtonStyle.success, custom_id=f"sugg_yes_{message.id}_{sugg_data['flat_idx']}_{sugg_data['original_idx']}", original_query=current_original_content_for_faq, matched_keyword_text=sugg_data['keyword'], faq_item_index=sugg_data['flat_idx'], original_faq_item_idx=sugg_data['original_idx'], helpful=True)
                    btn_no = SuggestionButton(label="❌ Not quite", style=discord.ButtonStyle.danger, custom_id=f"sugg_no_{message.id}_{sugg_data['flat_idx']}_{sugg_data['original_idx']}", original_query=current_original_content_for_faq, matched_keyword_text=sugg_data['keyword'], faq_item_index=sugg_data['flat_idx'], original_faq_item_idx=sugg_data['original_idx'], helpful=False)
                    sugg_view = discord.ui.View(timeout=300); sugg_view.add_item(btn_yes); sugg_view.add_item(btn_no)
                    
                    await message.channel.send(embed=embed, view=sugg_view)
                    temp_suggestion_forward_texts.append(f"Suggestion {i+1} (Score {sugg_data['score']:.2f}, Orig. FAQ Idx: {sugg_data['original_idx']}) '{sugg_data['keyword']}':\n{sugg_data['answer']}")
                    logger.info(f"Semantic Suggestion {i+1} offered.", extra={**log_extra_base, 'details': f"{logger_details_base_sugg} Sugg Score: {sugg_data['score']:.2f}, Keyword: '{sugg_data['keyword']}', Orig. FAQ Idx: {sugg_data['original_idx']}."})
                
                faq_answer_part_text = "\n---\n".join(temp_suggestion_forward_texts) # For context storage and logging
            else: # No suggestions met threshold or were distinct enough
                if not greeting_matched_this_interaction:
                    faq_answer_part_text = faq_data.get("fallback_message", "I'm sorry, I couldn't find a clear answer for that.")
                    await message.channel.send(faq_answer_part_text)
                    logger.info("Unanswered query, fallback sent (no distinct suggestions met threshold).", extra={**log_extra_base, 'details': dynamic_decision_details})
        
        elif action == "fallback": # Semantic search ran, but scores were too low or rules led to fallback
            if not greeting_matched_this_interaction:
                faq_answer_part_text = faq_data.get("fallback_message", "I'm sorry, I couldn't find an answer for that.")
                await message.channel.send(faq_answer_part_text)
                logger.info("Unanswered query, fallback sent (low semantic score or dynamic rules).", extra={**log_extra_base, 'details': dynamic_decision_details})


    # --- Context Storage (General) ---
    # This will store the result of semantic search or fallback.
    # If context was applied (list_item_resolved, rewritten, prepended), current_original_content_for_faq is the modified query.
    # faq_answer_part_text is the bot's response to *that* query (direct answer, suggestions text, or fallback text).
    if faq_answer_part_text: # faq_answer_part_text is the outcome of semantic search / suggestions / fallback
        if author_id not in user_context_history:
            user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
        
        user_context_history[author_id].append((current_original_content_for_faq, faq_answer_part_text))
        
        context_storage_log_extra = {'user_id': author_id, 'username': author_name, 
                                     'original_query_text': initial_user_message_content_for_forwarding} 
        
        stored_q_part_for_log = current_original_content_for_faq[:100]
        stored_a_part_for_log_display = ""
        if isinstance(faq_answer_part_text, dict) and "display_text_for_log" in faq_answer_part_text: # Should not happen here, faq_answer_part_text is string now
            stored_a_part_for_log_display = faq_answer_part_text["display_text_for_log"][:100] + "..."
        elif isinstance(faq_answer_part_text, str):
            stored_a_part_for_log_display = faq_answer_part_text[:100] + "..."
        else: 
            stored_a_part_for_log_display = str(faq_answer_part_text)[:100] + "..."

        logger.info(
            f"Context Stored (General): For user {author_id}.",
            extra={
                **context_storage_log_extra,
                'details': f"Stored Q-part (processed query): '{stored_q_part_for_log}...'. Stored A-part: '{stored_a_part_for_log_display}'"
            }
        )

    # --- Consolidate bot replies for forwarding ---
    # `bot_reply_parts_for_forwarding` might contain greeting if one was sent.
    # `faq_answer_part_text` contains the semantic result (answer, suggestions summary, or fallback).
    # We need to combine these intelligently for the admin log.
    
    # If a greeting was sent, it's already in bot_reply_parts_for_forwarding.
    # If a semantic answer/suggestions/fallback was also determined, faq_answer_part_text will have it.
    
    final_bot_reply_for_forwarding = ""
    
    # Capture greeting part if it exists in bot_reply_parts_for_forwarding (it would be the first)
    greeting_text_from_parts = ""
    if greeting_matched_this_interaction and bot_reply_parts_for_forwarding:
        greeting_text_from_parts = bot_reply_parts_for_forwarding[0]

    # `faq_answer_part_text` is the string from semantic search (answer, suggestions, or fallback)
    # This is what was shown to the user for the "main query" part.
    semantic_or_fallback_text_for_log = ""
    if isinstance(faq_answer_part_text, str): # Ensure it's a string
        semantic_or_fallback_text_for_log = faq_answer_part_text
    
    if greeting_text_from_parts and semantic_or_fallback_text_for_log:
        # Check if the greeting text is the *same* as the semantic text.
        # This can happen if e.g. "thanks" -> "you're welcome", and "you're welcome" is also in semantic results.
        # More likely, if the greeting was terminal and its reply was stored as faq_answer_part_text (though this shouldn't happen with current flow).
        # The main case is a greeting *followed by* a semantic answer.
        if greeting_text_from_parts.strip() == semantic_or_fallback_text_for_log.strip():
            final_bot_reply_for_forwarding = greeting_text_from_parts
        else:
            # current_original_content_for_faq is the part of the query that went to semantic search
            final_bot_reply_for_forwarding = (
                f"{greeting_text_from_parts}\n"
                f"THEN, for the query part ('{current_original_content_for_faq}'):\n" 
                f"{semantic_or_fallback_text_for_log}"
            )
    elif greeting_text_from_parts: # Only a greeting was processed and sent (and it was terminal)
        final_bot_reply_for_forwarding = greeting_text_from_parts
    elif semantic_or_fallback_text_for_log: # Only a semantic/fallback answer was sent
        final_bot_reply_for_forwarding = semantic_or_fallback_text_for_log
    # If neither, final_bot_reply_for_forwarding remains empty. This can happen if a command like admin logs was processed.
    # Those commands handle their own forwarding or don't need it.

    if final_bot_reply_for_forwarding:
        await forward_qa_to_admins(initial_user_message_content_for_forwarding, final_bot_reply_for_forwarding, message.author)
            
    # Fallback for direct commands (pronounce, list codes, define code) that have their own return paths
    # and call forward_qa_to_admins directly. This final forwarding is for the main Q&A flow.


# --- Run the Bot ---
if __name__ == "__main__":
    # Ensure ADMIN_USER_IDS are integers
    actual_admin_ids = {uid for uid in ADMIN_USER_IDS if isinstance(uid, int)}
    if len(actual_admin_ids) < len(ADMIN_USER_IDS):
        print("INFO: Placeholder or non-integer admin IDs were filtered out. Ensure all admin IDs are correct integers.")
        logger.info("Non-integer admin IDs filtered out.", extra={'details': f"Original: {ADMIN_USER_IDS}, Filtered: {actual_admin_ids}"})
    ADMIN_USER_IDS = actual_admin_ids

    if not ADMIN_USER_IDS: # Check after filtering
        print("WARNING: ADMIN_USER_IDS set is empty after filtering. Log forwarding will not work for any user by default or by command.")
        logger.warning("ADMIN_USER_IDS is empty after filtering. Log forwarding disabled.")

    if TOKEN:
        try:
            bot.run(TOKEN)
        except discord.PrivilegedIntentsRequired:
            print("Error: Privileged Intents (Message Content) are not enabled for the bot in the Discord Developer Portal.")
            logger.critical("Bot failed to run: Privileged Intents Required.", extra={'details': 'Privileged Intents Error'})
        except discord.LoginFailure:
            print("Error: Improper token has been passed. Bot cannot log in.")
            logger.critical("Bot failed to run: Login Failure (Improper Token).", extra={'details': 'Login Failure Error'})
        except Exception as e:
            print(f"An error occurred while trying to run the bot: {e}")
            logger.exception("Bot failed to run due to an unhandled exception.", extra={'details': str(e)})
    else:
        print("Error: DISCORD_TOKEN not found in .env file or environment variables. Bot cannot start.")
        logger.critical("DISCORD_TOKEN not found. Bot cannot start.", extra={'details': 'Token Missing'})