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
SEMANTIC_SEARCH_THRESHOLD = 0.58
SUGGESTION_THRESHOLD = 0.50
MAX_SUGGESTIONS_TO_SHOW = 2
CANDIDATES_TO_FETCH_FOR_SUGGESTIONS = 5

MAX_QUERY_LENGTH = 14

DYNAMIC_THRESHOLD_ENABLED = True
CONFIDENCE_GAP_FOR_DIRECT_ANSWER = 0.06
SIMILAR_SCORE_CLUSTER_THRESHOLD = 0.04
ABSOLUTE_HIGH_SCORE_OVERRIDE = 0.65

CONTEXT_WINDOW_SIZE = 3
CONTEXT_QUERY_LENGTH_THRESHOLD = 11 # Max words for a query to be considered "short" for context
CONTEXT_PRONOUNS = {
    "it", "that", "this", "those", "them", "they", "he", "she", "him", "her", "its", # Subject/Object
    "one" # Can sometimes be used contextually e.g. "how about that one?"
}
interrogative_words = {"what", "who", "which", "whose", "where", "when", "why", "how"}
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
    1342311589298311230,
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
            nlp = None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading spaCy model: {e}")
            nlp = None

def get_likely_referent_from_previous_query(text: str) -> str | None:
    if not nlp or not text:
        return None
    doc = nlp(text)

    candidate_chunks = []

    # 1. Analyze Noun Chunks
    for chunk in doc.noun_chunks:
        root = chunk.root
        chunk_text = chunk.text.strip()

        # Basic filtering:
        # a. Skip if the chunk's root is an interrogative word acting as such.
        if root.lower_ in interrogative_words and root.dep_ in ["nsubj", "attr", "dobj", "advmod"]: # advmod for 'how'
            # Check if it's truly just the interrogative word or a phrase starting with it
            if chunk_text.lower().split()[0] == root.lower_ and len(chunk_text.split()) < 3: # e.g. "what", "what is it"
                 logger.debug(f"Pronoun Resolution: Skipping interrogative-rooted chunk '{chunk_text}'")
                 continue

        # b. Skip if the chunk is essentially just a pronoun we want to replace (it, that, this)
        #    unless it's part of a more descriptive phrase (e.g., "that specific policy").
        if root.pos_ == "PRON" and chunk_text.lower() in CONTEXT_PRONOUNS:
            is_descriptive_pronoun_phrase = False
            for token_in_chunk in chunk:
                # If there's an adjective or another noun in the chunk, it's more descriptive
                if token_in_chunk.pos_ in ["ADJ", "NOUN"] and token_in_chunk.lower_ not in CONTEXT_PRONOUNS:
                    is_descriptive_pronoun_phrase = True
                    break
            if not is_descriptive_pronoun_phrase:
                logger.debug(f"Pronoun Resolution: Skipping simple pronoun chunk '{chunk_text}' as referent candidate.")
                continue

        priority = 0
        # Assign priority based on root POS
        if root.pos_ == "PROPN":
            priority += 50
        elif root.pos_ == "NOUN":
            priority += 20

        # Boost priority based on dependency role
        if root.dep_ in ["nsubj", "nsubjpass"]:  # Subject
            priority += 30
        elif root.dep_ == "dobj":  # Direct Object
            priority += 20
        elif root.dep_ in ["attr", "appos"]: # Attribute or Apposition
            priority += 15
        elif root.dep_ == "pobj":  # Object of preposition
            priority += 10

        # Small bonus for length (longer chunks can sometimes be more specific)
        priority += len(chunk_text.split())

        # Penalize if the root is still a pronoun, even if part of a phrase
        if root.pos_ == "PRON":
            priority -= 25


        if priority > 0: # Only consider chunks with some positive indication
             # Store chunk along with its start position for tie-breaking (prefer later mentioned)
            candidate_chunks.append({"text": chunk_text, "priority": priority, "start_char": chunk.start_char})

    if candidate_chunks:
        # Sort by priority (desc), then by start_char (desc - for recency)
        sorted_candidates = sorted(candidate_chunks, key=lambda x: (x["priority"], x["start_char"]), reverse=True)
        best_chunk = sorted_candidates[0]["text"]
        logger.debug(f"Pronoun Resolution: Best Noun Chunk: '{best_chunk}' (Priority: {sorted_candidates[0]['priority']}) from: {sorted_candidates}")
        return best_chunk

    # 2. Fallback: If no good noun chunks, look for individual NOUN/PROPN tokens (similar to old logic but more careful)
    # This part is a safety net.
    potential_single_tokens = []
    for token in reversed(doc): # Iterate from end for recency
        token_text = token.text.strip()
        # Avoid interrogatives and simple pronouns
        if token.pos_ in ["NOUN", "PROPN"] and \
           token.lower_ not in CONTEXT_PRONOUNS and \
           token.lower_ not in interrogative_words:

            priority = 0
            if token.pos_ == "PROPN": priority = 5
            elif token.dep_ in ["nsubj", "nsubjpass"]: priority = 4
            elif token.dep_ == "dobj": priority = 3
            elif token.dep_ in ["attr", "appos", "pobj"]: priority = 2
            elif token.pos_ == "NOUN": priority = 1

            if priority > 0:
                potential_single_tokens.append({"text": token_text, "priority": priority})

    if potential_single_tokens:
        # In this fallback, simple recency might be enough, or pick highest priority among recent.
        # For simplicity, let's just take the first one found (most recent with any priority).
        # A more complex sort could be added if needed.
        best_single_token = potential_single_tokens[0]["text"] # Already reversed, so potential_single_tokens[0] is last
        logger.debug(f"Pronoun Resolution (Fallback to Single Token): Using '{best_single_token}' from: {potential_single_tokens}")
        return best_single_token

    logger.debug(f"Pronoun Resolution: No suitable referent found in '{text}'.")
    return None

def load_faq_data_from_url():
    global faq_data
    default_faq_structure = {
        "greetings_and_pleasantries": [],
        "general_faqs": [],
        "call_codes": {}, # Expects {"CODE_NAME": {"keywords": [], "answer": "description"}}
        "fallback_message": "I couldn't find that. Try asking general questions (e.g., 'what is wrap up time?' then.. 'how can i reduce that?'), typing 'list codes' (then ask about one, like 'the first one'), or typing 'pronounce [word]' for resources. For name pronunciation use quotes 'John Smith'."
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
    faq_data.setdefault("call_codes", {}) # This will be the nested structure
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

        # Embed general FAQs
        for original_idx, item in enumerate(general_faqs_list):
            if isinstance(item, dict) and item.get("keywords"):
                keywords_data = item["keywords"]
                if isinstance(keywords_data, list):
                    for keyword_entry in keywords_data:
                        if isinstance(keyword_entry, str) and keyword_entry.strip():
                            current_flat_faq_questions.append(keyword_entry.strip().lower())
                            current_flat_faq_original_indices.append(original_idx) # Store index to general_faqs list
                elif isinstance(keywords_data, str) and keywords_data.strip():
                    current_flat_faq_questions.append(keywords_data.strip().lower())
                    current_flat_faq_original_indices.append(original_idx)

        # Embed call_codes if they have keywords (for semantic searchability of codes)
        call_codes_dict = faq_data.get("call_codes", {})
        if isinstance(call_codes_dict, dict):
            for code_name, code_obj in call_codes_dict.items():
                if isinstance(code_obj, dict) and "keywords" in code_obj and "answer" in code_obj:
                    code_keywords = code_obj["keywords"]
                    code_original_idx_identifier = f"code_{code_name.upper().replace(' ', '_')}"
                    if isinstance(code_keywords, list):
                        for kw in code_keywords:
                            if isinstance(kw, str) and kw.strip():
                                current_flat_faq_questions.append(kw.strip().lower())
                                current_flat_faq_original_indices.append(code_original_idx_identifier)
                    elif isinstance(code_keywords, str) and code_keywords.strip():
                        current_flat_faq_questions.append(code_keywords.strip().lower())
                        current_flat_faq_original_indices.append(code_original_idx_identifier)


        faq_questions = current_flat_faq_questions
        faq_original_indices = current_flat_faq_original_indices

        if not model:
            logger.error("CRITICAL: Model is not loaded. Cannot build embeddings.")
            faq_embeddings = []
            return

        if faq_questions:
            faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
            logger.info(f"FAQ embeddings created with shape: {faq_embeddings.shape} for {len(faq_questions)} query items.")
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
    if current_faq_embeddings.ndim == 1:
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
                split_at = temp_content.rfind('\n\n', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 3:
                    split_at = temp_content.rfind('\n', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 3:
                    split_at = temp_content.rfind(' ', 0, MAX_LEN)
                if split_at == -1 or split_at < MAX_LEN / 3:
                    split_at = MAX_LEN
                parts.append(temp_content[:split_at])
                temp_content = temp_content[split_at:].lstrip()
            else:
                parts.append(temp_content)
                break
        for i, part in enumerate(parts):
            if part.strip():
                current_view = view if i == len(parts) - 1 else None
                msg = await channel.send(part, view=current_view)
                messages_sent.append(msg)
    return messages_sent

def parse_ordinal_or_number_to_index(text: str) -> int | None:
    text_lower = text.lower().strip()
    mapping = {
        "first": 0, "1st": 0, "one": 0, "1": 0,
        "second": 1, "2nd": 1, "two": 1, "2": 1,
        "third": 2, "3rd": 2, "three": 2, "3": 2,
        "fourth": 3, "4th": 3, "four": 3, "4": 3,
        "fifth": 4, "5th": 4, "five": 4, "5": 4,
        "last": -1
    }
    if text_lower in mapping:
        return mapping[text_lower]
    try:
        num = int(text_lower)
        if num > 0:
            return num - 1
    except ValueError:
        pass
    return None

async def forward_qa_to_admins(user_query: str, bot_answer_obj: any, original_author: discord.User):
    bot_answer_str = ""
    if isinstance(bot_answer_obj, dict) and "display_text_for_log" in bot_answer_obj:
        bot_answer_str = bot_answer_obj["display_text_for_log"]
    elif isinstance(bot_answer_obj, str):
        bot_answer_str = bot_answer_obj
    elif bot_answer_obj is None:
        return
    else:
        bot_answer_str = str(bot_answer_obj)

    if not bot_answer_str or not bot_answer_str.strip():
        return

    safe_user_query = user_query.replace("`", "'")
    safe_bot_answer = bot_answer_str.replace("`", "'")
    MAX_QUERY_LOG_LEN = 600
    MAX_ANSWER_LOG_LEN = 1200
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
    if len(forward_message) > 1990:
        error_notice = "... (Log message content was too long to display fully)"
        max_query_fallback_len = max(50, MAX_QUERY_LOG_LEN // 3)
        log_query_part_fallback = safe_user_query[:max_query_fallback_len] + "..."
        base_len_for_fallback = len(
            f"**User:** {original_author.name}#{original_author.discriminator} (ID: {original_author.id})\n"
            f"**Asked:** ```\n{log_query_part_fallback}\n```\n"
            f"**Bot Answered:** ```\n\n```"
        )
        max_answer_fallback_len = 1990 - base_len_for_fallback - len(error_notice) - 10
        max_answer_fallback_len = max(50, max_answer_fallback_len)
        log_answer_part_fallback = safe_bot_answer[:max_answer_fallback_len - 3] + "..." + f"\n{error_notice}"
        forward_message = (
            f"**User:** {original_author.name}#{original_author.discriminator} (ID: {original_author.id})\n"
            f"**Asked:** ```\n{log_query_part_fallback}\n```\n"
            f"**Bot Answered:** ```\n{log_answer_part_fallback}\n```"
        )
        if len(forward_message) > 1990:
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
        except discord.Forbidden:
            logger.warning(f"Bot is blocked by or cannot DM admin user {admin_id}. Disabling logs for them.")
            admin_log_activation_status[admin_id] = False
        except discord.HTTPException as e:
            if e.status == 400 and e.code == 50035:
                logger.error(f"Failed to send log to admin {admin_id} due to message length or formatting. Original Message Length: {len(forward_message)}. Query: '{user_query[:50]}...', Answer: '{bot_answer_str[:50]}...'. Code: {e.code}, Status: {e.status}")
                try:
                    if admin_user:
                         await admin_user.send(f"Failed to send a full Q&A log due to message length. User: {original_author.name}, Query: '{user_query[:100]}...'")
                except Exception as inner_e:
                    logger.error(f"Failed to send even the short error notification to admin {admin_id}: {inner_e}")
            else:
                logger.error(f"HTTPException forwarding Q&A to admin {admin_id}: {e}")
        except Exception as e:
            logger.error(f"Error forwarding Q&A to admin {admin_id}: {e}")

class SuggestionButton(discord.ui.Button):
    def __init__(self, label, style, custom_id, original_query, matched_keyword_text, faq_item_index, original_faq_item_idx, helpful):
        super().__init__(label=label, style=style, custom_id=custom_id)
        self.original_query = original_query
        self.matched_keyword_text = matched_keyword_text
        self.faq_item_index = faq_item_index
        self.original_faq_item_idx = original_faq_item_idx
        self.helpful = helpful

    async def callback(self, interaction: discord.Interaction):
        logger.info(f"SuggestionButton callback initiated by user {interaction.user.id} for custom_id {self.custom_id}. Helpful: {self.helpful}. Original FAQ Idx: {self.original_faq_item_idx}")
        try:
            feedback_type = "Helpful" if self.helpful else "Not Helpful"
            original_item_idx_str = str(self.original_faq_item_idx) if self.original_faq_item_idx is not None else "N/A"
            log_message = f"Suggestion Feedback: User clicked '{feedback_type}'."
            log_extra = {
                'user_id': interaction.user.id,
                'username': str(interaction.user),
                'original_query_text': self.original_query,
                'details': (f"Feedback given: {feedback_type}. "
                            f"For matched keyword: '{self.matched_keyword_text}'. "
                            f"FAQ Item Index (flat keyword): {self.faq_item_index}. "
                            f"Original FAQ Item Index (of answer): {original_item_idx_str}.")
            }
            logger.info(log_message, extra=log_extra)
            try:
                if interaction.response.is_done():
                    logger.warning(f"SuggestionButton callback: Interaction for custom_id {self.custom_id} already responded to.")
                else:
                    await interaction.response.send_message(f"Thanks for your feedback on the suggestion about '{self.matched_keyword_text}'!", ephemeral=True)
                    logger.info(f"SuggestionButton callback: Successfully sent ephemeral response for custom_id {self.custom_id}.")
            except discord.errors.InteractionResponded:
                logger.warning(f"SuggestionButton callback: InteractionResponded error for custom_id {self.custom_id}.")
            except Exception as e_resp:
                logger.error(f"SuggestionButton callback: ERROR sending ephemeral response for custom_id {self.custom_id}. Error: {e_resp}", exc_info=True)

            if self.view:
                logger.info(f"SuggestionButton callback: View found for custom_id {self.custom_id}. Attempting to disable buttons.")
                for item in self.view.children:
                    if isinstance(item, discord.ui.Button):
                        item.disabled = True
                try:
                    if interaction.message:
                        await interaction.message.edit(view=self.view)
                        logger.info(f"SuggestionButton callback: Successfully edited message {interaction.message.id} to disable buttons.")
                    else:
                        logger.error(f"SuggestionButton callback: interaction.message is None for custom_id {self.custom_id}. Cannot disable buttons.")
                except discord.NotFound:
                    logger.warning(f"SuggestionButton callback: Could not find message to edit for custom_id {self.custom_id}.")
                except discord.Forbidden:
                     logger.error(f"SuggestionButton callback: Missing permissions to edit message for custom_id {self.custom_id}.")
                except Exception as e_edit:
                    logger.error(f"SuggestionButton callback: ERROR editing message to disable buttons for custom_id {self.custom_id}. Error: {e_edit}", exc_info=True)
            else:
                logger.warning(f"SuggestionButton callback: self.view is None for custom_id {self.custom_id}. Cannot disable buttons.")
        except Exception as e_outer:
            logger.error(f"SuggestionButton callback: UNHANDLED EXCEPTION in callback for custom_id {self.custom_id}. Error: {e_outer}", exc_info=True)
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message("Sorry, there was an error processing your feedback.", ephemeral=True)
            except discord.errors.InteractionResponded: pass
            except Exception: logger.error(f"SuggestionButton callback: CRITICAL - Failed to even send a generic error message after outer exception.", exc_info=True)

@bot.event
async def on_ready():
    global nlp
    print(f'{bot.user.name} (ID: {bot.user.id}) has connected to Discord!')
    print(f'Listening for DMs. All interactions are handled as direct messages.')
    load_spacy_model()
    load_faq_data_from_url()
    global admin_log_activation_status
    activated_admins_count = 0
    if ADMIN_USER_IDS:
        for admin_id in ADMIN_USER_IDS:
            admin_log_activation_status[admin_id] = True
            activated_admins_count +=1
        logger.info(f"Default log forwarding activated for {activated_admins_count} admin(s). Admins can use 'mute logs' to opt-out.")
        print(f"Default log forwarding activated for {activated_admins_count} admin(s).")
    else:
        logger.warning("No ADMIN_USER_IDS defined. Log forwarding will not be active by default for anyone.")
        print("WARNING: No ADMIN_USER_IDS defined. Log forwarding will not be active by default for anyone.")
    activity = discord.Activity(
        name="DM for HELP!",
        type=discord.ActivityType.custom,
        state="DM for HELP!"
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
    original_message_content_for_processing = message.content.strip()

    author_id = message.author.id
    author_name = str(message.author)
    log_extra_base = {'user_id': author_id, 'username': author_name, 'original_query_text': original_message_content_for_processing}

    # --- MAX_QUERY_LENGTH Check ---
    query_words_for_length_check = original_message_content_for_processing.split()
    if len(query_words_for_length_check) > MAX_QUERY_LENGTH:
        response_text = "For best results please keep questions brief and only ask one question at a time"
        await message.channel.send(response_text)
        logger.info(
            "Query exceeded MAX_QUERY_LENGTH.",
            extra={**log_extra_base, 'details': f"Query length: {len(query_words_for_length_check)} words. Limit: {MAX_QUERY_LENGTH}."}
        )
        # No need to forward this specific interaction to admins, as it's a structural rejection
        return
    # --- End MAX_QUERY_LENGTH Check ---


    context_was_applied_this_turn = False
    context_application_method = "none"
    faq_answer_part_text = None
    greeting_matched_this_interaction = False

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
    if author_id in user_context_history and user_context_history[author_id]:
        previous_user_query_context_text, previous_bot_answer_context_obj = user_context_history[author_id][-1]
        cleaned_query_for_word_count = re.sub(r'[^\w\s]', '', user_query_lower_for_processing)
        query_words = cleaned_query_for_word_count.split()
        is_short_query_for_list_follow_up = len(query_words) <= CONTEXT_QUERY_LENGTH_THRESHOLD + 2

        if is_short_query_for_list_follow_up and \
           isinstance(previous_bot_answer_context_obj, dict) and \
           previous_bot_answer_context_obj.get("type") == "code_list_sent":
            codes_ordered = previous_bot_answer_context_obj.get("codes_ordered")
            if codes_ordered:
                potential_ordinals = re.findall(r'\b(first|second|third|fourth|fifth|last|\d+)\b', user_query_lower_for_processing)
                target_code_name = None
                found_ordinal_str = None
                for p_ord in potential_ordinals:
                    index = parse_ordinal_or_number_to_index(p_ord)
                    if index is not None:
                        actual_index = index
                        if index == -1:
                            if not codes_ordered: continue
                            actual_index = len(codes_ordered) - 1
                        if 0 <= actual_index < len(codes_ordered):
                            target_code_name = codes_ordered[actual_index]
                            found_ordinal_str = p_ord
                            break
                if target_code_name:
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
                    log_extra_base['original_query_text'] = current_original_content_for_faq
                    context_was_applied_this_turn = True

        if not context_was_applied_this_turn and nlp:
            found_pronoun_in_current_query = None
            for word_idx, word in enumerate(query_words):
                if word in CONTEXT_PRONOUNS:
                    found_pronoun_in_current_query = word
                    break
            if len(query_words) <= CONTEXT_QUERY_LENGTH_THRESHOLD and found_pronoun_in_current_query:
                referent = None
                if isinstance(previous_user_query_context_text, str):
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
                    else:
                        if isinstance(previous_user_query_context_text, str):
                            current_original_content_for_faq = f"{previous_user_query_context_text} {original_message_content_for_processing}"
                            current_query_for_faq_lower = f"{previous_user_query_context_text.lower()} {user_query_lower_for_processing}"
                            context_application_method = "prepended (rewrite failed)"
                            logger.info("Context Used (Prepended - Pronoun Rewrite Failed)...", extra={'user_id':author_id, 'username':author_name, 'original_query_text': initial_user_message_content_for_forwarding, 'details': f"Failed to replace '{found_pronoun_in_current_query}' with '{referent}'. Enhanced: '{current_original_content_for_faq}'."})
                elif isinstance(previous_user_query_context_text, str):
                    current_original_content_for_faq = f"{previous_user_query_context_text} {original_message_content_for_processing}"
                    current_query_for_faq_lower = f"{previous_user_query_context_text.lower()} {user_query_lower_for_processing}"
                    context_application_method = "prepended (no referent)"
                    logger.info("Context Used (Prepended - No Referent Found)...", extra={'user_id':author_id, 'username':author_name, 'original_query_text': initial_user_message_content_for_forwarding, 'details': f"Enhanced: '{current_original_content_for_faq}'."})
                if context_application_method in ["rewritten", "prepended (rewrite failed)", "prepended (no referent)"]:
                    log_extra_base['original_query_text'] = current_original_content_for_faq
                    context_was_applied_this_turn = True
            elif author_id in user_context_history and user_context_history[author_id]:
                reason = []
                if not nlp: reason.append("spaCy (nlp) not loaded")
                if not (len(query_words) <= CONTEXT_QUERY_LENGTH_THRESHOLD):
                    reason.append(f"Query too long (len {len(query_words)} > {CONTEXT_QUERY_LENGTH_THRESHOLD})")
                if not found_pronoun_in_current_query and (len(query_words) <= CONTEXT_QUERY_LENGTH_THRESHOLD):
                    reason.append("No qualifying pronoun found in short query")
                if reason:
                    logger.info(
                        f"Context NOT Used: Conditions not met for pronoun resolution. {' & '.join(reason)}.",
                        extra=log_extra_base
                    )
    elif not nlp and author_id in user_context_history and user_context_history[author_id]:
         logger.warning("Context NOT Used: spaCy model (nlp) not loaded. Pronoun resolution disabled.", extra=log_extra_base)

    # --- Pronunciation Handling ---
    pronounce_prefix = "!pronounce "
    pronounce_prefix_no_bang = "pronounce "
    word_to_pronounce_input = None
    if original_message_content_for_processing.lower().startswith(pronounce_prefix):
        word_to_pronounce_input = original_message_content_for_processing[len(pronounce_prefix):].strip()
    elif original_message_content_for_processing.lower().startswith(pronounce_prefix_no_bang):
        temp_word = original_message_content_for_processing[len(pronounce_prefix_no_bang):].strip()
        if temp_word and len(original_message_content_for_processing.split()) < 5:
             word_to_pronounce_input = temp_word
    if word_to_pronounce_input:
        response_to_user = ""
        pronunciation_view = discord.ui.View()
        if not word_to_pronounce_input:
            response_to_user = "Please tell me what word or phrase you want to pronounce. Usage: `pronounce 'word or phrase'`"
            await message.channel.send(response_to_user)
        else:
            async with message.channel.typing():
                audio_url = await get_pronunciation_audio_url(word_to_pronounce_input)
            encoded_word_for_google = urllib.parse.quote_plus(word_to_pronounce_input)
            google_link = f"https://www.google.com/search?q=how+to+pronounce+{encoded_word_for_google}"
            youtube_link = f"https://www.youtube.com/results?search_query=how+to+pronounce+{encoded_word_for_google}"
            response_message_lines = [f"Pronunciation resources for \"**{word_to_pronounce_input}**\":"]
            log_audio_status = "Audio not found from API."
            if audio_url:
                pronunciation_view.add_item(discord.ui.Button(label="Play Sound", style=discord.ButtonStyle.link, url=audio_url, emoji="🔊"))
                log_audio_status = f"Audio found from API: {audio_url}"
            else:
                response_message_lines.append(f"• for \"{word_to_pronounce_input}\"..")
            pronunciation_view.add_item(discord.ui.Button(label="Search on Google", style=discord.ButtonStyle.link, url=google_link))
            pronunciation_view.add_item(discord.ui.Button(label="Search on YouTube", style=discord.ButtonStyle.link, url=youtube_link))
            if not audio_url:
                 response_message_lines.append(f"• Check Google/YouTube for pronunciation resources automatically." )
            response_to_user = "\n".join(response_message_lines)
            if pronunciation_view.children:
                 await message.channel.send(response_to_user, view=pronunciation_view)
            else:
                 await message.channel.send(response_to_user)
        bot_reply_parts_for_forwarding.append(response_to_user + (" (View with buttons also sent)" if pronunciation_view.children else ""))
        cmd_log_extra = {'user_id': author_id, 'username': author_name, 'original_query_text': initial_user_message_content_for_forwarding}
        logger.info(f"Pronunciation request processed for '{word_to_pronounce_input}'.", extra={**cmd_log_extra, 'details': f"Audio from API: {log_audio_status}. Links provided."})
        await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
        return

    # --- List Codes Handling ---
    if (user_query_lower_for_processing == "list codes" or \
        user_query_lower_for_processing == "codes" or \
        user_query_lower_for_processing == "code list") and \
       not context_application_method == "list_item_resolved":
        call_codes_section = faq_data.get("call_codes", {})
        cmd_log_extra = {'user_id': author_id, 'username': author_name, 'original_query_text': initial_user_message_content_for_forwarding}
        faq_answer_part_for_context_storage = None

        if not call_codes_section:
            reply_text = "I don't have any call codes defined at the moment."
            await message.channel.send(reply_text)
            bot_reply_parts_for_forwarding.append(reply_text)
            faq_answer_part_for_context_storage = reply_text
            logger.info("Command 'list codes' - no codes found.", extra=cmd_log_extra)
        else:
            embed = discord.Embed(title="☎️ Call Disposition Codes", color=discord.Color.blue())
            current_field_value = ""
            field_count = 0
            MAX_FIELD_VALUE_LEN = 1020
            MAX_FIELDS = 24
            temp_forward_text_parts_for_log = ["Call Codes Sent:"]
            ordered_code_keys_for_context = list(call_codes_section.keys())

            for code_key in ordered_code_keys_for_context:
                code_obj = call_codes_section.get(code_key)
                description = "Not defined."
                if isinstance(code_obj, dict) and "answer" in code_obj:
                    description = code_obj["answer"]
                elif isinstance(code_obj, str):
                    description = code_obj

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
            if current_field_value and field_count < MAX_FIELDS:
                embed.add_field(name=f"Codes (Part {field_count + 1})" if field_count > 0 or not embed.fields else "Codes", value=current_field_value, inline=False)

            if not embed.fields and not current_field_value:
                reply_text = "No call codes formatted. Check data."
                await message.channel.send(reply_text)
                bot_reply_parts_for_forwarding.append(reply_text)
                faq_answer_part_for_context_storage = reply_text
            elif not embed.fields:
                 reply_text = "Found codes, but couldn't display them. Try again/ask manager."
                 await message.channel.send(reply_text)
                 bot_reply_parts_for_forwarding.append(reply_text)
                 faq_answer_part_for_context_storage = reply_text
            else:
                await message.channel.send(embed=embed)
                bot_reply_text_for_log = "[Bot sent 'list codes' as an embed.]\n" + "\n".join(temp_forward_text_parts_for_log)
                bot_reply_parts_for_forwarding.append(bot_reply_text_for_log)
                faq_answer_part_for_context_storage = {
                    "type": "code_list_sent",
                    "codes_ordered": ordered_code_keys_for_context,
                    "display_text_for_log": bot_reply_text_for_log
                }
            logger.info("Command 'list codes' processed.", extra={**cmd_log_extra, 'details': f"Displayed {len(call_codes_section) if call_codes_section else 0} codes."})

        if author_id not in user_context_history:
            user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
        user_context_history[author_id].append((initial_user_message_content_for_forwarding, faq_answer_part_for_context_storage))
        log_details_ctx_list_q = initial_user_message_content_for_forwarding[:100]
        log_details_ctx_list_a = str(faq_answer_part_for_context_storage)[:100] if not isinstance(faq_answer_part_for_context_storage, dict) else "special code_list_sent object"
        logger.info(f"Context Stored after 'list codes' for user {author_id}.", extra={**cmd_log_extra, 'details': f"Stored Q: '{log_details_ctx_list_q}...'. Stored A: '{log_details_ctx_list_a}...'"})
        await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
        return

    # --- Define Code Handling (Updated for nested call_codes structure) ---
    define_pattern = r"^(?:what is|define|explain)\s+([\w\s-]+)\??$"
    match_define_code = re.match(define_pattern, current_query_for_faq_lower)

    if match_define_code:
        code_name_from_query_to_lookup = match_define_code.group(1).strip()
        code_name_query_upper = code_name_from_query_to_lookup.upper()

        call_codes_section = faq_data.get("call_codes", {})
        found_code_key = next((k for k in call_codes_section if k.upper() == code_name_query_upper), None)

        defined_code_embed = None
        temp_bot_reply_for_define = None
        cmd_log_extra_define = {'user_id': author_id, 'username': author_name, 'original_query_text': initial_user_message_content_for_forwarding}

        if found_code_key:
            code_data_object = call_codes_section[found_code_key]
            description_string = "Error: Code definition not found or format incorrect."
            if isinstance(code_data_object, dict) and "answer" in code_data_object:
                description_string = code_data_object["answer"]
            elif isinstance(code_data_object, str): # Fallback for old structure
                description_string = code_data_object
            else:
                logger.error(f"Define Code: Code data for '{found_code_key}' is not in expected format. Found: {type(code_data_object)}", extra=cmd_log_extra_define)

            defined_code_embed = discord.Embed(title=f"Definition: {found_code_key.upper()}", description=description_string, color=discord.Color.purple())
            temp_bot_reply_for_define = f"Definition: {found_code_key.upper()}\n{description_string}"
            log_detail_msg = f"Defined code '{found_code_key.upper()}' (exact match - from query '{current_original_content_for_faq}')."
            if context_was_applied_this_turn : log_detail_msg += f" Context method: {context_application_method}."
            logger.info(log_detail_msg, extra=cmd_log_extra_define)

        elif call_codes_section: # Only attempt fuzzy if there are codes to search AND exact match failed
            original_term_for_fuzzy_search = code_name_from_query_to_lookup # Use term from initial "define" regex

            fuzzy_match_result = process.extractOne(
                original_term_for_fuzzy_search,
                list(call_codes_section.keys()),
                scorer=fuzz.token_set_ratio
            )

            if fuzzy_match_result: # Check if extractOne returned something
                best_match_code, score = fuzzy_match_result
                if score > 80 and best_match_code: # Your existing threshold
                    code_data_object_fuzzy = call_codes_section[best_match_code]
                    description_string_fuzzy = "Error: Fuzzy matched code definition format incorrect."
                    if isinstance(code_data_object_fuzzy, dict) and "answer" in code_data_object_fuzzy:
                        description_string_fuzzy = code_data_object_fuzzy["answer"]
                    elif isinstance(code_data_object_fuzzy, str):
                         description_string_fuzzy = code_data_object_fuzzy

                    defined_code_embed = discord.Embed(title=f"Definition (for '{original_term_for_fuzzy_search}'): {best_match_code.upper()}", description=description_string_fuzzy, color=discord.Color.purple())
                    temp_bot_reply_for_define = f"Definition (for '{original_term_for_fuzzy_search}'): {best_match_code.upper()}\n{description_string_fuzzy}"
                    logger.info(f"Defined code '{best_match_code.upper()}' (fuzzy match on '{original_term_for_fuzzy_search}').", extra={**cmd_log_extra_define, 'details': f"Score: {score}."})
            # No explicit 'else' for no fuzzy match; if defined_code_embed is still None, nothing will be sent.

        if defined_code_embed:
            await message.channel.send(embed=defined_code_embed)
            if temp_bot_reply_for_define: bot_reply_parts_for_forwarding.append(temp_bot_reply_for_define)
            if author_id not in user_context_history:
                user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
            user_context_history[author_id].append((current_original_content_for_faq, temp_bot_reply_for_define))
            log_details_ctx_define_q = current_original_content_for_faq[:100]
            log_details_ctx_define_a = temp_bot_reply_for_define[:100] if temp_bot_reply_for_define else "N/A"
            logger.info(f"Context Stored after 'define code' for user {author_id}.", extra={**cmd_log_extra_define, 'details': f"Stored Q: '{log_details_ctx_define_q}...'. Stored A: '{log_details_ctx_define_a}...'"})
            await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
            return # Crucial: if define command was processed, don't fall through to semantic search


    # --- Greeting Handling ---
    greetings_data = faq_data.get("greetings_and_pleasantries", [])
    best_greeting_match_entry = None
    len_of_matched_greeting_keyword = 0
    matched_greeting_keyword_original_casing = None
    greeting_log_extra = {**log_extra_base}
    for greeting_entry in greetings_data:
        keywords = greeting_entry.get("keywords", [])
        if not keywords: continue
        for kw in keywords:
            kw_lower = kw.lower()
            if current_query_for_faq_lower.startswith(kw_lower):
                if len(kw_lower) > len_of_matched_greeting_keyword:
                    len_of_matched_greeting_keyword = len(kw_lower)
                    best_greeting_match_entry = greeting_entry
                    matched_greeting_keyword_original_casing = current_original_content_for_faq[:len(kw_lower)]
    if best_greeting_match_entry:
        greeting_matched_this_interaction = True
        response_type = best_greeting_match_entry.get("response_type")
        greeting_reply_text = f"Hello {message.author.mention}!"
        if response_type == "standard_greeting":
            reply_template = best_greeting_match_entry.get("greeting_reply_template", "Hello there, {user_mention}!")
            actual_greeting_cased = matched_greeting_keyword_original_casing.capitalize() if matched_greeting_keyword_original_casing else "Hello"
            greeting_reply_text = reply_template.format(actual_greeting_cased=actual_greeting_cased, user_mention=message.author.mention)
        elif response_type == "specific_reply":
            greeting_reply_text = best_greeting_match_entry.get("reply_text", "I acknowledge that.")
        await message.channel.send(greeting_reply_text)
        bot_reply_parts_for_forwarding.append(greeting_reply_text)
        remainder_after_greeting = current_original_content_for_faq[len_of_matched_greeting_keyword:].strip()
        if remainder_after_greeting and remainder_after_greeting[0] in string.punctuation:
            remainder_after_greeting = remainder_after_greeting[1:].strip()
        logger.info(f"Greeting matched (prefix).", extra={**greeting_log_extra, 'details': f"Matched: '{matched_greeting_keyword_original_casing}'. Remainder for FAQ: '{remainder_after_greeting}'."})
        if not is_text_empty_or_punctuation_only(remainder_after_greeting) and len(remainder_after_greeting) > 3:
            current_query_for_faq_lower = remainder_after_greeting.lower()
            current_original_content_for_faq = remainder_after_greeting
            log_extra_base['original_query_text'] = current_original_content_for_faq
            logger.info(f"Processing remainder of query after greeting: '{current_original_content_for_faq}'", extra=log_extra_base)
        else:
            if author_id not in user_context_history:
                user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
            user_context_history[author_id].append((matched_greeting_keyword_original_casing if matched_greeting_keyword_original_casing else current_original_content_for_faq, greeting_reply_text))
            logger.info(f"Context Stored after terminal greeting.", extra={**greeting_log_extra, 'details': f"Stored Q: '{matched_greeting_keyword_original_casing}'. Stored A: '{greeting_reply_text[:100]}...'"})
            await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
            return
    elif len(current_query_for_faq_lower) < 30:
        for greeting_entry in greetings_data:
            keywords = greeting_entry.get("keywords", [])
            if not keywords: continue
            match_result = process.extractOne(current_query_for_faq_lower, keywords, scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_MATCH_THRESHOLD_GREETINGS)
            if match_result:
                matched_keyword_from_fuzz, score = match_result
                greeting_matched_this_interaction = True
                response_type = greeting_entry.get("response_type")
                reply_text = f"Hello {message.author.mention}!"
                if response_type == "standard_greeting":
                    reply_template = greeting_entry.get("greeting_reply_template", "Hello there, {user_mention}!")
                    actual_greeting_cased = next((kw for kw in keywords if kw.lower() == matched_keyword_from_fuzz.lower()), matched_keyword_from_fuzz)
                    reply_text = reply_template.format(actual_greeting_cased=actual_greeting_cased.capitalize(), user_mention=message.author.mention)
                elif response_type == "specific_reply":
                    reply_text = greeting_entry.get("reply_text", "I acknowledge that.")
                await message.channel.send(reply_text)
                bot_reply_parts_for_forwarding.append(reply_text)
                logger.info(f"Greeting matched (fuzzy, short query).", extra={**greeting_log_extra, 'details': f"Matched '{current_query_for_faq_lower}' with '{matched_keyword_from_fuzz}', Score: {score}."})
                if author_id not in user_context_history:
                    user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
                user_context_history[author_id].append((current_original_content_for_faq, reply_text))
                logger.info(f"Context Stored after terminal fuzzy greeting.", extra={**greeting_log_extra, 'details': f"Stored Q: '{current_original_content_for_faq}'. Stored A: '{reply_text[:100]}...'"})
                await forward_qa_to_admins(initial_user_message_content_for_forwarding, "\n".join(bot_reply_parts_for_forwarding), message.author)
                return

    # --- Semantic Search and Dynamic Threshold Logic ---
    faq_items_original_list = faq_data.get("general_faqs", [])
    call_codes_faq_section = faq_data.get("call_codes", {})

    top_indices, top_scores = semantic_search_top_n_matches(current_query_for_faq_lower, n=CANDIDATES_TO_FETCH_FOR_SUGGESTIONS)
    action = "fallback"
    dynamic_decision_details = "N/A"

    if context_was_applied_this_turn:
        top_matches_log_details = []
        if top_indices:
            for i_log in range(min(len(top_indices), 3)):
                idx_log = top_indices[i_log]
                score_log = top_scores[i_log]
                if 0 <= idx_log < len(faq_original_indices) and 0 <= idx_log < len(faq_questions):
                    original_faq_idx_ref = faq_original_indices[idx_log]
                    keyword_text_log = faq_questions[idx_log]
                    top_matches_log_details.append(f"(Score: {score_log:.3f}, OrigRef: {original_faq_idx_ref}, MatchedKW: '{keyword_text_log[:60]}...')")
                else:
                    top_matches_log_details.append(f"(Score: {score_log:.3f}, FlatIdx: {idx_log} - out of bounds)")
        logger.info(
            f"Semantic search on CONTEXT-APPLIED query ({context_application_method}). Top matches: {'; '.join(top_matches_log_details) if top_matches_log_details else 'No semantic hits.'}",
            extra={**log_extra_base, 'details': f"Applied query: '{current_original_content_for_faq}'. Raw short query (if applicable): '{original_message_content_for_processing if context_application_method != 'none' else 'N/A'}'."}
        )

    if not top_indices:
        if not greeting_matched_this_interaction:
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
                                                       gap < SIMILAR_SCORE_CLUSTER_THRESHOLD and
                                                       secondary_score >= SUGGESTION_THRESHOLD)
                if is_dynamically_confirmed_direct:
                    action = "direct_answer"
                    dynamic_decision_details = f"Dynamic: Direct. P={primary_score:.2f}, S={secondary_score:.2f}. Gap {gap:.2f} >= {CONFIDENCE_GAP_FOR_DIRECT_ANSWER} or num_cand=1."
                elif is_ambiguous_cluster_for_suggestions:
                    action = "suggestions"
                    dynamic_decision_details = f"Dynamic: Suggestions (ambiguous cluster). P={primary_score:.2f}, S={secondary_score:.2f}. Gap {gap:.2f} < {SIMILAR_SCORE_CLUSTER_THRESHOLD} & S >= {SUGGESTION_THRESHOLD}."
                else:
                    if primary_score >= SUGGESTION_THRESHOLD :
                         action = "direct_answer"
                         dynamic_decision_details = f"Dynamic: Direct (P >= SemThresh, not cluster, weak/no secondary). P={primary_score:.2f}, S={secondary_score:.2f}."
                    else:
                         action = "suggestions" # Fallback to suggestions if P is high enough but doesn't meet direct criteria
                         dynamic_decision_details = f"Dynamic: Suggestions (P >= SemThresh but not dyn_direct/cluster, fallback to sugg). P={primary_score:.2f}, S={secondary_score:.2f}. Gap {gap:.2f}."
            elif primary_score >= SUGGESTION_THRESHOLD:
                action = "suggestions"
                dynamic_decision_details = f"Dynamic: Suggestions (P < SemThresh but P >= SugThresh). P={primary_score:.2f}."
            if action == "fallback" and dynamic_decision_details == "N/A":
                 dynamic_decision_details = f"Dynamic: Fallback. P={primary_score:.2f} < {SUGGESTION_THRESHOLD} (or other rule miss)."
        else:
            dynamic_decision_details = "Static Thresholds Used."
            if primary_score >= SEMANTIC_SEARCH_THRESHOLD:
                action = "direct_answer"
            elif primary_score >= SUGGESTION_THRESHOLD:
                action = "suggestions"

        if action == "direct_answer":
            primary_faq_flat_idx = top_indices[0]
            if not (0 <= primary_faq_flat_idx < len(faq_questions) and 0 <= primary_faq_flat_idx < len(faq_original_indices)):
                logger.error(f"Semantic primary match index {primary_faq_flat_idx} out of bounds. {dynamic_decision_details}", extra=log_extra_base)
                faq_answer_part_text = faq_data.get("fallback_message", "Sorry, an error occurred.")
                await message.channel.send(faq_answer_part_text)
            else:
                original_faq_ref = faq_original_indices[primary_faq_flat_idx]
                matched_keyword_text_log = faq_questions[primary_faq_flat_idx]

                if isinstance(original_faq_ref, int) and 0 <= original_faq_ref < len(faq_items_original_list): # It's a general_faq
                    matched_item_primary = faq_items_original_list[original_faq_ref]
                    faq_answer_part_text = matched_item_primary.get("answer", "Answer not available.")
                    logger.info("Semantic FAQ Direct Match (General FAQ).", extra={**log_extra_base, 'details': f"Score: {primary_score:.2f}. Matched Keyword Flat Idx: '{primary_faq_flat_idx}', Original FAQ Idx: {original_faq_ref}. Keyword: '{matched_keyword_text_log}'. Decision: {dynamic_decision_details}"})
                elif isinstance(original_faq_ref, str) and original_faq_ref.startswith("code_"): # It's a call_code
                    code_name_from_ref = original_faq_ref[len("code_"):].replace('_', ' ')
                    code_obj = call_codes_faq_section.get(code_name_from_ref)
                    if not code_obj:
                         code_obj = call_codes_faq_section.get(code_name_from_ref.upper())

                    if isinstance(code_obj, dict) and "answer" in code_obj:
                        faq_answer_part_text = code_obj["answer"]
                        embed = discord.Embed(title=f"Definition: {code_name_from_ref.upper()}", description=faq_answer_part_text, color=discord.Color.purple())
                        await message.channel.send(embed=embed)
                        logger.info("Semantic FAQ Direct Match (Call Code).", extra={**log_extra_base, 'details': f"Score: {primary_score:.2f}. Matched Keyword Flat Idx: '{primary_faq_flat_idx}', Matched Code: {code_name_from_ref}. Keyword: '{matched_keyword_text_log}'. Decision: {dynamic_decision_details}"})
                    else:
                        logger.error(f"Semantic direct match for code '{code_name_from_ref}' failed to find answer. Ref: {original_faq_ref}. {dynamic_decision_details}", extra=log_extra_base)
                        faq_answer_part_text = faq_data.get("fallback_message", "Sorry, an error retrieving code definition.")
                else:
                    logger.error(f"Semantic primary original ref '{original_faq_ref}' is invalid type or out of bounds. {dynamic_decision_details}", extra=log_extra_base)
                    faq_answer_part_text = faq_data.get("fallback_message", "Sorry, an error occurred.")

                if faq_answer_part_text and not (isinstance(original_faq_ref, str) and original_faq_ref.startswith("code_")):
                    await send_long_message(message.channel, faq_answer_part_text, view=None)


        elif action == "suggestions":
            suggestions_to_display = []
            shown_original_faq_refs = set()
            for i in range(len(top_indices)):
                current_flat_idx = top_indices[i]
                current_score = top_scores[i]
                if current_score < SUGGESTION_THRESHOLD: break
                if not (0 <= current_flat_idx < len(faq_original_indices) and 0 <= current_flat_idx < len(faq_questions)):
                    logger.warning(f"Suggestion selection: flat_idx {current_flat_idx} out of bounds. Decision: {dynamic_decision_details}")
                    continue

                current_original_faq_ref = faq_original_indices[current_flat_idx]
                if current_original_faq_ref not in shown_original_faq_refs:
                    answer_text = "Answer not available."
                    matched_keyword_text = faq_questions[current_flat_idx]
                    if matched_keyword_text: matched_keyword_text = matched_keyword_text[0].upper() + matched_keyword_text[1:]

                    sugg_original_idx_for_button = current_original_faq_ref

                    if isinstance(current_original_faq_ref, int) and 0 <= current_original_faq_ref < len(faq_items_original_list):
                        matched_item = faq_items_original_list[current_original_faq_ref]
                        answer_text = matched_item.get("answer", "Answer not available.")
                    elif isinstance(current_original_faq_ref, str) and current_original_faq_ref.startswith("code_"):
                        code_name_from_ref = current_original_faq_ref[len("code_"):].replace('_', ' ')
                        code_obj = call_codes_faq_section.get(code_name_from_ref) or call_codes_faq_section.get(code_name_from_ref.upper())
                        if isinstance(code_obj, dict) and "answer" in code_obj:
                            answer_text = f"Definition for **{code_name_from_ref.upper()}**: {code_obj['answer']}"
                        else:
                            logger.warning(f"Suggestion for code '{code_name_from_ref}' failed to find answer. Ref: {current_original_faq_ref}")
                            continue
                    else:
                        logger.warning(f"Suggestion selection: original_faq_ref '{current_original_faq_ref}' invalid. Decision: {dynamic_decision_details}")
                        continue

                    suggestions_to_display.append({
                        "score": current_score, "flat_idx": current_flat_idx, "original_ref": sugg_original_idx_for_button,
                        "answer": answer_text, "keyword": matched_keyword_text
                    })
                    shown_original_faq_refs.add(current_original_faq_ref)
                if len(suggestions_to_display) >= MAX_SUGGESTIONS_TO_SHOW: break

            if suggestions_to_display:
                user_verbatim_input_this_turn = initial_user_message_content_for_forwarding
                intro_message_text = f"I'm not sure I have an exact answer for '{user_verbatim_input_this_turn}', try typing `list codes` or perhaps one of these is helpful?\n"
                await send_long_message(message.channel, intro_message_text)
                temp_suggestion_forward_texts = [intro_message_text.strip()]
                logger_details_base_sugg = f"Decision: {dynamic_decision_details}. Primary Score: {primary_score:.2f}."
                for i, sugg_data in enumerate(suggestions_to_display):
                    embed_title = f"🤔 Suggestion {i+1}: Related to '{sugg_data['keyword']}'"
                    if len(embed_title) > 256: embed_title = embed_title[:253] + "..."
                    embed = discord.Embed(title=embed_title, description=sugg_data['answer'], color=discord.Color.gold())
                    btn_yes = SuggestionButton(label="✅ Helpful!", style=discord.ButtonStyle.success, custom_id=f"sugg_yes_{message.id}_{sugg_data['flat_idx']}_{sugg_data['original_ref']}", original_query=current_original_content_for_faq, matched_keyword_text=sugg_data['keyword'], faq_item_index=sugg_data['flat_idx'], original_faq_item_idx=sugg_data['original_ref'], helpful=True)
                    btn_no = SuggestionButton(label="❌ Not quite", style=discord.ButtonStyle.danger, custom_id=f"sugg_no_{message.id}_{sugg_data['flat_idx']}_{sugg_data['original_ref']}", original_query=current_original_content_for_faq, matched_keyword_text=sugg_data['keyword'], faq_item_index=sugg_data['flat_idx'], original_faq_item_idx=sugg_data['original_ref'], helpful=False)
                    sugg_view = discord.ui.View(timeout=300); sugg_view.add_item(btn_yes); sugg_view.add_item(btn_no)
                    await message.channel.send(embed=embed, view=sugg_view)
                    temp_suggestion_forward_texts.append(f"Suggestion {i+1} (Score {sugg_data['score']:.2f}, Orig. Ref: {sugg_data['original_ref']}) '{sugg_data['keyword']}':\n{sugg_data['answer']}")
                    logger.info(f"Semantic Suggestion {i+1} offered.", extra={**log_extra_base, 'details': f"{logger_details_base_sugg} Sugg Score: {sugg_data['score']:.2f}, Keyword: '{sugg_data['keyword']}', Orig. Ref: {sugg_data['original_ref']}."})
                faq_answer_part_text = "\n---\n".join(temp_suggestion_forward_texts)
            else:
                if not greeting_matched_this_interaction:
                    faq_answer_part_text = faq_data.get("fallback_message", "I'm sorry, I couldn't find a clear answer for that.")
                    await message.channel.send(faq_answer_part_text)
                    logger.info("Unanswered query, fallback sent (no distinct suggestions met threshold).", extra={**log_extra_base, 'details': dynamic_decision_details})

        elif action == "fallback":
            if not greeting_matched_this_interaction:
                faq_answer_part_text = faq_data.get("fallback_message", "I'm sorry, I couldn't find an answer for that.")
                await message.channel.send(faq_answer_part_text)
                logger.info("Unanswered query, fallback sent (low semantic score or dynamic rules).", extra={**log_extra_base, 'details': dynamic_decision_details})

    # --- Context Storage (General) ---
    if faq_answer_part_text:
        if author_id not in user_context_history:
            user_context_history[author_id] = collections.deque(maxlen=CONTEXT_WINDOW_SIZE)
        user_context_history[author_id].append((current_original_content_for_faq, faq_answer_part_text))
        context_storage_log_extra = {'user_id': author_id, 'username': author_name,
                                     'original_query_text': initial_user_message_content_for_forwarding}
        stored_q_part_for_log = current_original_content_for_faq[:100]
        stored_a_part_for_log_display = ""
        if isinstance(faq_answer_part_text, dict) and "display_text_for_log" in faq_answer_part_text:
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
    final_bot_reply_for_forwarding = ""
    greeting_text_from_parts = ""
    if greeting_matched_this_interaction and bot_reply_parts_for_forwarding:
        greeting_text_from_parts = bot_reply_parts_for_forwarding[0]
    semantic_or_fallback_text_for_log = ""
    if isinstance(faq_answer_part_text, str):
        semantic_or_fallback_text_for_log = faq_answer_part_text
    if greeting_text_from_parts and semantic_or_fallback_text_for_log:
        if greeting_text_from_parts.strip() == semantic_or_fallback_text_for_log.strip():
            final_bot_reply_for_forwarding = greeting_text_from_parts
        else:
            final_bot_reply_for_forwarding = (
                f"{greeting_text_from_parts}\n"
                f"THEN, for the query part ('{current_original_content_for_faq}'):\n"
                f"{semantic_or_fallback_text_for_log}"
            )
    elif greeting_text_from_parts:
        final_bot_reply_for_forwarding = greeting_text_from_parts
    elif semantic_or_fallback_text_for_log:
        final_bot_reply_for_forwarding = semantic_or_fallback_text_for_log
    if final_bot_reply_for_forwarding:
        await forward_qa_to_admins(initial_user_message_content_for_forwarding, final_bot_reply_for_forwarding, message.author)

# --- Run the Bot ---
if __name__ == "__main__":
    actual_admin_ids = {uid for uid in ADMIN_USER_IDS if isinstance(uid, int)}
    if len(actual_admin_ids) < len(ADMIN_USER_IDS):
        print("INFO: Placeholder or non-integer admin IDs were filtered out. Ensure all admin IDs are correct integers.")
        logger.info("Non-integer admin IDs filtered out.", extra={'details': f"Original: {ADMIN_USER_IDS}, Filtered: {actual_admin_ids}"})
    ADMIN_USER_IDS = actual_admin_ids
    if not ADMIN_USER_IDS:
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