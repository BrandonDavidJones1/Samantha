
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

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
FAQ_URL = "https://raw.githubusercontent.com/BrandonDavidJones1/Samantha/main/faq_data.json"
FUZZY_MATCH_THRESHOLD_GREETINGS = 75
LOG_FILE = "unanswered_queries.log"
SEMANTIC_SEARCH_THRESHOLD = 0.60
SUGGESTION_THRESHOLD = 0.45
MAX_SUGGESTIONS_TO_SHOW = 2 # Max number of distinct suggestions to show
CANDIDATES_TO_FETCH_FOR_SUGGESTIONS = 5 # How many top semantic matches to consider for suggestions

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

# --- Admin and Log Forwarding Globals ---
ADMIN_USER_IDS = {
    1342311589298311230,
    770409748922368000,
    1011068427189887037,
}
admin_log_activation_status = {} # Will be populated in on_ready

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
        logger.info(f"Raw general_faqs_list type: {type(general_faqs_list)}, length: {len(general_faqs_list) if isinstance(general_faqs_list, list) else 'N/A'}")

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


def semantic_search_top_n_matches(query, n=2): # n is now the number of candidates to fetch
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

    actual_n = min(n, cosine_scores.numel()) # Use the passed 'n' (number of candidates)
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

async def forward_qa_to_admins(user_query: str, bot_answer: str, original_author: discord.User):
    if not bot_answer:
        return

    safe_user_query = user_query.replace("`", "'")
    safe_bot_answer = bot_answer.replace("`", "'")

    MAX_QUERY_LOG_LEN = 600
    MAX_ANSWER_LOG_LEN = 1200

    log_query_part = safe_user_query
    if len(log_query_part) > MAX_QUERY_LOG_LEN:
        log_query_part = log_query_part[:MAX_QUERY_LOG_LEN - 3] + "..."

    log_answer_part = safe_bot_answer
    if len(log_answer_part) > MAX_ANSWER_LOG_LEN:
        log_answer_part = log_answer_part[:MAX_ANSWER_LOG_LEN - 3] + "..."

    forward_message = (
        f"**User:** {original_author.name}#{original_author.discriminator} (ID: {original_author.id})\n"
        f"**Asked:** ```\n{log_query_part}\n```\n"
        f"**Bot Answered:** ```\n{log_answer_part}\n```"
    )

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
        admin_user = None # Initialize for the HTTPException block
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
                logger.error(f"Failed to send log to admin {admin_id} due to message length or formatting. Original Message Length: {len(forward_message)}. Query: '{user_query[:50]}...', Answer: '{bot_answer[:50]}...'. Code: {e.code}, Status: {e.status}")
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
                return

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
                        logger.error(f"SuggestionButton callback: interaction.message is None for custom_id {self.custom_id}.")
                except discord.NotFound:
                    logger.warning(f"SuggestionButton callback: Could not find message to edit for custom_id {self.custom_id}.")
                except discord.Forbidden:
                     logger.error(f"SuggestionButton callback: Missing permissions to edit message for custom_id {self.custom_id}.")
                except Exception as e_edit:
                    logger.error(f"SuggestionButton callback: ERROR editing message to disable buttons for custom_id {self.custom_id}. Error: {e_edit}", exc_info=True)
            else:
                logger.warning(f"SuggestionButton callback: self.view is None for custom_id {self.custom_id}.")

        except Exception as e_outer:
            logger.error(f"SuggestionButton callback: UNHANDLED EXCEPTION in callback for custom_id {self.custom_id}. Error: {e_outer}", exc_info=True)
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message("Sorry, there was an error processing your feedback.", ephemeral=True)
            except discord.errors.InteractionResponded: pass
            except Exception: logger.error(f"SuggestionButton callback: CRITICAL - Failed to even send a generic error message after outer exception.", exc_info=True)


@bot.event
async def on_ready():
    print(f'{bot.user.name} (ID: {bot.user.id}) has connected to Discord!')
    print(f'Listening for DMs. All interactions are handled as direct messages.')
    load_faq_data_from_url()

    # Initialize admin log activation status - ON by default for all admins
    global admin_log_activation_status # Ensure we are modifying the global dict
    activated_admins_count = 0
    if ADMIN_USER_IDS: # Check if there are any admin IDs defined
        for admin_id in ADMIN_USER_IDS:
            admin_log_activation_status[admin_id] = True # Default to ON
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
                extra={'details': 'Bot presence set, FAQ loaded from URL, listening for DMs.'})


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or not isinstance(message.channel, discord.DMChannel):
        return

    user_query_lower = message.content.lower().strip()
    original_message_content = message.content.strip()
    author_id = message.author.id
    author_name = str(message.author)
    log_extra_base = {'user_id': author_id, 'username': author_name, 'original_query_text': original_message_content}

    if author_id in ADMIN_USER_IDS:
        if user_query_lower == "activate logs":
            admin_log_activation_status[author_id] = True
            await message.channel.send("✅ Real-time Q&A log forwarding **activated** for you.")
            logger.info("Admin command: activate logs.", extra={**log_extra_base, 'details': f"Admin {author_name} activated logs."})
            return
        elif user_query_lower == "mute logs":
            admin_log_activation_status[author_id] = False
            await message.channel.send("🔇 Real-time Q&A log forwarding **muted** for you.")
            logger.info("Admin command: mute logs.", extra={**log_extra_base, 'details': f"Admin {author_name} muted logs."})
            return

    if is_text_empty_or_punctuation_only(original_message_content):
        logger.info("Ignoring empty or punctuation-only query.", extra={**log_extra_base, 'details': 'Query was empty or only punctuation.'})
        return

    bot_reply_text_for_forwarding = None

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
        response_to_user = ""
        pronunciation_view = discord.ui.View()
        if not word_to_pronounce_input:
            response_to_user = "Please tell me what word or phrase you want to pronounce. Usage: `pronounce [word or phrase]`"
            await message.channel.send(response_to_user)
        else:
            async with message.channel.typing():
                audio_url = await get_pronunciation_audio_url(word_to_pronounce_input)
            encoded_word_for_google = urllib.parse.quote_plus(word_to_pronounce_input)
            google_link = f"https://www.google.com/search?q=how+to+pronounce+{encoded_word_for_google}"
            youtube_link = f"https://www.youtube.com/playlist?list=PLvJSE3hDJAyN2a-i1GXZPXOpDPQZcerDc"

            response_message_lines = [f"Pronunciation resources for \"**{word_to_pronounce_input}**\":"]
            log_audio_status = "Audio not found from API."
            if audio_url:
                pronunciation_view.add_item(discord.ui.Button(label="Play Sound", style=discord.ButtonStyle.link, url=audio_url, emoji="🔊"))
                log_audio_status = f"Audio found from API: {audio_url}"
            else:
                response_message_lines.append(f"• Sorry, I couldn't find a direct audio pronunciation for \"{word_to_pronounce_input}\" from an API.")
            pronunciation_view.add_item(discord.ui.Button(label="Search on Google", style=discord.ButtonStyle.link, url=google_link))
            pronunciation_view.add_item(discord.ui.Button(label="Check LTS YouTube Playlist", style=discord.ButtonStyle.link, url=youtube_link))
            if not audio_url:
                 response_message_lines.append(f"• You can check Google/YouTube for pronunciation resources.")
            response_to_user = "\n".join(response_message_lines)

            if pronunciation_view.children:
                 await message.channel.send(response_to_user, view=pronunciation_view)
            else:
                 await message.channel.send(response_to_user)

        bot_reply_text_for_forwarding = response_to_user + (" (View with buttons also sent)" if pronunciation_view.children else "")
        logger.info(f"Pronunciation request processed for '{word_to_pronounce_input}'.", extra={**log_extra_base, 'details': f"Audio from API: {log_audio_status}. Links provided."})
        await forward_qa_to_admins(original_message_content, bot_reply_text_for_forwarding, message.author)
        return

    if user_query_lower == "list codes":
        call_codes_data = faq_data.get("call_codes", {})
        if not call_codes_data:
            reply_text = "I don't have any call codes defined at the moment."
            await message.channel.send(reply_text)
            bot_reply_text_for_forwarding = reply_text
            logger.info("Command 'list codes' - no codes found.", extra=log_extra_base)
        else:
            embed = discord.Embed(title="☎️ Call Disposition Codes", color=discord.Color.blue())
            current_field_value = ""
            field_count = 0
            MAX_FIELD_VALUE_LEN = 1020
            MAX_FIELDS = 24
            temp_forward_text_parts = ["Call Codes Sent:"]
            for code, description in call_codes_data.items():
                entry_text = f"**{code.upper()}**: {description}\n\n"
                temp_forward_text_parts.append(f"{code.upper()}: {description}")
                if len(current_field_value) + len(entry_text) > MAX_FIELD_VALUE_LEN and field_count < MAX_FIELDS:
                    embed.add_field(name=f"Codes (Part {field_count + 1})" if field_count > 0 else "Codes", value=current_field_value, inline=False)
                    current_field_value = ""; field_count += 1
                if field_count >= MAX_FIELDS:
                    logger.warning(f"List codes: Exceeded max embed fields ({MAX_FIELDS}).")
                    await message.channel.send("The list of codes is very long and might be truncated.")
                    break
                current_field_value += entry_text
            if current_field_value and field_count < MAX_FIELDS:
                embed.add_field(name=f"Codes (Part {field_count + 1})" if field_count > 0 or not embed.fields else "Codes", value=current_field_value, inline=False)
            if not embed.fields and not current_field_value:
                reply_text = "No call codes formatted. Check data."
                await message.channel.send(reply_text); bot_reply_text_for_forwarding = reply_text
                logger.info("Command 'list codes' - no codes formatted.", extra=log_extra_base)
            elif not embed.fields:
                reply_text = "Found codes, but couldn't display them. Try again/ask manager."
                await message.channel.send(reply_text); bot_reply_text_for_forwarding = reply_text
            else:
                await message.channel.send(embed=embed)
                bot_reply_text_for_forwarding = "[Bot sent 'list codes' as an embed.]\n" + "\n".join(temp_forward_text_parts)
            logger.info("Command 'list codes' processed.", extra={**log_extra_base, 'details': f"Displayed {len(call_codes_data) if call_codes_data else 0} codes."})
        await forward_qa_to_admins(original_message_content, bot_reply_text_for_forwarding, message.author)
        return

    match_define_code = re.match(r"^(?:what is|define|explain)\s+([\w\s-]+)\??$", user_query_lower)
    if match_define_code:
        code_name_query_original_case = match_define_code.group(1).strip()
        code_name_query_upper = code_name_query_original_case.upper()
        call_codes_data = faq_data.get("call_codes", {})
        found_code_key = next((k for k in call_codes_data if k.upper() == code_name_query_upper), None)
        defined_code_embed = None
        if found_code_key:
            defined_code_embed = discord.Embed(title=f"Definition: {found_code_key.upper()}", description=call_codes_data[found_code_key], color=discord.Color.purple())
            bot_reply_text_for_forwarding = f"Definition: {found_code_key.upper()}\n{call_codes_data[found_code_key]}"
            logger.info(f"Defined code '{found_code_key.upper()}' (exact match).", extra={**log_extra_base})
        else:
            best_match_code, score = process.extractOne(code_name_query_original_case, list(call_codes_data.keys()), scorer=fuzz.token_set_ratio)
            if score > 80 and best_match_code:
                defined_code_embed = discord.Embed(title=f"Definition (for '{code_name_query_original_case}'): {best_match_code.upper()}", description=call_codes_data[best_match_code], color=discord.Color.purple())
                bot_reply_text_for_forwarding = f"Definition (for '{code_name_query_original_case}'): {best_match_code.upper()}\n{call_codes_data[best_match_code]}"
                logger.info(f"Defined code '{best_match_code.upper()}' (fuzzy match).", extra={**log_extra_base, 'details': f"Score: {score}."})
        if defined_code_embed:
            await message.channel.send(embed=defined_code_embed)
            await forward_qa_to_admins(original_message_content, bot_reply_text_for_forwarding, message.author)
            return

    greetings_data = faq_data.get("greetings_and_pleasantries", [])
    for greeting_entry in greetings_data:
        keywords = greeting_entry.get("keywords", [])
        if not keywords: continue
        match_result = process.extractOne(user_query_lower, keywords, scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_MATCH_THRESHOLD_GREETINGS)
        if match_result:
            matched_keyword_from_fuzz, score = match_result
            response_type = greeting_entry.get("response_type")
            reply_text = f"Hello {message.author.mention}!"
            if response_type == "standard_greeting":
                reply_template = greeting_entry.get("greeting_reply_template", "Hello there, {user_mention}!")
                actual_greeting_cased = next((kw for kw in keywords if kw.lower() == matched_keyword_from_fuzz.lower()), matched_keyword_from_fuzz)
                reply_text = reply_template.format(actual_greeting_cased=actual_greeting_cased.capitalize(), user_mention=message.author.mention)
            elif response_type == "specific_reply":
                reply_text = greeting_entry.get("reply_text", "I acknowledge that.")
            await message.channel.send(reply_text)
            bot_reply_text_for_forwarding = reply_text
            logger.info(f"Greeting matched.", extra={**log_extra_base, 'details': f"Matched: '{matched_keyword_from_fuzz}', Score: {score}."})
            await forward_qa_to_admins(original_message_content, bot_reply_text_for_forwarding, message.author)
            return

    faq_items_original_list = faq_data.get("general_faqs", [])
    top_indices, top_scores = semantic_search_top_n_matches(user_query_lower, n=CANDIDATES_TO_FETCH_FOR_SUGGESTIONS)

    if not top_indices:
        bot_reply_text_for_forwarding = faq_data.get("fallback_message", "I'm sorry, I couldn't find an answer right now.")
        await message.channel.send(bot_reply_text_for_forwarding)
        logger.info("Unanswered query, fallback sent (no semantic hits).", extra=log_extra_base)
    else:
        primary_score = top_scores[0]
        if primary_score >= SEMANTIC_SEARCH_THRESHOLD:
            primary_faq_flat_idx = top_indices[0]
            if not (0 <= primary_faq_flat_idx < len(faq_questions) and 0 <= primary_faq_flat_idx < len(faq_original_indices)):
                logger.error(f"Semantic primary match index {primary_faq_flat_idx} out of bounds for direct hit.", extra=log_extra_base)
                bot_reply_text_for_forwarding = faq_data.get("fallback_message", "Sorry, an error occurred.")
                await message.channel.send(bot_reply_text_for_forwarding)
            else:
                original_faq_item_idx_primary = faq_original_indices[primary_faq_flat_idx]
                if not (0 <= original_faq_item_idx_primary < len(faq_items_original_list)):
                    logger.error(f"Semantic primary original item index {original_faq_item_idx_primary} out of bounds for direct hit.", extra=log_extra_base)
                    bot_reply_text_for_forwarding = faq_data.get("fallback_message", "Sorry, an error occurred.")
                    await message.channel.send(bot_reply_text_for_forwarding)
                else:
                    matched_item_primary = faq_items_original_list[original_faq_item_idx_primary]
                    answer_primary = matched_item_primary.get("answer", "Answer not available.")
                    await send_long_message(message.channel, answer_primary, view=None)
                    bot_reply_text_for_forwarding = answer_primary
                    logger.info("Semantic FAQ Direct Match.", extra={**log_extra_base, 'details': f"Score: {primary_score:.2f}, Matched Keyword Flat Idx: '{primary_faq_flat_idx}', Original FAQ Idx: {original_faq_item_idx_primary}."})

        elif primary_score >= SUGGESTION_THRESHOLD:
            suggestions_to_display = []
            shown_original_faq_indices = set()

            for i in range(len(top_indices)):
                current_flat_idx = top_indices[i]
                current_score = top_scores[i]

                if current_score < SUGGESTION_THRESHOLD: break
                if not (0 <= current_flat_idx < len(faq_original_indices) and 0 <= current_flat_idx < len(faq_questions)):
                    logger.warning(f"Suggestion selection: flat_idx {current_flat_idx} out of bounds.")
                    continue

                current_original_faq_idx = faq_original_indices[current_flat_idx]

                if current_original_faq_idx not in shown_original_faq_indices:
                    if not (0 <= current_original_faq_idx < len(faq_items_original_list)):
                        logger.warning(f"Suggestion selection: original_faq_idx {current_original_faq_idx} out of bounds.")
                        continue

                    matched_item = faq_items_original_list[current_original_faq_idx]
                    answer_text = matched_item.get("answer", "Answer not available.")
                    matched_keyword_text = faq_questions[current_flat_idx].capitalize()

                    suggestions_to_display.append({
                        "score": current_score,
                        "flat_idx": current_flat_idx,
                        "original_idx": current_original_faq_idx,
                        "answer": answer_text,
                        "keyword": matched_keyword_text
                    })
                    shown_original_faq_indices.add(current_original_faq_idx)

                if len(suggestions_to_display) >= MAX_SUGGESTIONS_TO_SHOW: break

            if suggestions_to_display:
                intro_message = f"I'm not sure I have an exact answer for '{original_message_content}', but perhaps one of these is helpful?\n"
                await send_long_message(message.channel, intro_message)
                temp_suggestion_forward_texts = [intro_message]

                for i, sugg_data in enumerate(suggestions_to_display):
                    embed_title = f"🤔 Suggestion {i+1}: Related to '{sugg_data['keyword']}'"
                    if len(embed_title) > 256:
                        embed_title = embed_title[:253] + "..."

                    embed = discord.Embed(
                        title=embed_title,
                        description=sugg_data['answer'],
                        color=discord.Color.gold()
                    )
                    btn_yes = SuggestionButton(label="✅ Helpful!", style=discord.ButtonStyle.success, custom_id=f"sugg_yes_{sugg_data['flat_idx']}_{sugg_data['original_idx']}", original_query=original_message_content, matched_keyword_text=sugg_data['keyword'], faq_item_index=sugg_data['flat_idx'], original_faq_item_idx=sugg_data['original_idx'], helpful=True)
                    btn_no = SuggestionButton(label="❌ Not quite", style=discord.ButtonStyle.danger, custom_id=f"sugg_no_{sugg_data['flat_idx']}_{sugg_data['original_idx']}", original_query=original_message_content, matched_keyword_text=sugg_data['keyword'], faq_item_index=sugg_data['flat_idx'], original_faq_item_idx=sugg_data['original_idx'], helpful=False)
                    sugg_view = discord.ui.View(timeout=300)
                    sugg_view.add_item(btn_yes); sugg_view.add_item(btn_no)
                    await message.channel.send(embed=embed, view=sugg_view)

                    temp_suggestion_forward_texts.append(f"Suggestion {i+1} (Score {sugg_data['score']:.2f}, Orig. FAQ Idx: {sugg_data['original_idx']}) '{sugg_data['keyword']}':\n{sugg_data['answer']}")
                    logger.info(f"Semantic Suggestion {i+1} offered.", extra={**log_extra_base, 'details': f"Score: {sugg_data['score']:.2f}, Keyword: '{sugg_data['keyword']}', Orig. FAQ Idx: {sugg_data['original_idx']}."})

                bot_reply_text_for_forwarding = "\n---\n".join(temp_suggestion_forward_texts)
            else:
                bot_reply_text_for_forwarding = faq_data.get("fallback_message", "I'm sorry, I couldn't find a clear answer for that.")
                await message.channel.send(bot_reply_text_for_forwarding)
                logger.info("Unanswered query, fallback sent (no distinct suggestions met threshold after filtering).", extra=log_extra_base)
        else:
            bot_reply_text_for_forwarding = faq_data.get("fallback_message", "I'm sorry, I couldn't find an answer for that.")
            await message.channel.send(bot_reply_text_for_forwarding)
            details_str = f"Primary score {primary_score:.2f} below suggestion threshold {SUGGESTION_THRESHOLD}."
            logger.info("Unanswered query, fallback sent (low semantic score).", extra={**log_extra_base, 'details': details_str})

    if bot_reply_text_for_forwarding:
        await forward_qa_to_admins(original_message_content, bot_reply_text_for_forwarding, message.author)

# --- Run the Bot ---
if __name__ == "__main__":
    # Ensure ADMIN_USER_IDS contains only integers before using it
    # This must happen before on_ready uses ADMIN_USER_IDS
    actual_admin_ids = {uid for uid in ADMIN_USER_IDS if isinstance(uid, int)}
    if len(actual_admin_ids) < len(ADMIN_USER_IDS):
        print("INFO: Placeholder or non-integer admin IDs were filtered out. Ensure all admin IDs are correct integers.")
    ADMIN_USER_IDS = actual_admin_ids # Update the global set

    if not ADMIN_USER_IDS: # Check after filtering
        print("WARNING: ADMIN_USER_IDS set is empty after filtering. Log forwarding will not work for any user by default or by command.")


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