import discord
import os
import json
from dotenv import load_dotenv
import string
from thefuzz import process, fuzz
import logging
from sentence_transformers import SentenceTransformer, util
import torch
import re # Added for specific command matching

# --- Configuration ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
FAQ_FILE = "faq_data.json"
FUZZY_MATCH_THRESHOLD_GREETINGS = 75
LOG_FILE = "unanswered_queries.log"
SEMANTIC_SEARCH_THRESHOLD = 0.50 # Main threshold for a confident answer
SUGGESTION_THRESHOLD = 0.40    # Lower threshold to offer a suggestion

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
    global faq_embeddings, faq_questions, model
    try:
        if model is None:
            logger.info("Loading sentence transformer model ('all-MiniLM-L6-v2')...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded.")
        
        general_faqs_list = faq_data.get("general_faqs", [])
        if not isinstance(general_faqs_list, list):
            logger.error(f"general_faqs in {FAQ_FILE} is not a list. Semantic search may not work correctly.")
            general_faqs_list = []

        current_faq_questions = []
        for item in general_faqs_list:
            if isinstance(item, dict) and item.get("keywords"):
                keywords_data = item["keywords"]
                if isinstance(keywords_data, list) and keywords_data:
                    current_faq_questions.append(keywords_data[0]) 
                elif isinstance(keywords_data, str) and keywords_data.strip():
                    current_faq_questions.append(keywords_data)
        
        faq_questions = current_faq_questions

        if not model:
            logger.error("Model is not loaded. Cannot build embeddings.")
            faq_embeddings = [] 
            return

        if faq_questions:
            logger.info(f"Encoding {len(faq_questions)} FAQ questions...")
            faq_embeddings = model.encode(faq_questions, convert_to_tensor=True) 
            logger.info(f"FAQ embeddings created with shape: {faq_embeddings.shape}")
        else:
            logger.info("No FAQ questions to encode. Creating empty embeddings tensor.")
            embedding_dim = model.get_sentence_embedding_dimension()
            device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            faq_embeddings = torch.empty((0, embedding_dim), dtype=torch.float, device=device)
            logger.info(f"Empty FAQ embeddings created with shape: {faq_embeddings.shape}")

    except Exception as e:
        logger.exception("Error building semantic embeddings:")
        faq_questions = [] 
        if model:
            try:
                embedding_dim = model.get_sentence_embedding_dimension()
                device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
                faq_embeddings = torch.empty((0, embedding_dim), dtype=torch.float, device=device)
            except Exception: faq_embeddings = [] 
        else: faq_embeddings = []

def is_text_empty_or_punctuation_only(text):
    return not text or all(char in string.punctuation or char.isspace() for char in text)

def semantic_search_best_match(query): # Removed threshold from here
    if model is None:
        logger.warning("Semantic search: Model not ready.")
        return None, 0.0 # Return None index, 0.0 score
    
    if not isinstance(faq_embeddings, torch.Tensor) or faq_embeddings.shape[0] == 0:
        if isinstance(faq_embeddings, list) and not faq_embeddings:
             logger.warning("Semantic search: FAQ embeddings list is empty.")
        elif isinstance(faq_embeddings, torch.Tensor) and faq_embeddings.shape[0] == 0:
             logger.warning("Semantic search: FAQ embeddings tensor is empty.")
        else:
            logger.error(f"Semantic search: FAQ embeddings unexpected type or state: {type(faq_embeddings)}")
        return None, 0.0
        
    query_embedding = model.encode(query, convert_to_tensor=True)
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
    
    # Now, always return the best index and score found.
    # The decision to use it as a direct answer or suggestion happens in on_message.
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
    await bot.change_presence(activity=discord.Game(name="DM me for help!"))
    logger.info("Bot started and ready.", extra={'user_id': 'SYSTEM', 'username': 'SYSTEM', 'extra_info': 'Bot Ready'})


@bot.event
async def on_message(message):
    if message.author == bot.user or not isinstance(message.channel, discord.DMChannel):
        return

    user_query_lower = message.content.lower().strip()
    original_message_content = message.content.strip() 
    log_user_info = {'user_id': message.author.id, 'username': str(message.author)}

    if is_text_empty_or_punctuation_only(original_message_content):
        logger.info(f"Ignoring empty/punctuation-only query: '{original_message_content}'", extra={**log_user_info, 'extra_info': 'Empty query ignored'})
        return

    # --- Greetings Handler ---
    greetings_data = faq_data.get("greetings_and_pleasantries", [])
    for greeting_entry in greetings_data:
        keywords = greeting_entry.get("keywords", [])
        if not keywords: continue
        match_result = process.extractOne(user_query_lower, keywords, scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_MATCH_THRESHOLD_GREETINGS)
        if match_result:
            matched_keyword, score = match_result 
            response_type = greeting_entry.get("response_type")
            if response_type == "standard_greeting":
                reply_template = greeting_entry.get("greeting_reply_template", "Hello there, {user_mention}!")
                actual_greeting_cased = matched_keyword 
                for kw_original in keywords:
                    if kw_original.lower() == matched_keyword.lower():
                        actual_greeting_cased = kw_original; break
                reply = reply_template.format(actual_greeting_cased=actual_greeting_cased.capitalize(), user_mention=message.author.mention)
                await message.channel.send(reply)
            elif response_type == "specific_reply":
                await message.channel.send(greeting_entry.get("reply_text", "I acknowledge that."))
            else: await message.channel.send(f"Hello {message.author.mention}!")
            logger.info(f"Greeting matched. Keyword: '{matched_keyword}' (Score: {score}). Query: '{original_message_content}'", extra={**log_user_info, 'extra_info': 'Greeting answered'})
            return

    # --- "List Codes" Command Handler ---
    if user_query_lower == "list codes":
        # ... (list codes logic - unchanged from your version) ...
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
    # Example: "define WRONG NUMBER" or "what is answering machine"
    match_define_code = re.match(r"^(?:what is|define|explain)\s+([\w\s-]+)\??$", user_query_lower)
    if match_define_code:
        code_name_query_original = match_define_code.group(1).strip()
        code_name_query_upper = code_name_query_original.upper()
        call_codes_data = faq_data.get("call_codes", {})
        
        if code_name_query_upper in call_codes_data:
            description = call_codes_data[code_name_query_upper]
            embed = discord.Embed(title=f"Definition: {code_name_query_upper}", description=description, color=discord.Color.purple())
            await message.channel.send(embed=embed)
            logger.info(f"Defined code '{code_name_query_upper}' (exact). Query: '{original_message_content}'", extra=log_user_info)
            return
        else:
            # Fuzzy match on keys if no exact uppercase match
            best_match_code, score = process.extractOne(code_name_query_original, call_codes_data.keys(), scorer=fuzz.token_set_ratio) # Use original query for fuzzy
            if score > 80: # Threshold for fuzzy code match
                description = call_codes_data[best_match_code] # best_match_code is already in correct case from dict keys
                embed = discord.Embed(title=f"Definition (for '{code_name_query_original}'): {best_match_code}", description=description, color=discord.Color.purple())
                await message.channel.send(embed=embed)
                logger.info(f"Defined code '{best_match_code}' (fuzzy, score {score}). Query: '{original_message_content}'", extra=log_user_info)
                return


    # --- Semantic Matching (as fallback) ---
    faq_items = faq_data.get("general_faqs", [])
    if not isinstance(faq_items, list): 
        logger.error("general_faqs is not a list. Cannot perform semantic search.")
        faq_items = [] 

    semantic_idx, semantic_score = semantic_search_best_match(user_query_lower)

    if semantic_idx is not None and 0 <= semantic_idx < len(faq_items):
        matched_item = faq_items[semantic_idx]
        answer = matched_item.get("answer", "Sorry, the answer for this item is missing.")
        
        primary_keyword_for_title = "Related Information"
        if isinstance(matched_item.get("keywords"), list) and matched_item["keywords"]:
            primary_keyword_for_title = matched_item['keywords'][0].capitalize()
        elif isinstance(matched_item.get("keywords"), str):
             primary_keyword_for_title = matched_item.get("keywords").capitalize()

        log_extra_info = ""
        response_message_prefix = ""

        if semantic_score >= SEMANTIC_SEARCH_THRESHOLD:
            log_extra_info = f"Semantic FAQ Direct. Score: {semantic_score:.2f}"
            response_message_prefix = f"💡 **{primary_keyword_for_title}**\n\n"
            embed_title = f"💡 {primary_keyword_for_title}"
        elif semantic_score >= SUGGESTION_THRESHOLD:
            log_extra_info = f"Semantic FAQ Suggestion. Score: {semantic_score:.2f}"
            response_message_prefix = (
                f"I'm not sure I have an exact answer for that, but perhaps this is related to **'{primary_keyword_for_title}'**?\n\n"
                f"Here's some info on that:\n\n"
            )
            embed_title = f"🤔 Related to: {primary_keyword_for_title}"
        
        if response_message_prefix: # If either direct match or suggestion
            faq_embed = discord.Embed(title=embed_title, description=answer, color=discord.Color.green())
            
            full_response_text_for_long_send = response_message_prefix + answer
            
            if len(answer) > 4096: # Embed description limit
                 await send_long_message(message.channel, full_response_text_for_long_send)
            else:
                # Send prefix separately if it exists and won't fit with embed description logic
                if semantic_score >= SUGGESTION_THRESHOLD and semantic_score < SEMANTIC_SEARCH_THRESHOLD:
                     await message.channel.send(response_message_prefix.split("\n\nHere's some info on that:\n\n")[0]) # Send suggestion intro
                     await message.channel.send(embed=faq_embed) # Send embed with answer
                else: # Direct match
                    # For direct matches, the title implies the context, prefix might be redundant with embed
                    await message.channel.send(embed=faq_embed)


            matched_question_log = "N/A"
            if isinstance(matched_item.get("keywords"), list) and matched_item["keywords"]:
                matched_question_log = matched_item['keywords'][0]
            elif isinstance(matched_item.get("keywords"), str):
                matched_question_log = matched_item.get("keywords")
            
            logger.info(f"{log_extra_info}. Matched Q: '{matched_question_log}'. User Query: '{original_message_content}'",
                        extra={**log_user_info, 'extra_info': 'Semantic match processed'})
            return

    # --- Fallback if nothing else matched or score too low for suggestion ---
    fallback_message_template = faq_data.get("fallback_message", "I'm sorry, I couldn't find an answer for that right now. Please ask your manager.")
    await message.channel.send(fallback_message_template)
    logger.info(f"Unanswered query. Fallback triggered. Last semantic score: {semantic_score:.2f}. Query: '{original_message_content}'", extra={**log_user_info, 'extra_info': 'Fallback - No answer/suggestion'})

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