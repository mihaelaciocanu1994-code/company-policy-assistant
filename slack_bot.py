"""
Company Policy Assistant - Slack Bot
Answers employee policy questions and identifies the relevant policy using Vector RAG (Gemini Embedding 2 + Gemini 2.5 Flash).

Runs locally using Slack Socket Mode (no public URL or ngrok required).
"""

import os
import sys
import re
import time
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

from search_engine import (
    PolicyDatabase,
    VectorIndexSearchEngine,
    RulesBasedSearchEngine
)

# Initialize policy retrieval engines
print("Loading policy database and vector retrieval engines...")
db = PolicyDatabase("policies.csv")
vector_engine = VectorIndexSearchEngine(db)
rules_engine = RulesBasedSearchEngine(db)
print(f"Loaded {len(db.df)} company policies successfully.")

def answer_policy_question(query: str) -> dict:
    """Processes an employee policy question and returns the answer and relevant policy."""
    # Clean user query (e.g. remove Slack mention tags like <@U12345>)
    clean_query = re.sub(r"<@[A-Z0-9]+>", "", query).strip()
    if not clean_query:
        return {
            "answer": "Hello! I am your Company Policy Assistant. Please ask me any question regarding company policies.",
            "relevant_policy": "None",
            "department": "N/A",
            "category": "N/A",
            "latency": 0.0,
            "similarity": 0.0
        }

    # 1. Use the preferred LLM with Vector Index (Gemini Embedding 2 + Gemini 2.5 Flash)
    res = vector_engine.search_and_generate(clean_query, embedding_model="gemini-embedding-2")
    
    # Check if policy was identified
    policy_name = res.get("relevant_policy", "Unknown")
    policy_meta = db.find_by_title(policy_name)
    dept = policy_meta.get("department", "General") if policy_meta else "General"
    cat = policy_meta.get("category", "Operations") if policy_meta else "Operations"

    return {
        "answer": res.get("answer", "No answer found."),
        "relevant_policy": policy_name,
        "department": dept,
        "category": cat,
        "latency": res.get("response_time", 0.0),
        "similarity": res.get("similarity_score", 0.0)
    }

def format_slack_blocks(user_query: str, result: dict) -> list:
    """Formats the bot response using Slack Block Kit."""
    policy_badge = f"*{result['relevant_policy']}*"
    if result["department"] != "N/A":
        policy_badge += f" ({result['department']} Department • {result['category']})"

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"📋 *Relevant Policy:* {policy_badge}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"💬 *Answer:*\n{result['answer']}"
            }
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"⚡ _Response time: {result['latency']:.2f}s | Vector similarity: {result['similarity']:.3f} | Model: Gemini Embedding 2 + Gemini 2.5 Flash_"
                }
            ]
        },
        {"type": "divider"}
    ]
    return blocks

# -------------------------------------------------------------
# SLACK BOLT APPLICATION
# -------------------------------------------------------------
bot_token = os.getenv("SLACK_BOT_TOKEN")
app_token = os.getenv("SLACK_APP_TOKEN")

def run_test_cli(query: str):
    """Runs a simulated Slack query from the command line."""
    print(f"\n--- Simulated Slack Inquiry ---")
    print(f"User Query: {query}")
    start = time.perf_counter()
    res = answer_policy_question(query)
    print(f"Relevant Policy: {res['relevant_policy']} ({res['department']} Dept)")
    print(f"Answer: {res['answer']}")
    print(f"Latency: {res['latency']}s | Similarity: {res['similarity']}")
    print("--------------------------------\n")

if __name__ == "__main__":
    # If called with --test argument
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_q = sys.argv[2] if len(sys.argv) > 2 else "How many vacation days do I get and do they roll over?"
        run_test_cli(test_q)
        sys.exit(0)

    is_placeholder = any(p in (bot_token or "") or p in (app_token or "") for p in ["your-bot-token-here", "your-app-token-here", "placeholder"])
    is_valid_format = bot_token and app_token and bot_token.startswith("xoxb-") and app_token.startswith("xapp-") and len(bot_token) > 30 and len(app_token) > 30

    if not is_valid_format or is_placeholder:
        print("\n" + "="*65)
        print("⚠️  SLACK TOKENS NOT CONFIGURED (PLACEHOLDERS DETECTED)")
        print("="*65)
        print("Your .env file currently contains placeholder values:")
        print(f"   SLACK_BOT_TOKEN = {bot_token}")
        print(f"   SLACK_APP_TOKEN = {app_token}")
        print("\nYou must replace these with your REAL Slack tokens from api.slack.com:")
        print("1. SLACK_BOT_TOKEN:")
        print("   Go to https://api.slack.com/apps -> Your App -> OAuth & Permissions")
        print("   Copy 'Bot User OAuth Token' (starts with xoxb-..., ~55 characters)")
        print("2. SLACK_APP_TOKEN:")
        print("   Go to https://api.slack.com/apps -> Your App -> Basic Information -> App-Level Tokens")
        print("   (or Socket Mode tab) -> Copy token (starts with xapp-..., ~80 characters)")
        print("\nEdit your .env file at:")
        print(f"   {os.path.abspath('.env')}")
        print("="*65 + "\n")
        sys.exit(1)

    try:
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError:
        print("Please install slack_bolt: pip install slack_bolt")
        sys.exit(1)

    app = App(token=bot_token)

    # 1. Handle @Mentions in Channels
    @app.event("app_mention")
    def handle_mention(event, say):
        user_query = event.get("text", "")
        thread_ts = event.get("thread_ts", event.get("ts"))
        result = answer_policy_question(user_query)
        blocks = format_slack_blocks(user_query, result)
        say(blocks=blocks, text=f"Relevant Policy: {result['relevant_policy']} - {result['answer']}", thread_ts=thread_ts)

    # 2. Handle Direct Messages (DMs)
    @app.event("message")
    def handle_dm(event, say):
        # Ignore bot messages or message subtype events
        if event.get("bot_id") or event.get("subtype"):
            return
        # Only reply if direct message channel (channel type 'im')
        if event.get("channel_type") == "im":
            user_query = event.get("text", "")
            result = answer_policy_question(user_query)
            blocks = format_slack_blocks(user_query, result)
            say(blocks=blocks, text=f"Relevant Policy: {result['relevant_policy']} - {result['answer']}")

    # 3. Optional Slash Command /policy
    @app.command("/policy")
    def handle_slash_policy(ack, respond, command):
        ack()
        query = command.get("text", "")
        if not query:
            respond("Please provide a question, e.g. `/policy How many vacation days do I get?`")
            return
        result = answer_policy_question(query)
        blocks = format_slack_blocks(query, result)
        respond(blocks=blocks, text=f"Relevant Policy: {result['relevant_policy']} - {result['answer']}")

    print("\n⚡ Starting Slack Bot in Socket Mode...")
    print("🤖 Listening for @mentions and direct messages...")
    handler = SocketModeHandler(app, app_token)
    handler.start()
