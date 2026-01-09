import asyncio
import threading
import uvicorn
import traceback  # Added for debugging
from fastapi import FastAPI
from telegram import Update, constants
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

# --- 1. RAG BACKEND HANDLING ---
try:
    from rag_backend import get_rag_response
except ImportError:
    print("CRITICAL: rag_backend.py not found!")
    def get_rag_response(query, lang):
        return "Backend file missing!"

# --- 2. CONFIGURATION ---
TELEGRAM_BOT_TOKEN = ""

# --- 3. HELPER: TYPING INDICATOR ---
async def send_typing_indicator(context, chat_id, stop_event):
    while not stop_event.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
            await asyncio.sleep(4)
        except: break

# --- 4. THE BOT LOGIC ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    query = update.message.text
    chat_id = update.effective_chat.id

    # Start typing indicator
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(send_typing_indicator(context, chat_id, stop_typing))

    try:
        # Step A: Simple Language Detection (more stable)
        from langdetect import detect as det_lang
        try:
            lang_code = det_lang(query)
        except:
            lang_code = "en"

        # Step B: Call your RAG function
        # IMPORTANT: Check if get_rag_response takes (query, lang_code)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, get_rag_response, query, lang_code)

    except Exception as e:
        # THIS PRINTS THE REAL ERROR TO YOUR TERMINAL
        print("\n" + "="*50)
        print("ERROR DETECTED IN HANDLE_MESSAGE:")
        print(traceback.format_exc()) 
        print("="*50 + "\n")
        response = "I encountered an error while analyzing your message."
    
    finally:
        stop_typing.set()
        await typing_task
        await context.bot.send_message(chat_id=chat_id, text=response)

# --- 5. FASTAPI SETUP ---
app = FastAPI()
@app.get("/")
async def root(): return {"status": "online"}

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()

    t_request = HTTPXRequest(connect_timeout=30, read_timeout=30)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).request(t_request).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Debug Bot is ONLINE. Watch this terminal for errors!")
    application.run_polling(drop_pending_updates=True)