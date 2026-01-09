🩺 HBot: AI-Powered Multilingual Healthcare Assistant
HBot is a Retrieval-Augmented Generation (RAG) chatbot designed to provide reliable healthcare information. It integrates a FastAPI backend with a Telegram frontend, featuring real-time language detection and a background processing architecture to handle complex medical queries.
🚀 Features
RAG Integration: Uses Retrieval-Augmented Generation to fetch context-aware medical answers from a verified knowledge base.
Multilingual Support: Automatically detects the user's language (Hindi, English, etc.) and adjusts responses accordingly.
Asynchronous Processing: Built with python-telegram-bot and FastAPI to ensure the bot remains responsive while the AI "thinks."
Typing Indicators: Real-time feedback in Telegram to improve user experience during complex analysis.
Hybrid Server: Runs both a Web API (FastAPI) and a Telegram Bot (Polling) simultaneously.
🛠️ Tech Stack
Language: Python 3.11+
Web Framework: FastAPI & Uvicorn
Bot Framework: Python-Telegram-Bot (v20+)
AI Logic: RAG (Retrieval-Augmented Generation)
Language Detection: langdetect / fast-langdetect
Concurrency: asyncio & threading
