# rag_backend.py - UPDATED WITH MODERN LANGCHAIN IMPORTS

# --- UPDATED IMPORTS ---
from langchain_ollama import OllamaLLM as Ollama # Import from the new package
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma # Import from the new package
from langchain_huggingface import HuggingFaceEmbeddings # Import from the new package
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. Initialize models (LLM only at module level) ---
llm = Ollama(model="llama3.1")

# Lazy initialization of embeddings and vectorstore
embeddings = None
vectorstore = None
retriever = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
    return embeddings

def get_retriever():
    global retriever
    if retriever is None:
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=get_embeddings())
        retriever = vectorstore.as_retriever()
    return retriever

# --- 2. Create language-specific prompt templates ---
PROMPT_TEMPLATES = {
    "en": """
You are a helpful and respectful AI assistant for public health queries in Odisha, India.
Your answers should be based only on the context provided below.
If the context does not contain the answer, state clearly that you do not have enough information.
Provide answers that are concise, easy to understand, and in ENGLISH.

Context:
{context}

Question:
{input}

Answer in English:
""",
    "or": """
ଆପଣ ଓଡ଼ିଶା, ଭାରତରେ ଜନସ୍ୱାସ୍ଥ୍ୟ ସମ୍ବନ୍ଧୀୟ ପ୍ରଶ୍ନ ପାଇଁ ଜଣେ ସାହାଯ୍ୟକାରୀ AI ଆସିଷ୍ଟାଣ୍ଟ ଅଟନ୍ତି।
ଆପଣଙ୍କର ଉତ୍ତର କେବଳ ନିମ୍ନରେ ଦିଆଯାଇଥିବା ପ୍ରସଙ୍ଗ ଉପରେ ଆଧାରିତ ହେବା ଉଚିତ୍।
ଯଦି ପ୍ରସଙ୍ଗରେ ଉତ୍ତର ନାହିଁ, ତେବେ ସ୍ପଷ୍ଟ ଭାବରେ କୁହନ୍ତୁ ଯେ ଆପଣଙ୍କ ପାଖରେ ଯଥେଷ୍ଟ ସୂଚନା ନାହିଁ।
ଉତ୍ତର ସଂକ୍ଷିପ୍ତ, ସହଜରେ ବୁଝିହେବା ଭଳି ଏବଂ ଓଡ଼ିଆରେ ଦିଅନ୍ତୁ।

ପ୍ରସଙ୍ଗ:
{context}

ପ୍ରଶ୍ନ:
{input}

ଓଡ଼ିଆରେ ଉତ୍ତର:
""",
    "hi": """
आप ओडिशा, भारत में सार्वजनिक स्वास्थ्य संबंधी प्रश्नों के लिए एक सहायक एआई असिस्टेंट हैं।
आपके उत्तर केवल नीचे दिए गए संदर्भ पर आधारित होने चाहिए।
यदि संदर्भ में उत्तर नहीं है, तो स्पष्ट रूप से बताएं कि आपके पास पर्याप्त जानकारी नहीं है।
संक्षिप्त, समझने में आसान और हिंदी में उत्तर प्रदान करें।

संदर्भ:
{context}

प्रश्न:
{input}

हिंदी में उत्तर:
"""
}

def get_rag_response(query: str, language_code: str):
    """
    Gets a response from the RAG chain, dynamically selecting the prompt
    based on the detected language.
    """
    print(f"Generating response for language: {language_code}")

    prompt_template_str = PROMPT_TEMPLATES.get(language_code, PROMPT_TEMPLATES["en"])

    prompt = ChatPromptTemplate.from_template(prompt_template_str)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(get_retriever(), question_answer_chain)

    response = rag_chain.invoke({"input": query})
    return response["answer"]
