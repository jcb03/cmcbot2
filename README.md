# Install requirements
pip install sentence-transformers chromadb ollama pandas numpy rank-bm25

# Install Ollama and download model
ollama pull llama3.1:8b

# Run the bot
python hybrid_chat_bot.py