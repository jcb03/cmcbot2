from data_processor import ChatDataProcessor
from hybrid_vector_store import HybridVectorStore
from llm_generator import HybridLLMGenerator
import gc
import time
import os
from pathlib import Path

# Load environment variables
def load_env():
    env_path = Path('.') / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.strip().split('=', 1)
                        os.environ[key.strip()] = val.strip().strip('"\'')

class SimpleHinglishChatBot:
    def __init__(self, json_file_path):
        load_env()
        
        self.json_file_path = json_file_path
        self.processor = ChatDataProcessor(json_file_path)
        self.hybrid_store = HybridVectorStore()
        self.llm = HybridLLMGenerator()
        self.setup_complete = False
        self.total_messages = 0
        
        # Check OpenAI key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("❌ OPENAI_API_KEY required! Add it to .env file.")
        
        print("✅ OpenAI API key detected - using simple hybrid retrieval!")
    
    def setup(self):
        """Simple setup - no BS"""
        print("🚀 Setting up SIMPLE Hinglish Discord Chat Bot...")
        print("🎯 RRF + BM25 + OpenAI Embeddings ONLY!")
        print("=" * 60)
        
        # Check existing setup
        if self.hybrid_store.is_setup_complete():
            print("🎉 EXISTING SETUP FOUND!")
            
            if self.hybrid_store.load_existing_indices():
                vector_count = self.hybrid_store.collection.count()
                bm25_count = len(self.hybrid_store.documents)
                
                print(f"✅ Loaded existing system:")
                print(f"📊 Vector store: {vector_count} chunks")
                print(f"📊 BM25 index: {bm25_count} documents")
                
                self.setup_complete = True
                self.total_messages = bm25_count * 6
                
                print("\n" + "=" * 60)
                print("🎉 Bot ready - NO REPROCESSING!")
                print(f"{self.hybrid_store.get_stats()}")
                print("🔍 Retrieval: SIMPLE RRF + BM25 + OpenAI")
                print("🧠 LLM: Llama 3.1 8B")
                print("=" * 60)
                return True
        
        # Process data
        print("🔧 Processing data with SIMPLE pipeline...")
        start_time = time.time()
        
        messages = self.processor.load_and_clean_data()
        self.total_messages = len(messages)
        
        if not messages:
            print("❌ No messages found!")
            return False
        
        chunks = self.processor.chunk_messages(messages, chunk_size=8)
        
        if not chunks:
            print("❌ No chunks created!")
            return False
        
        del messages
        gc.collect()
        
        # Build hybrid system
        self.hybrid_store.add_chunks(chunks, batch_size=400)
        
        del chunks
        gc.collect()
        
        setup_time = time.time() - start_time
        self.setup_complete = True
        
        print("\n" + "=" * 60)
        print("🎉 SIMPLE Chat Bot setup complete!")
        print(f"⏱️  Setup time: {setup_time:.1f} seconds")
        print(f"📊 Processed: {self.total_messages:,} messages")
        print(f"{self.hybrid_store.get_stats()}")
        print("🔍 Retrieval: SIMPLE RRF + BM25 + OpenAI")
        print("🧠 LLM: Llama 3.1 8B")
        print("=" * 60)
        return True
    
    def ask_question(self, question):
        """SIMPLE: Just ask and get answer - NO FILTERING BULLSH*T"""
        if not self.setup_complete:
            return "❌ Setup incomplete! Run setup() first."
        
        print(f"🤔 Processing: '{question}'")
        
        # Simple hybrid search
        start_search = time.time()
        results = self.hybrid_store.hybrid_search(question, n_results=4)
        search_time = time.time() - start_search
        
        if not results or not results['documents'] or not results['documents'][0]:
            return "🤔 Koi conversation nahi mila bhai! Try different keywords."
        
        # Extract results - NO FILTERING
        conversation_chunks = results['documents'][0]
        chunk_metadata = results['metadatas'][0]
        distances = results['distances'][0]
        
        # Generate response - NO BS FILTERING
        print("🧠 Generating AI response...")
        start_llm = time.time()
        
        response = self.llm.generate_response(
            question, 
            conversation_chunks, 
            chunk_metadata
        )
        
        llm_time = time.time() - start_llm
        
        # Simple performance note
        performance_note = f"\n\n⚡ *Search: {search_time:.2f}s | Generation: {llm_time:.2f}s*"
        
        return f"🤖 **AI Response:**\n\n{response}{performance_note}"
    
    def get_stats(self):
        """Simple stats"""
        if not self.setup_complete:
            return "❌ Bot not ready"
        
        return f"""📊 **SIMPLE Hinglish Chat Bot Stats:**

🔢 **Data:** {self.total_messages:,} messages processed
📊 **Storage:** {self.hybrid_store.get_stats()}

🔍 **Retrieval:** RRF + BM25 + OpenAI Embeddings ONLY
🧠 **LLM:** Llama 3.1 8B (Local)

🎯 **Features:**
   • ✅ No filtering bullsh*t - returns all results
   • ✅ No context expansion - just raw chunks  
   • ✅ Simple RRF fusion (50% vector, 50% BM25)
   • ✅ OpenAI embeddings for accuracy
   • ✅ Perfect for Hinglish queries"""

# Main execution
if __name__ == "__main__":
    # Initialize simple bot
    bot = SimpleHinglishChatBot("channel_export.json")
    
    # Setup
    if not bot.setup():
        print("❌ Setup failed!")
        exit()
    
    # Show stats
    print(f"\n{bot.get_stats()}")
    
    # Test questions
    test_questions = [
        "theabbie kon hai?",
        "bhai competitive programming ke baare mein kya discuss hua?",
        "CMC server mein kaun active hai?",
        "priyansh ke baare mein kya baat hui?",
        "job aur career advice?"
    ]
    
    print("\n" + "="*60)
    print("🧪 Testing SIMPLE hybrid responses...")
    print("="*60)
    
    for q in test_questions:
        print(f"\n❓ **Question:** {q}")
        response = bot.ask_question(q)
        print(response)
        print("-" * 40)
    
    # Interactive mode
    print("\n" + "="*60)
    print("💬 SIMPLE Interactive Mode!")
    print("Commands: 'stats', 'reset', 'quit'")
    print("="*60)
    
    while True:
        user_input = input("\n🗣️  You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Bye! Simple chat bot shutting down...")
            break
        elif user_input.lower() == 'reset':
            bot.hybrid_store.force_reset()
            print("🔄 Restart bot to reprocess data")
        elif user_input.lower() == 'stats':
            print(bot.get_stats())
        else:
            response = bot.ask_question(user_input)
            print(f"\n{response}")
