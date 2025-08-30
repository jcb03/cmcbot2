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
        """Simple setup with hybrid vector store"""
        print("🚀 Setting up CMC Lore Bot...")
        print("🎯 RRF + BM25 + OpenAI Embeddings")
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
                print("🎉 CMC Lore Bot ready - NO REPROCESSING REQUIRED!")
                print(f"{self.hybrid_store.get_stats()}")
                print("🔍 Retrieval: RRF + BM25 + OpenAI")
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
        print("🎉 CMC Lore Bot setup complete!")
        print(f"⏱️  Setup time: {setup_time:.1f} seconds")
        print(f"📊 Processed: {self.total_messages:,} messages")
        print(f"{self.hybrid_store.get_stats()}")
        print("🔍 Retrieval: SIMPLE RRF + BM25 + OpenAI")
        print("🧠 LLM: Llama 3.1 8B")
        print("=" * 60)
        return True
    
    def ask_question(self, question):
        """SIMPLE: Just ask and get answer"""
        if not self.setup_complete:
            return "❌ Setup incomplete! Run setup() first."
        
        print(f"🤔 Processing: '{question}'")
        
        # Simple hybrid search
        start_search = time.time()
        results = self.hybrid_store.hybrid_search(question, n_results=4)
        search_time = time.time() - start_search
        
        if not results or not results['documents'] or not results['documents'][0]:
            return "🤔 Koi conversation nahi mila bhai! Try different keywords."
        
        # Extract results 
        conversation_chunks = results['documents'][0]
        chunk_metadata = results['metadatas'][0]
        distances = results['distances'][0]
        
        # Generate response 
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
            return "❌ CMC Lore Bot not ready"
        
        return f"""📊 CMC Lore Bot Stats:        

🔢 Data: {self.total_messages:,} messages processed
📊 Storage: {self.hybrid_store.get_stats()}

🔍 Retrieval: RRF + BM25 + OpenAI Embeddings
🧠 LLM: Llama 3.1 8B (Local)

🎯 Features:
   • ✅ CMC Lore bot can summarize or answer queries related to CMC Discord chats
   • ✅ RRF fusion (50% vector, 50% BM25)
   • ✅ OpenAI embeddings + Llama 3.1 8B
   • ✅ Perfect for Hinglish queries"""

    def lookup_document(self, doc_id):
        """Lookup document by ID with detailed info"""
        if not self.setup_complete:
            return "❌ Bot not ready!"
    
        doc_info = self.hybrid_store.get_doc_by_id(doc_id)
    
        if doc_info is None:
            return f"❌ Document ID {doc_id} not found! Valid range: 0-{len(self.hybrid_store.documents)-1}"
    
        result = f"""📄 Document ID {doc_info['doc_id']}:

    👥 Users: {doc_info['users']}
    📅 Time: {doc_info['start_time'][:10]}
    💬 Messages: {doc_info['message_count']}
    🔖 Chunk ID: {doc_info['chunk_id']}

    📝 Content:
    {doc_info['content']}
    """
    
        return result  
    
    def get_document_by_id(self, doc_id):
        """Simple method to get just document text"""
        return self.hybrid_store.get_doc_content_only(doc_id)

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
    
    # Interactive mode
    print("\n" + "="*60)
    print("💬 Interactive Mode!")
    print("Commands: 'stats', 'reset', 'quit', 'doc <doc_id>', 'lookup <doc_id>'")
    print("="*60)
    
    while True:
        user_input = input("\n🗣️  You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Bye! CMC Lore bot shutting down...")
            break
        elif user_input.lower() == 'reset':
            bot.hybrid_store.force_reset()
            print("🔄 Restart bot to reprocess data")
        elif user_input.lower() == 'stats':
            print(bot.get_stats())
        elif user_input.lower().startswith('lookup '):
            try:
                doc_id = int(user_input.split(" ", 1)[1].strip())
                result = bot.lookup_document(doc_id)
                print(result)
            except (ValueError, IndexError):
                print("❌ Usage: lookup <doc_id>")
                print("Example: lookup 12345")
        elif user_input.lower().startswith('doc '):
            try:
                doc_id = int(user_input.split(" ", 1)[1].strip())
                content = bot.get_document_by_id(doc_id)
                if content:
                    print(f"\n📄 Document {doc_id}:\n{content}")
                else:
                    print(f"❌ Document {doc_id} not found!")
            except (ValueError, IndexError):
                print("❌ Usage: doc <doc_id>")    
        else:
            response = bot.ask_question(user_input)
            print(f"\n{response}")
