from data_processor import ChatDataProcessor
from hybrid_vector_store import HybridVectorStore
from llm_generator import HybridLLMGenerator
import gc
import time

class HybridHinglishChatBot:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.processor = ChatDataProcessor(json_file_path)
        self.hybrid_store = HybridVectorStore()
        self.llm = HybridLLMGenerator()
        self.setup_complete = False
        self.total_messages = 0
    
    def setup(self):
        """Smart setup - only process if not already done"""
        print("🚀 Setting up HYBRID Hinglish Discord Chat Bot...")
        print("🎯 Using RRF + BM25 + Dense Embeddings for maximum accuracy!")
        print("=" * 70)
        
        # CHECK IF ALREADY SETUP
        if self.hybrid_store.is_setup_complete():
            print("🎉 EXISTING SETUP FOUND!")
            print("📂 Loading pre-built indices...")
            
            # Load existing BM25 index
            if self.hybrid_store.load_existing_indices():
                vector_count = self.hybrid_store.collection.count()
                bm25_count = len(self.hybrid_store.documents)
                
                print(f"✅ Loaded existing hybrid system:")
                print(f"📊 Vector store: {vector_count} chunks")
                print(f"📊 BM25 index: {bm25_count} documents")
                
                self.setup_complete = True
                self.total_messages = bm25_count * 5  # Estimate
                
                print("\n" + "=" * 70)
                print("🎉 Bot ready from existing data - NO REPROCESSING NEEDED!")
                print(f"{self.hybrid_store.get_stats()}")
                print("🔍 Retrieval: RRF + BM25 + LaBSE Embeddings")
                print("🧠 LLM: Llama 3.1 8B (Local)")
                print("=" * 70)
                return True
        
        # IF NOT SETUP, DO FULL PROCESSING
        print("🔧 No existing setup found - processing data...")
        start_time = time.time()
        
        # Process Discord data
        messages = self.processor.load_and_clean_data()
        self.total_messages = len(messages)
        
        if not messages:
            print("❌ No messages found!")
            return False
        
        # Create chunks
        chunks = self.processor.chunk_messages(messages)
        
        if not chunks:
            print("❌ No chunks created!")
            return False
        
        # Free memory
        del messages
        gc.collect()
        
        # Build hybrid indices (first time)
        self.hybrid_store.add_chunks(chunks)
        
        # Free memory
        del chunks
        gc.collect()
        
        setup_time = time.time() - start_time
        
        self.setup_complete = True
        print("\n" + "=" * 70)
        print("🎉 HYBRID Chat Bot setup complete!")
        print(f"⏱️  Setup time: {setup_time:.1f} seconds")
        print(f"📊 Processed: {self.total_messages:,} messages")
        print(f"{self.hybrid_store.get_stats()}")
        print("🔍 Retrieval: RRF + BM25 + LaBSE Embeddings")
        print("🧠 LLM: Llama 3.1 8B (Local)")
        print("💬 Optimized for: Hinglish queries like 'bhai kya discuss hua tha'")
        print("=" * 70)
        return True
    
    def ask_question(self, question, search_mode="hybrid"):
        """Ask question with hybrid retrieval"""
        if not self.setup_complete:
            return "❌ Setup incomplete! Run setup() first."
        
        print(f"🤔 Processing: '{question}'")
        
        # Perform hybrid search
        start_search = time.time()
        results = self.hybrid_store.hybrid_search(question, n_results=4)
        search_time = time.time() - start_search
        
        if not results or not results['documents'] or not results['documents'][0]:
            return "🤔 Koi relevant conversation nahi mila bhai! Try different keywords."
        
        # Extract results
        conversation_chunks = results['documents'][0]
        chunk_metadata = results['metadatas'][0]
        
        # Generate intelligent response
        print("🧠 Generating AI response...")
        start_llm = time.time()
        
        retrieval_info = f"Vector+BM25+RRF, {len(conversation_chunks)} chunks, {search_time:.2f}s search"
        response = self.llm.generate_response(
            question, 
            conversation_chunks, 
            chunk_metadata,
            retrieval_info
        )
        
        llm_time = time.time() - start_llm
        
        # Add performance stats
        performance_note = f"\n\n⚡ *Search: {search_time:.2f}s | Generation: {llm_time:.2f}s*"
        
        return f"🤖 **AI Response (Hybrid Retrieval):**\n\n{response}{performance_note}"
    
    def get_summary(self):
        """Get conversation summary"""
        if not self.setup_complete:
            return "❌ Setup incomplete!"
        
        # Get diverse sample using hybrid search
        sample_queries = [
            "programming discussion coding",
            "server community members", 
            "job work career discussion",
            "competitive programming contest"
        ]
        
        all_chunks = []
        for query in sample_queries:
            results = self.hybrid_store.hybrid_search(query, n_results=2)
            if results and results['documents']:
                all_chunks.extend(results['documents'][0])
        
        # Remove duplicates
        unique_chunks = list(set(all_chunks))[:8]
        
        if not unique_chunks:
            return "🤔 No conversations available for summary"
        
        print("📝 Generating summary...")
        summary = self.llm.summarize_conversations(unique_chunks, "hybrid (RRF+BM25+Embeddings)")
        
        return f"📋 **Community Summary (Hybrid Analysis):**\n\n{summary}"
    
    def reset_setup(self):
        """Force reset - delete all indices to reprocess"""
        import os
        import shutil
        
        files_to_remove = [
            "./setup_complete.flag",
            "./bm25_index.pkl",
            "./hybrid_chat_db"
        ]
        
        for path in files_to_remove:
            try:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    print(f"🗑️  Removed: {path}")
            except Exception as e:
                print(f"❌ Error removing {path}: {e}")
        
        print("✅ Setup reset complete - next run will reprocess data")
    
    def get_stats(self):
        """Get bot statistics"""
        if not self.setup_complete:
            return "❌ Bot not ready"
        
        return f"""📊 **Hybrid Hinglish Chat Bot Stats:**

🔢 **Data:**
   • Messages processed: {self.total_messages:,}
   • {self.hybrid_store.get_stats()}

🔍 **Retrieval System:**
   • Method: RRF + BM25 + Dense Embeddings
   • Embedding Model: LaBSE (Multilingual)
   • Keyword Search: BM25 with Hinglish tokenization
   • Fusion: Reciprocal Rank Fusion (RRF)

🧠 **Generation:**
   • LLM: Llama 3.1 8B (Local)
   • Context: Up to 4 conversation chunks
   • Style: Natural Hinglish responses

🎯 **Performance:**
   • ✅ Persistent storage - no reprocessing needed
   • ✅ Sub-second search + generation
   • ✅ Perfect for mixed Hindi-English queries"""

# Main execution
if __name__ == "__main__":
    # Initialize hybrid bot
    bot = HybridHinglishChatBot("channel_export.json")
    
    # Smart setup - will skip if already done
    print("Checking for existing setup...")
    if not bot.setup():
        print("❌ Setup failed!")
        exit()
    
    # Show stats
    print(f"\n{bot.get_stats()}")
    
    # Interactive mode
    print("\n" + "="*70)
    print("💬 HYBRID Interactive Mode!")
    print("🎯 Ask anything in Hindi/English/Hinglish mix")
    print("🔧 Commands: 'summary', 'stats', 'reset' (force reprocess), 'quit'")
    print("="*70)
    
    while True:
        user_input = input("\n🗣️  You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Bye bhai! Hybrid chat bot shutting down...")
            break
        elif user_input.lower() == 'reset':
            bot.reset_setup()
            print("🔄 Restart the bot to reprocess data from scratch")
        elif user_input.lower() in ['summary', 'summarize']:
            response = bot.get_summary()
        elif user_input.lower() in ['stats', 'statistics']:
            response = bot.get_stats()
        else:
            response = bot.ask_question(user_input)
        
        print(f"\n{response}")
