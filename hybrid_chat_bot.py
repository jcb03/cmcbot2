from data_processor import ChatDataProcessor
from hybrid_vector_store import HybridVectorStore
from llm_generator import HybridLLMGenerator
import gc
import time

class HybridHinglishChatBot:
    def __init__(self, json_file_path):
        self.processor = ChatDataProcessor(json_file_path)
        self.hybrid_store = HybridVectorStore()
        self.llm = HybridLLMGenerator()
        self.setup_complete = False
        self.total_messages = 0
    
    def setup(self):
        """Complete setup with hybrid retrieval system"""
        print("🚀 Setting up HYBRID Hinglish Discord Chat Bot...")
        print("🎯 Using RRF + BM25 + Dense Embeddings for maximum accuracy!")
        print("=" * 70)
        
        start_time = time.time()
        
        # Process Discord data
        messages = self.processor.load_and_clean_data()
        self.total_messages = len(messages)
        
        if not messages:
            print("❌ No messages found!")
            return False
        
        # Create optimized chunks
        chunks = self.processor.chunk_messages(messages)
        
        if not chunks:
            print("❌ No chunks created!")
            return False
        
        # Free memory
        del messages
        gc.collect()
        
        # Build hybrid indices
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
    
    def get_summary(self, method="hybrid"):
        """Get conversation summary using hybrid retrieval"""
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
        
        # Remove duplicates and limit
        unique_chunks = list(set(all_chunks))[:8]
        
        if not unique_chunks:
            return "🤔 No conversations available for summary"
        
        # Generate summary
        print("📝 Generating hybrid retrieval summary...")
        summary = self.llm.summarize_conversations(unique_chunks, "hybrid (RRF+BM25+Embeddings)")
        
        return f"📋 **Community Summary (Hybrid Analysis):**\n\n{summary}"
    
    def compare_search_methods(self, question):
        """Compare different search methods"""
        if not self.setup_complete:
            return "❌ Setup incomplete!"
        
        print(f"🔬 Comparing search methods for: '{question}'")
        
        # Hybrid search
        hybrid_results = self.hybrid_store.hybrid_search(question, n_results=3)
        
        # Vector-only search (through ChromaDB directly)
        vector_results = self.hybrid_store.collection.query(
            query_embeddings=[self.hybrid_store.embedding_model.encode([question]).tolist()[0]],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        comparison = f"""🔍 **Search Method Comparison:**

**Hybrid (RRF + BM25 + Embeddings):**
{len(hybrid_results['documents'][0]) if hybrid_results else 0} results found

**Vector Only:**  
{len(vector_results['documents'][0]) if vector_results else 0} results found

**Recommendation:** Hybrid search typically gives better results for mixed queries."""
        
        return comparison
    
    def get_stats(self):
        """Comprehensive bot statistics"""
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

🎯 **Optimizations:**
   • Perfect for: "bhai kya discuss hua tha" queries
   • Handles: Exact keywords + semantic similarity
   • Performance: Sub-second search + generation"""

# Main execution
if __name__ == "__main__":
    # Initialize hybrid bot
    bot = HybridHinglishChatBot("channel_export.json")
    
    # Setup
    print("Starting HYBRID setup - this combines the best of all worlds...")
    if not bot.setup():
        print("❌ Setup failed!")
        exit()
    
    # Show comprehensive stats
    print(f"\n{bot.get_stats()}")
    
    # Test with challenging Hinglish queries
    test_questions = [
        "bhai competitive programming ke baare mein kya discuss hua tha?",
        "CMC server mein kaun sabse active member hai?", 
        "job aur career ke baare mein koi baat hui thi?",
        "programming contest ya coding competition ki baat hui?",
        "server ke rules ya moderation ke baare mein kya hua?",
        "koi funny ya interesting conversation hui thi?"
    ]
    
    print("\n" + "="*70)
    print("🧪 Testing HYBRID AI responses with challenging Hinglish queries...")
    print("="*70)
    
    for q in test_questions:
        print(f"\n❓ **Test Question:** {q}")
        response = bot.ask_question(q)
        print(response)
        print("-" * 50)
    
    # Get comprehensive summary
    print(f"\n{bot.get_summary()}")
    
    # Interactive mode
    print("\n" + "="*70)
    print("💬 HYBRID Interactive Mode!")
    print("🎯 Ask anything in Hindi/English/Hinglish mix")
    print("🔧 Commands: 'summary', 'stats', 'compare [question]', 'quit'")
    print("="*70)
    
    while True:
        user_input = input("\n🗣️  You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Bye bhai! Hybrid chat bot shutting down...")
            break
        elif user_input.lower().startswith('compare '):
            question = user_input[8:].strip()
            response = bot.compare_search_methods(question)
        elif user_input.lower() in ['summary', 'summarize']:
            response = bot.get_summary()
        elif user_input.lower() in ['stats', 'statistics']:
            response = bot.get_stats()
        else:
            response = bot.ask_question(user_input)
        
        print(f"\n{response}")
