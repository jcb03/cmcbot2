import os
from pathlib import Path

# Load environment variables from .env file
def load_env():
    env_path = Path('.') / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.strip().split('=', 1)
                        os.environ[key.strip()] = val.strip().strip('"\'')

class HybridLLMGenerator:
    def __init__(self, model_name="llama3.1:8b"):
        load_env()  # Load .env variables
        
        self.model_name = model_name
        print(f"🧠 Initializing LLM: {model_name}")
        
        try:
            import ollama
            # Test model availability
            test_response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'Test'}]
            )
            print("✅ LLM ready for FIXED responses with OpenAI embeddings!")
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            print("Run: ollama pull llama3.1:8b")
            raise e
    
    def generate_response(self, question, conversation_chunks, chunk_metadata, retrieval_info=None):
        """Generate precise response with FIXED hybrid retrieval context"""
        
        # Prepare enhanced context
        context_parts = []
        for i, (chunk, metadata) in enumerate(zip(conversation_chunks, chunk_metadata)):
            try:
                from datetime import datetime
                time_str = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00')).strftime("%d %b %Y")
            except:
                time_str = "Unknown date"
            
            users = metadata.get('users', 'Unknown users')
            
            context_parts.append(f"""
CONVERSATION {i+1} ({time_str}) - Users: {users}
Content:
{chunk}
""")
        
        context = '\n'.join(context_parts)
        
        # Enhanced prompt for better responses with OpenAI embeddings
        prompt = f"""You are analyzing Discord conversations from a competitive programming community (CMC server). These conversations were retrieved using advanced hybrid search with high-quality embeddings.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the provided conversations - no speculation
2. Use natural Hindi-English mix (Hinglish) like the community speaks
3. If the conversations don't directly answer the question, say "yeh specific info available nahi hai in these conversations"
4. Be conversational but accurate - use "bhai", "yaar" when appropriate
5. Mention specific users, dates, or details when relevant
6. Keep response focused and under 300 words
7. If no relevant info found, don't make up answers
8. Use the retrieval info to improve your answer if provided

Retrieved Conversations:
{context}

User Question: {question}

Response:"""

        try:
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.4,    # Very low for factual accuracy
                    'max_tokens': 400,     # Focused responses
                    'top_p': 0.85,         # Controlled sampling
                    'top_k': 20,           # Limited vocabulary
                    'repeat_penalty': 1.1   # Prevent repetition
                }
            )
            
            llm_response = response['message']['content']
            
            # Add retrieval quality info
            if retrieval_info:
                footer = f"\n\n📊 *{retrieval_info}*"
                llm_response += footer
            
            return llm_response
        
        except Exception as e:
            return f"❌ LLM Generation Error: {str(e)}"
    
    def summarize_conversations(self, conversation_chunks, retrieval_method="fixed_hybrid"):
        """Generate focused summary"""
        sample_chunks = conversation_chunks[:4]
        combined_text = '\n\n'.join(sample_chunks)
        
        prompt = f"""Analyze these Discord conversations from a competitive programming community and create a concise summary.

Conversations (Retrieved via {retrieval_method}):
{combined_text}

Create a focused summary covering:
1. Main topics discussed (be specific)
2. Key participants and their contributions
3. Important decisions or announcements
4. Technical discussions (algorithms, coding problems, etc.)
5. Use the retrieval info to improve your answer if provided

Write in natural Hinglish style, be precise, under 150 words:"""

        try:
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.4,
                    'max_tokens': 250,
                    'top_p': 0.8
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"❌ Summary Error: {str(e)}"
