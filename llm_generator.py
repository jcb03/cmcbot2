import ollama
from datetime import datetime

class HybridLLMGenerator:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        print(f"🧠 Initializing LLM: {model_name}")
        
        try:
            # Test model availability
            test_response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'Test'}]
            )
            print("✅ LLM ready for hybrid retrieval responses!")
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            print("Run: ollama pull llama3.1:8b")
            raise e
    
    def generate_response(self, question, conversation_chunks, chunk_metadata, retrieval_info=None):
        """Generate response using hybrid retrieval context"""
        
        # Prepare enriched context
        context_parts = []
        for i, (chunk, metadata) in enumerate(zip(conversation_chunks, chunk_metadata)):
            # Format timestamp
            try:
                time_str = datetime.fromisoformat(metadata['start_time'].replace('Z', '+00:00')).strftime("%d %b %Y")
            except:
                time_str = "Unknown date"
            
            users = metadata.get('users', 'Unknown users')
            context_parts.append(f"Conversation {i+1} ({time_str}) - Users: {users}:\n{chunk}\n")
        
        context = '\n'.join(context_parts)
        
        # Enhanced prompt for hybrid retrieval
        prompt = f"""You are analyzing Discord conversations from a competitive programming community (CMC server). The conversations are retrieved using advanced hybrid search (both semantic similarity and keyword matching).

Retrieved Conversations:
{context}

User Question: {question}

Instructions:
1. Answer in natural Hindi-English mix (Hinglish) like the community speaks
2. Be conversational and use words like "bhai", "yaar" when appropriate  
3. Reference specific users, dates, or details from the conversations when relevant
4. If multiple conversations discuss the topic, synthesize the information
5. If information is incomplete, mention "zyada detail available nahi hai"
6. Keep response under 250 words
7. Be accurate - only use information from the provided conversations

Response:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.6,  # Slightly more creative for conversational tone
                    'max_tokens': 350,
                    'top_p': 0.9
                }
            )
            
            llm_response = response['message']['content']
            
            # Add retrieval info if available
            if retrieval_info:
                footer = f"\n\n📊 *Retrieved using hybrid search: {retrieval_info}*"
                llm_response += footer
            
            return llm_response
        
        except Exception as e:
            return f"❌ LLM Generation Error: {str(e)}"
    
    def summarize_conversations(self, conversation_chunks, retrieval_method="hybrid"):
        """Generate summary using hybrid retrieval"""
        
        # Limit chunks for processing
        sample_chunks = conversation_chunks[:6]
        combined_text = '\n\n'.join(sample_chunks)
        
        prompt = f"""Analyze these Discord conversations from a competitive programming community and create a comprehensive summary.

Conversations (Retrieved via {retrieval_method} search):
{combined_text}

Create a summary covering:
1. Main topics and discussions
2. Key participants and their contributions  
3. Important announcements or decisions
4. Technical discussions (coding, algorithms, etc.)
5. Community activities or events mentioned

Write in natural Hinglish style, keep it engaging and under 200 words:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.5,
                    'max_tokens': 300
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"❌ Summary Error: {str(e)}"
