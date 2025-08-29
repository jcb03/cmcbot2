import json
from datetime import datetime
import re

class ChatDataProcessor:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
    
    def load_and_clean_data(self):
        """Load Discord export with better Hinglish preprocessing"""
        print("📂 Loading Discord export...")
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        raw_messages = data.get('messages', [])
        
        print(f"📊 Found {len(raw_messages)} raw messages")
        
        for idx, msg in enumerate(raw_messages):
            content = msg.get('content', '').strip()
            author_info = msg.get('author', {})
            author_name = author_info.get('nickname') or author_info.get('name', 'unknown')
            timestamp = msg.get('timestamp', datetime.now().isoformat())
            
            # Enhanced filtering for Hinglish content
            if (content and 
                len(content) > 3 and 
                not author_info.get('isBot', False) and
                not self._is_junk_message(content) and
                self._has_meaningful_content(content)):
                
                # Clean content for better indexing
                cleaned_content = self._clean_hinglish_text(content)
                
                messages.append({
                    'text': cleaned_content,
                    'timestamp': timestamp,
                    'user': author_name,
                    'message_id': msg.get('id', len(messages)),
                    'original_index': idx  # FIXED: Track original position for context expansion
                })
        
        print(f"✅ Processed {len(messages)} valid messages")
        return messages
    
    def _clean_hinglish_text(self, text):
        """Clean text for better BM25 and embedding performance"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive punctuation but preserve meaning
        text = re.sub(r'[!]{3,}', '!!', text)
        text = re.sub(r'[?]{3,}', '??', text) 
        text = re.sub(r'[.]{4,}', '...', text)
        return text.strip()
    
    def _has_meaningful_content(self, text):
        """Enhanced content validation for Hinglish"""
        # Must have alphabetic characters
        if not re.search(r'[a-zA-Z]', text):
            return False
        # Check for meaningful words (both English and romanized Hindi)
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        return len(words) >= 2
    
    def _is_junk_message(self, text):
        """Enhanced junk detection for Discord messages"""
        # Emoji only
        if re.match(r'^[:][a-zA-Z0-9_]+[:]$', text):
            return True
        # Very short after cleaning
        if len(text.strip()) <= 3:
            return True
        # All caps spam (but allow genuine caps)
        if len(text) > 15 and text.isupper() and len(set(text.replace(' ', ''))) < 4:
            return True
        # Excessive repetition
        words = text.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:  # Allow some repetition for emphasis
                return True
        return False
    
    def tokenize_hinglish(self, text):
        """Hinglish-aware tokenizer"""
        # Split on word boundaries, preserve romanized Hindi words
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out very short tokens (but keep Hindi particles like 'ke', 'ka', 'ki')
        meaningful_tokens = []
        for token in tokens:
            if len(token) >= 2:  # Keep 2+ character words
                meaningful_tokens.append(token)
        
        return meaningful_tokens
    
    def chunk_messages(self, messages, chunk_size=8):
        """FIXED: Create optimized chunks with overlap and quality scoring"""
        print("🔗 Creating optimized conversation chunks with overlap...")
        chunks = []
        
        # Sort by timestamp to maintain conversation flow
        messages.sort(key=lambda x: x['timestamp'])
        
        # FIXED: Use sliding window with 50% overlap for better context
        step_size = max(1, chunk_size // 2)  # 50% overlap
        
        for i in range(0, len(messages), step_size):
            chunk_msgs = messages[i:i + chunk_size]
            
            if len(chunk_msgs) < 3:  # Skip very small chunks
                continue
            
            # Build conversation with better formatting
            conversation_lines = []
            users_in_chunk = set()
            
            for msg in chunk_msgs:
                # Format: User: message (better for both BM25 and LLM understanding)
                line = f"{msg['user']}: {msg['text']}"
                conversation_lines.append(line)
                users_in_chunk.add(msg['user'])
            
            combined_text = '\n'.join(conversation_lines)
            
            # Quality filter - skip low-quality chunks
            if len(combined_text.strip()) < 50 or len(users_in_chunk) < 2:
                continue
            
            # FIXED: Access list elements properly
            chunks.append({
                'text': combined_text,
                'start_time': chunk_msgs[0]['timestamp'],  # FIXED: Access first element
                'end_time': chunk_msgs[-1]['timestamp'],   # FIXED: Access last element  
                'message_count': len(chunk_msgs),
                'chunk_id': f"chunk_{len(chunks)}",
                'users': list(users_in_chunk),
                'start_index': chunk_msgs[0]['original_index'],  # FIXED: Access first element
                'end_index': chunk_msgs[-1]['original_index'],   # FIXED: Access last element
                'quality_score': self._calculate_chunk_quality(chunk_msgs)
            })
        
        print(f"✅ Created {len(chunks)} optimized chunks with 50% overlap")
        return chunks
    
    def _calculate_chunk_quality(self, messages):
        """Calculate chunk quality score for better ranking"""
        if not messages:
            return 0
        
        score = 0
        total_text_length = sum(len(msg['text']) for msg in messages)
        unique_users = len(set(msg['user'] for msg in messages))
        
        # Factors that increase quality
        score += min(total_text_length / 10, 50)  # Length bonus (max 50)
        score += unique_users * 15  # User diversity bonus (more important for Discord)
        score += len(messages) * 8   # Message count bonus
        
        # Bonus for Hinglish content (romanized Hindi + English mix)
        hinglish_indicators = ['bhai', 'yaar', 'kya', 'hai', 'thi', 'tha', 'hain', 'mein', 'ke', 'ka', 'ki']
        all_text = ' '.join(msg['text'].lower() for msg in messages)
        hinglish_count = sum(1 for indicator in hinglish_indicators if indicator in all_text)
        score += hinglish_count * 5  # Bonus for Hinglish content
        
        return min(score, 100)  # Cap at 100
