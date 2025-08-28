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
        
        for msg in raw_messages:
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
                
                # Clean content for better BM25 indexing
                cleaned_content = self._clean_hinglish_text(content)
                
                messages.append({
                    'text': cleaned_content,
                    'timestamp': timestamp,
                    'user': author_name,
                    'message_id': msg.get('id', len(messages))
                })
        
        print(f"✅ Processed {len(messages)} valid messages")
        return messages
    
    def _clean_hinglish_text(self, text):
        """Clean text for better BM25 and embedding performance"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}|[?]{2,}|[.]{3,}', '.', text)
        return text.strip()
    
    def _has_meaningful_content(self, text):
        """Check if message has meaningful content for indexing"""
        # Must have some alphabetic characters
        if not re.search(r'[a-zA-Z]', text):
            return False
        # Should not be just numbers or punctuation
        alpha_chars = len(re.findall(r'[a-zA-Z]', text))
        return alpha_chars >= 3
    
    def _is_junk_message(self, text):
        """Enhanced junk detection"""
        # Emoji only
        if re.match(r'^[:][a-zA-Z0-9_]+[:]$', text):
            return True
        # Very short
        if len(text.strip()) <= 2:
            return True
        # All caps spam
        if len(text) > 10 and text.isupper() and len(set(text)) < 3:
            return True
        # Excessive repetition
        words = text.split()
        if len(words) > 2:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return True
        return False
    
    def chunk_messages(self, messages, chunk_size=5):
        """Create optimized chunks for hybrid retrieval"""
        print("🔗 Creating optimized conversation chunks...")
        chunks = []
        
        # Sort by timestamp
        messages.sort(key=lambda x: x['timestamp'])
        
        for i in range(0, len(messages), chunk_size):
            chunk_msgs = messages[i:i + chunk_size]
            
            # Build conversation with better formatting for BM25
            conversation_lines = []
            users_in_chunk = set()
            
            for msg in chunk_msgs:
                # Format: User: message (better for keyword search)
                line = f"{msg['user']}: {msg['text']}"
                conversation_lines.append(line)
                users_in_chunk.add(msg['user'])
            
            combined_text = '\n'.join(conversation_lines)
            
            # Skip very short chunks
            if len(combined_text.strip()) < 25:
                continue
            
            chunks.append({
                'text': combined_text,
                'start_time': chunk_msgs[0]['timestamp'],
                'end_time': chunk_msgs[-1]['timestamp'],
                'message_count': len(chunk_msgs),
                'chunk_id': f"chunk_{len(chunks)}",
                'users': list(users_in_chunk)
            })
        
        print(f"✅ Created {len(chunks)} optimized chunks")
        return chunks
