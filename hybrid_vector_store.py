import chromadb
from chromadb.config import Settings
import uuid
import math
import numpy as np
import pickle
import os
import glob
from pathlib import Path

# Load environment variables
def load_env():
    env_path = Path('.') / '.env'
    if env_path.exists():
        print("📂 Loading .env file...")
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.strip().split('=', 1)
                        os.environ[key.strip()] = val.strip().strip('"\'')
        print("✅ .env file loaded")

class HybridVectorStore:
    def __init__(self, collection_name="hinglish_hybrid_chat"):
        print("🔥 Initializing SIMPLE HYBRID retrieval system...")
        
        # Load environment variables
        load_env()
        
        # Check for OpenAI API key (REQUIRED)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file!")
        
        # Test OpenAI
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            print("🔄 Testing OpenAI connection...")
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=["test connection"]
            )
            
            test_embedding = response.data[0].embedding
            print(f"✅ OpenAI confirmed working! (embedding dim: {len(test_embedding)})")
            
        except Exception as e:
            raise ValueError(f"❌ OpenAI API test failed: {e}")
        
        # Database setup
        self.db_path = self._find_existing_db_path()
        self.collection_name = collection_name
        self.bm25_path = "./bm25_index.pkl"
        self.setup_flag_path = "./setup_complete.flag"
        
        # Initialize ChromaDB
        try:
            client_settings = Settings(anonymized_telemetry=False, is_persistent=True)
            self.client = chromadb.PersistentClient(path=self.db_path, settings=client_settings)
        except:
            self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Get/create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"📂 Found existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"🆕 Created new collection: {collection_name}")
        
        # BM25 components
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        
        print(f"💾 Simple hybrid vector store initialized at: {self.db_path}")
    
    def _get_openai_embeddings(self, texts, batch_size=50):
        """Get OpenAI embeddings (no normalization for simplicity)"""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.openai_api_key)
        all_embeddings = []
        total_batches = math.ceil(len(texts) / batch_size)
        
        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            batch_texts = texts[i:i + batch_size]
            
            print(f"🔥 OpenAI batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
            
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_texts
                )
                
                for item in response.data:
                    all_embeddings.append(item.embedding)
                    
            except Exception as e:
                print(f"❌ OpenAI batch {batch_num} failed: {e}")
                raise e
        
        return all_embeddings
    
    def _get_query_embedding(self, query):
        """Get query embedding from OpenAI"""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.openai_api_key)
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        
        return response.data[0].embedding
    
    def _find_existing_db_path(self):
        """Smart detection of existing vector databases"""
        possible_paths = [
            "./simple_hybrid_chat_db",
            "./openai_hybrid_chat_db",
            "./hybrid_chat_db"
        ]
        
        try:
            chat_db_folders = glob.glob("./*chat_db*")
            possible_paths.extend(chat_db_folders)
        except:
            pass
        
        for path in possible_paths:
            if os.path.exists(path):
                chroma_files = [
                    os.path.join(path, "chroma.sqlite3"),
                    os.path.join(path, "index")
                ]
                
                for chroma_file in chroma_files:
                    if os.path.exists(chroma_file):
                        print(f"🎯 Found existing ChromaDB at: {path}")
                        return path
        
        print("🔧 Creating new ChromaDB at: ./simple_hybrid_chat_db")
        return "./simple_hybrid_chat_db"
    
    def is_setup_complete(self):
        """Check if setup is complete"""
        setup_indicators = 0
        
        if os.path.exists(self.setup_flag_path):
            print("✅ Found setup completion flag")
            setup_indicators += 1
        
        try:
            vector_count = self.collection.count()
            if vector_count > 1000:
                print(f"✅ Found {vector_count:,} vectors in ChromaDB")
                setup_indicators += 1
        except Exception as e:
            print(f"⚠️  ChromaDB check failed: {e}")
        
        if os.path.exists(self.bm25_path):
            try:
                with open(self.bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    bm25_doc_count = len(data.get('documents', []))
                    if bm25_doc_count > 1000:
                        print(f"✅ Found BM25 index with {bm25_doc_count:,} documents")
                        setup_indicators += 1
            except Exception as e:
                print(f"⚠️  BM25 index issue: {e}")
        
        is_complete = setup_indicators >= 2
        
        if is_complete:
            print("🎉 COMPLETE SETUP DETECTED - skipping data processing!")
        else:
            print(f"🔧 Setup indicators: {setup_indicators}/3 - will process data")
        
        return is_complete
    
    def load_existing_indices(self):
        """Load existing BM25 index"""
        if os.path.exists(self.bm25_path):
            try:
                print("📂 Loading existing BM25 index...")
                with open(self.bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data.get('bm25')
                    self.documents = data.get('documents', [])
                    self.doc_metadata = data.get('doc_metadata', [])
                
                if self.bm25 and len(self.documents) > 0:
                    print(f"✅ BM25 index loaded! ({len(self.documents):,} documents)")
                    return True
                else:
                    print("⚠️  BM25 index is empty")
                    return False
            except Exception as e:
                print(f"❌ BM25 loading error: {e}")
                return False
        return False
    
    def mark_setup_complete(self):
        """Mark setup as complete"""
        try:
            with open(self.setup_flag_path, 'w') as f:
                import datetime
                timestamp = datetime.datetime.now().isoformat()
                f.write(f"SIMPLE_HYBRID_SETUP_COMPLETE_{timestamp}")
            print("✅ Setup completion flag created")
        except Exception as e:
            print(f"⚠️  Could not create setup flag: {e}")
    
    def add_chunks(self, chunks, batch_size=400):
        """Add chunks with simple OpenAI embeddings"""
        from rank_bm25 import BM25Okapi
        
        total_chunks = len(chunks)
        print(f"🚀 Adding {total_chunks} chunks with SIMPLE OpenAI embeddings...")
        
        all_texts = []
        all_metadata = []
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = math.ceil(total_chunks / batch_size)
            
            print(f"⚡ Processing batch {batch_num}/{total_batches}...")
            
            texts = []
            valid_chunks = []
            
            for chunk in batch_chunks:
                if chunk['text'] and len(chunk['text'].strip()) > 5:
                    texts.append(chunk['text'])
                    valid_chunks.append(chunk)
                    all_texts.append(chunk['text'])
                    all_metadata.append({
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'message_count': str(chunk['message_count']),
                        'chunk_id': chunk['chunk_id'],
                        'users': ', '.join(chunk['users'])
                    })
            
            if not texts:
                continue
            
            # Generate simple OpenAI embeddings
            embeddings = self._get_openai_embeddings(texts, batch_size=40)
            
            # Prepare ChromaDB data
            metadatas = []
            ids = []
            
            for j, chunk in enumerate(valid_chunks):
                metadatas.append({
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'message_count': str(chunk['message_count']),
                    'chunk_id': chunk['chunk_id'],
                    'users': ', '.join(chunk['users']),
                    'doc_idx': len(all_texts) - len(valid_chunks) + j
                })
                ids.append(str(uuid.uuid4()))
            
            # Add to ChromaDB
            try:
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"✅ Vector batch {batch_num} added ({len(texts)} chunks)")
            except Exception as e:
                print(f"❌ Error in batch {batch_num}: {e}")
                continue
        
        # Store documents for BM25
        self.documents = all_texts
        self.doc_metadata = all_metadata
        
        # Build BM25 index
        print("🔍 Building BM25 index...")
        tokenized_docs = []
        for doc in self.documents:
            tokens = self._tokenize_hinglish(doc.lower())
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Save BM25 index
        try:
            with open(self.bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents,
                    'doc_metadata': self.doc_metadata
                }, f)
            print("✅ BM25 index saved")
        except Exception as e:
            print(f"❌ BM25 save error: {e}")
        
        # Mark setup complete
        self.mark_setup_complete()
        
        final_vector_count = self.collection.count()
        print(f"🎉 SIMPLE HYBRID system ready!")
        print(f"📊 Vector store: {final_vector_count:,} chunks")
        print(f"📊 BM25 index: {len(self.documents):,} documents")
    
    def _tokenize_hinglish(self, text):
        """Hinglish tokenizer"""
        import re
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if len(t) >= 2]
    
    def hybrid_search(self, query, n_results=4):
        """SIMPLE: Just RRF + BM25 + OpenAI embeddings - NO FILTERING, NO EXPANSION"""
        if not self.bm25:
            if not self.load_existing_indices():
                return None
        
        print(f"🔍 Simple hybrid search: '{query}'")
        
        try:
            # 1. Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # 2. Vector search (get top 20 candidates)
            vector_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=20,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 3. BM25 search (get top 20 candidates)
            query_tokens = self._tokenize_hinglish(query.lower())
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:20]
            
            # 4. SIMPLE RRF fusion (50% vector, 50% BM25)
            def rrf_score(rank, k=60):
                return 1 / (k + rank + 1)
            
            doc_scores = {}
            
            # Add vector scores (50% weight)
            for i, (doc, metadata, distance) in enumerate(zip(
                vector_results['documents'][0], 
                vector_results['metadatas'][0],
                vector_results['distances'][0]
            )):
                doc_idx = int(metadata.get('doc_idx', i))
                similarity = 1 - distance
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 0.5 * rrf_score(i) * similarity
            
            # Add BM25 scores (50% weight)
            for i, doc_idx in enumerate(bm25_top_indices):
                if bm25_scores[doc_idx] > 0:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 0.5 * rrf_score(i)
            
            # 5. Sort by combined scores and return top N
            final_ranking = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            if not final_ranking:
                print("⚠️  No results found")
                return None
            
            # Prepare results (NO EXPANSION - just raw chunks)
            top_docs = []
            top_metadata = []
            top_distances = []
            
            for doc_idx, score in final_ranking[:n_results]:
                if doc_idx < len(self.documents) and doc_idx < len(self.doc_metadata):
                    top_docs.append(self.documents[doc_idx])
                    top_metadata.append(self.doc_metadata[doc_idx])
                    top_distances.append(1 - score)  # Convert score back to distance
            
            print(f"🎯 Simple results: {len(top_docs)} chunks")
            
            return {
                'documents': [top_docs],
                'metadatas': [top_metadata], 
                'distances': [top_distances]
            }
        
        except Exception as e:
            print(f"❌ Simple search error: {e}")
            return None
    
    def get_stats(self):
        """Get stats"""
        try:
            vector_count = self.collection.count()
        except:
            vector_count = 0
        
        bm25_count = len(self.documents) if self.documents else 0
        
        return f"💾 ✅ SIMPLE OpenAI Hybrid | Vector: {vector_count:,} | BM25: {bm25_count:,}"
    
    def force_reset(self):
        """Reset everything"""
        import shutil
        
        files_to_remove = [self.setup_flag_path, self.bm25_path]
        
        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Removed: {file_path}")
            except Exception as e:
                print(f"❌ Error removing {file_path}: {e}")
        
        try:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                print(f"🗑️ Removed ChromaDB: {self.db_path}")
        except Exception as e:
            print(f"❌ Error removing ChromaDB: {e}")
        
        print("✅ Reset complete - restart to reprocess with SIMPLE hybrid")
