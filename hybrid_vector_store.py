import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import uuid
import math
import numpy as np
import pickle
import os
import glob

class HybridVectorStore:
    def __init__(self, collection_name="hinglish_hybrid_chat"):
        print("🔥 Initializing HYBRID retrieval system...")
        
        # Smart path detection - find existing vector DB
        self.db_path = self._find_existing_db_path()
        self.collection_name = collection_name
        self.bm25_path = "./bm25_index.pkl"
        self.setup_flag_path = "./setup_complete.flag"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Try to get existing collection first, then create if not found
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"📂 Found existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"🆕 Created new collection: {collection_name}")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/LaBSE')
        print("✅ LaBSE embedding model loaded!")
        
        # BM25 components
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        
        print(f"💾 Hybrid vector store initialized at: {self.db_path}")
    
    def _find_existing_db_path(self):
        """Smart detection of existing vector databases"""
        possible_paths = [
            "./hybrid_chat_db",           # New preferred path
            "./discord_chat_db_v2",       # Your old path from LaBSE
            "./discord_chat_db",          # Even older path
            "./hinglish_chat_db",         # Alternative path
        ]
        
        # Also check for any folder ending with *chat_db*
        try:
            chat_db_folders = glob.glob("./*chat_db*")
            possible_paths.extend(chat_db_folders)
        except:
            pass
        
        for path in possible_paths:
            if os.path.exists(path):
                # Check if it has ChromaDB structure
                chroma_files = [
                    os.path.join(path, "chroma.sqlite3"),
                    os.path.join(path, "index"),
                    os.path.join(path, "data.parquet")
                ]
                
                # If any ChromaDB file exists, it's a valid DB
                for chroma_file in chroma_files:
                    if os.path.exists(chroma_file):
                        print(f"🎯 Found existing ChromaDB at: {path}")
                        return path
        
        # No existing DB found, create new one
        print("🔧 No existing ChromaDB found, creating new one at: ./hybrid_chat_db")
        return "./hybrid_chat_db"
    
    def is_setup_complete(self):
        """Enhanced setup detection with multiple checks"""
        
        setup_indicators = 0
        
        # Check 1: Setup flag exists
        if os.path.exists(self.setup_flag_path):
            print("✅ Found setup completion flag")
            setup_indicators += 1
        
        # Check 2: ChromaDB has significant data
        try:
            vector_count = self.collection.count()
            if vector_count > 1000:  # Must have substantial data
                print(f"✅ Found {vector_count:,} vectors in ChromaDB")
                setup_indicators += 1
            elif vector_count > 0:
                print(f"⚠️  Found only {vector_count} vectors (incomplete setup)")
        except Exception as e:
            print(f"⚠️  ChromaDB check failed: {e}")
            vector_count = 0
        
        # Check 3: BM25 index exists and is substantial
        if os.path.exists(self.bm25_path):
            try:
                with open(self.bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    bm25_doc_count = len(data.get('documents', []))
                    if bm25_doc_count > 1000:
                        print(f"✅ Found BM25 index with {bm25_doc_count:,} documents")
                        setup_indicators += 1
                    else:
                        print(f"⚠️  BM25 index has only {bm25_doc_count} documents")
            except Exception as e:
                print(f"⚠️  BM25 index corrupted: {e}")
        
        # Need at least 2 indicators for complete setup
        is_complete = setup_indicators >= 2
        
        if is_complete:
            print("🎉 COMPLETE SETUP DETECTED - skipping data processing!")
        else:
            print(f"🔧 Incomplete setup detected ({setup_indicators}/3 indicators) - will process data")
        
        return is_complete
    
    def load_existing_indices(self):
        """Load existing BM25 index and verify integrity"""
        if os.path.exists(self.bm25_path):
            try:
                print("📂 Loading existing BM25 index...")
                with open(self.bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data.get('bm25')
                    self.documents = data.get('documents', [])
                    self.doc_metadata = data.get('doc_metadata', [])
                
                if self.bm25 and len(self.documents) > 0:
                    print(f"✅ BM25 index loaded successfully! ({len(self.documents):,} documents)")
                    return True
                else:
                    print("⚠️  BM25 index is empty or corrupted")
                    return False
                    
            except Exception as e:
                print(f"❌ Error loading BM25 index: {e}")
                return False
        else:
            print("⚠️  BM25 index file not found")
            return False
    
    def mark_setup_complete(self):
        """Mark setup as complete"""
        try:
            with open(self.setup_flag_path, 'w') as f:
                import datetime
                timestamp = datetime.datetime.now().isoformat()
                f.write(f"HYBRID_SETUP_COMPLETE_{timestamp}")
            print("✅ Setup completion flag created")
        except Exception as e:
            print(f"⚠️  Could not create setup flag: {e}")
    
    def add_chunks(self, chunks, batch_size=2000):
        """Add chunks to both vector and BM25 indices"""
        total_chunks = len(chunks)
        print(f"🚀 Adding {total_chunks} chunks to HYBRID system...")
        
        all_texts = []
        all_metadata = []
        
        # Process in batches for vector store
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = math.ceil(total_chunks / batch_size)
            
            print(f"⚡ Processing batch {batch_num}/{total_batches} for vector store...")
            
            # Prepare batch data
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
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
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
                    'doc_idx': len(all_texts) - len(valid_chunks) + j  # Track document index
                })
                ids.append(str(uuid.uuid4()))
            
            # Add to ChromaDB
            try:
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"✅ Vector batch {batch_num} added ({len(texts)} chunks)")
            except Exception as e:
                print(f"❌ Error in vector batch {batch_num}: {e}")
                # Try to continue with next batch
                continue
        
        # Store documents and metadata for BM25
        self.documents = all_texts
        self.doc_metadata = all_metadata
        
        # Build BM25 index
        print("🔍 Building BM25 index for keyword search...")
        tokenized_docs = []
        for doc in self.documents:
            tokens = self._tokenize_hinglish(doc.lower())
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Save BM25 index with error handling
        try:
            with open(self.bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents,
                    'doc_metadata': self.doc_metadata
                }, f)
            print(f"✅ BM25 index saved successfully")
        except Exception as e:
            print(f"❌ Error saving BM25 index: {e}")
        
        # Mark setup complete
        self.mark_setup_complete()
        
        final_vector_count = self.collection.count()
        print(f"🎉 HYBRID system ready!")
        print(f"📊 Vector store: {final_vector_count:,} chunks")
        print(f"📊 BM25 index: {len(self.documents):,} documents")
    
    def _tokenize_hinglish(self, text):
        """Custom tokenizer for Hinglish text"""
        import re
        # Split on word boundaries, preserve Hinglish words
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def hybrid_search(self, query, n_results=5, alpha=0.5):
        """
        Hybrid search using RRF to combine vector and BM25 results
        alpha: weight for vector vs BM25 (0.5 = equal weight)
        """
        # Ensure BM25 is loaded
        if not self.bm25:
            if not self.load_existing_indices():
                print("❌ BM25 index not available for hybrid search!")
                # Fallback to vector-only search
                return self._vector_only_search(query, n_results)
        
        print(f"🔍 Performing HYBRID search for: '{query}'")
        
        try:
            # 1. Vector search with embeddings
            vector_results = self.collection.query(
                query_embeddings=[self.embedding_model.encode([query]).tolist()[0]],
                n_results=min(n_results * 3, len(self.documents) if self.documents else 100),
                include=['documents', 'metadatas', 'distances']
            )
            
            # 2. BM25 keyword search
            query_tokens = self._tokenize_hinglish(query.lower())
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Get top BM25 candidates
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:min(n_results * 3, len(bm25_scores))]
            
            # 3. RRF (Reciprocal Rank Fusion)
            def rrf_score(rank, k=60):
                return 1 / (k + rank + 1)
            
            doc_scores = {}
            
            # Add vector search scores using RRF
            for i, (doc, metadata) in enumerate(zip(vector_results['documents'][0], vector_results['metadatas'][0])):
                doc_idx = metadata.get('doc_idx', i)
                if doc_idx is not None:
                    doc_scores[doc_idx] = alpha * rrf_score(i)
            
            # Add BM25 scores using RRF
            for i, doc_idx in enumerate(bm25_top_indices):
                if bm25_scores[doc_idx] > 0:  # Only if there's actual keyword match
                    if doc_idx in doc_scores:
                        doc_scores[doc_idx] += (1 - alpha) * rrf_score(i)
                    else:
                        doc_scores[doc_idx] = (1 - alpha) * rrf_score(i)
            
            # Sort by combined RRF scores
            final_ranking = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Prepare results in ChromaDB format
            top_docs = []
            top_metadata = []
            top_distances = []
            
            for doc_idx, score in final_ranking[:n_results]:
                if doc_idx < len(self.documents):
                    top_docs.append(self.documents[doc_idx])
                    top_metadata.append(self.doc_metadata[doc_idx])
                    top_distances.append(1 - score)  # Convert score to distance
            
            print(f"🎯 Found {len(top_docs)} hybrid results")
            
            return {
                'documents': [top_docs],
                'metadatas': [top_metadata],
                'distances': [top_distances]
            }
            
        except Exception as e:
            print(f"❌ Hybrid search error: {e}")
            print("🔄 Falling back to vector-only search...")
            return self._vector_only_search(query, n_results)
    
    def _vector_only_search(self, query, n_results=5):
        """Fallback vector-only search"""
        try:
            results = self.collection.query(
                query_embeddings=[self.embedding_model.encode([query]).tolist()[0]],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            print(f"🎯 Found {len(results['documents'][0])} vector-only results")
            return results
        except Exception as e:
            print(f"❌ Vector search also failed: {e}")
            return None
    
    def get_stats(self):
        """Get comprehensive database statistics"""
        try:
            vector_count = self.collection.count()
        except:
            vector_count = 0
        
        bm25_count = len(self.documents) if self.documents else 0
        
        status = "✅ Hybrid Ready" if (vector_count > 0 and bm25_count > 0) else "⚠️ Partial Setup"
        
        return f"💾 {status} | Vector DB: {vector_count:,} | BM25: {bm25_count:,} chunks"
    
    def force_reset(self):
        """Force reset all indices"""
        import shutil
        
        files_to_remove = [
            self.setup_flag_path,
            self.bm25_path
        ]
        
        # Remove files
        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Removed: {file_path}")
            except Exception as e:
                print(f"❌ Error removing {file_path}: {e}")
        
        # Remove ChromaDB directory
        try:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                print(f"🗑️ Removed ChromaDB: {self.db_path}")
        except Exception as e:
            print(f"❌ Error removing ChromaDB: {e}")
        
        print("✅ Force reset complete - restart bot to reprocess data")
