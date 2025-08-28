import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import uuid
import math
import numpy as np
import pickle
import os

class HybridVectorStore:
    def __init__(self, collection_name="hinglish_hybrid_chat"):
        print("🔥 Initializing HYBRID retrieval system...")
        
        # Dense embedding model (best for Hinglish)
        self.embedding_model = SentenceTransformer('sentence-transformers/LaBSE')
        print("✅ LaBSE embedding model loaded!")
        
        # ChromaDB for vector storage
        self.client = chromadb.PersistentClient(path="./hybrid_chat_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 storage
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        self.bm25_path = "./bm25_index.pkl"
        
        print("💾 Hybrid vector store initialized!")
    
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
            
            for chunk in valid_chunks:
                metadatas.append({
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'message_count': str(chunk['message_count']),
                    'chunk_id': chunk['chunk_id'],
                    'users': ', '.join(chunk['users']),
                    'doc_idx': len(self.documents) + len(ids)  # Track document index
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
                raise e
        
        # Store documents and metadata for BM25
        self.documents = all_texts
        self.doc_metadata = all_metadata
        
        # Build BM25 index
        print("🔍 Building BM25 index for keyword search...")
        tokenized_docs = []
        for doc in self.documents:
            # Tokenize with both English and Hinglish considerations
            tokens = self._tokenize_hinglish(doc.lower())
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Save BM25 index
        with open(self.bm25_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'documents': self.documents,
                'doc_metadata': self.doc_metadata
            }, f)
        
        print(f"🎉 HYBRID system ready with {len(self.documents)} documents!")
        print(f"📊 Vector store: {self.collection.count()} chunks")
        print(f"📊 BM25 index: {len(self.documents)} documents")
    
    def _tokenize_hinglish(self, text):
        """Custom tokenizer for Hinglish text"""
        import re
        # Split on spaces and common punctuation, preserve Hinglish words
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def load_indices(self):
        """Load pre-built BM25 index"""
        if os.path.exists(self.bm25_path):
            print("📂 Loading existing BM25 index...")
            with open(self.bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.documents = data['documents']
                self.doc_metadata = data['doc_metadata']
            print("✅ BM25 index loaded!")
            return True
        return False
    
    def hybrid_search(self, query, n_results=5, alpha=0.5):
        """
        Hybrid search using RRF to combine vector and BM25 results
        alpha: weight for vector vs BM25 (0.5 = equal weight)
        """
        if not self.bm25:
            if not self.load_indices():
                print("❌ BM25 index not found!")
                return None
        
        print(f"🔍 Performing HYBRID search for: '{query}'")
        
        # 1. Vector search with embeddings
        vector_results = self.collection.query(
            query_embeddings=[self.embedding_model.encode([query]).tolist()[0]],
            n_results=min(n_results * 3, len(self.documents)),  # Get more candidates
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
    
    def get_stats(self):
        """Get database statistics"""
        vector_count = self.collection.count()
        bm25_count = len(self.documents) if self.documents else 0
        return f"💾 Vector DB: {vector_count} | BM25 Index: {bm25_count} chunks"
