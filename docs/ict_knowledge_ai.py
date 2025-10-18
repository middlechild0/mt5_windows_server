#!/usr/bin/env python3
"""
ICT Trading Knowledge AI System
===============================
This script processes ICT trading transcripts and creates an AI system that can:
1. Learn from transcribed trading concepts
2. Group related concepts automatically using clustering
3. Answer questions based on similarity search
4. Provide relevant trading insights from ICT's teachings
"""

import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: Could not download NLTK data. Some features may not work.")

class ICTKnowledgeAI:
    def __init__(self, max_features=5000, n_clusters=10, chunk_size=250):
        """
        Initialize the ICT Knowledge AI system.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF vectorizer
            n_clusters (int): Number of clusters for grouping concepts
            chunk_size (int): Target words per text chunk
        """
        self.max_features = max_features
        self.n_clusters = n_clusters
        self.chunk_size = chunk_size
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.svd = TruncatedSVD(n_components=100, random_state=42)  # Dimensionality reduction
        
        # Storage for processed data
        self.chunks = []
        self.chunk_vectors = None
        self.cluster_labels = None
        self.is_trained = False

    def clean_text(self, text):
        """
        Clean and preprocess the transcript text.
        
        Args:
            text (str): Raw transcript text
            
        Returns:
            str: Cleaned text
        """
        # Remove timestamps (common patterns: [00:00], 0:00, etc.)
        text = re.sub(r'\[?\d{1,2}:\d{2}(?::\d{2})?\]?', '', text)
        
        # Remove common filler words and phrases
        fillers = [
            r'\b(um|uh|er|ah|like|you know|so|well|actually|basically|literally)\b',
            r'\b(okay|alright|right|yeah|yes|no)\b',
            r'\b(and uh|and um|but uh|but um)\b'
        ]
        
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove very short segments (likely noise)
        sentences = sent_tokenize(text)
        sentences = [s for s in sentences if len(s.split()) > 5]
        
        return ' '.join(sentences)

    def create_chunks(self, text):
        """
        Split text into meaningful chunks of approximately target size.
        
        Args:
            text (str): Cleaned text
            
        Returns:
            list: List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, start new chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_word_count = sentence_words
            else:
                current_chunk += " " + sentence
                current_word_count += sentence_words
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.split()) >= 20]
        
        return chunks

    def remove_duplicates(self, chunks, similarity_threshold=0.8):
        """
        Remove duplicate or very similar chunks based on cosine similarity.
        
        Args:
            chunks (list): List of text chunks
            similarity_threshold (float): Threshold for considering chunks as duplicates
            
        Returns:
            list: Filtered list of unique chunks
        """
        if not chunks:
            return chunks
        
        # Create a simple vectorizer for duplicate detection
        temp_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        temp_vectors = temp_vectorizer.fit_transform(chunks)
        
        unique_chunks = []
        used_indices = set()
        
        for i, chunk in enumerate(chunks):
            if i in used_indices:
                continue
                
            # Find similar chunks
            similarities = cosine_similarity(temp_vectors[i:i+1], temp_vectors).flatten()
            similar_indices = np.where(similarities > similarity_threshold)[0]
            
            # Keep the longest chunk among similar ones
            best_chunk = chunk
            best_length = len(chunk)
            
            for idx in similar_indices:
                if len(chunks[idx]) > best_length:
                    best_chunk = chunks[idx]
                    best_length = len(chunks[idx])
                used_indices.add(idx)
            
            unique_chunks.append(best_chunk)
        
        return unique_chunks

    def train(self, transcript_file):
        """
        Train the AI system on the transcript data.
        
        Args:
            transcript_file (str): Path to the transcript file
        """
        print("🤖 Loading and processing transcript data...")
        
        # Load the transcript file
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except FileNotFoundError:
            print(f"❌ Error: Could not find transcript file '{transcript_file}'")
            return False
        
        if not raw_text.strip():
            print("❌ Error: Transcript file is empty")
            return False
        
        print(f"📄 Loaded {len(raw_text)} characters of transcript data")
        
        # Clean the text
        print("🧹 Cleaning text...")
        cleaned_text = self.clean_text(raw_text)
        print(f"✅ Cleaned text: {len(cleaned_text)} characters")
        
        # Create chunks
        print("📝 Creating text chunks...")
        self.chunks = self.create_chunks(cleaned_text)
        print(f"✅ Created {len(self.chunks)} initial chunks")
        
        # Remove duplicates
        print("🔍 Removing duplicate chunks...")
        self.chunks = self.remove_duplicates(self.chunks)
        print(f"✅ Final chunks after deduplication: {len(self.chunks)}")
        
        if len(self.chunks) < 5:
            print("❌ Error: Not enough content to train on after processing")
            return False
        
        # Vectorize the chunks
        print("🔢 Vectorizing text chunks...")
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        print(f"✅ Created {self.chunk_vectors.shape[1]} features")
        
        # Apply dimensionality reduction
        print("📉 Applying dimensionality reduction...")
        reduced_vectors = self.svd.fit_transform(self.chunk_vectors)
        
        # Cluster the chunks
        print(f"🎯 Clustering into {self.n_clusters} concept groups...")
        self.cluster_labels = self.kmeans.fit_predict(reduced_vectors)
        
        # Analyze clusters
        self.analyze_clusters()
        
        self.is_trained = True
        print("🎉 Training complete!")
        return True

    def analyze_clusters(self):
        """Analyze and display information about the discovered clusters."""
        print("\n📊 Cluster Analysis:")
        print("-" * 50)
        
        cluster_counts = Counter(self.cluster_labels)
        
        for cluster_id in range(self.n_clusters):
            count = cluster_counts.get(cluster_id, 0)
            if count == 0:
                continue
                
            print(f"\n🏷️  Cluster {cluster_id} ({count} chunks):")
            
            # Get chunks in this cluster
            cluster_chunks = [self.chunks[i] for i, label in enumerate(self.cluster_labels) if label == cluster_id]
            
            # Show a sample chunk from this cluster
            if cluster_chunks:
                sample = cluster_chunks[0][:200] + "..." if len(cluster_chunks[0]) > 200 else cluster_chunks[0]
                print(f"   Sample: {sample}")

    def query(self, question, top_k=3):
        """
        Find the most relevant chunks for a given question.
        
        Args:
            question (str): The question to search for
            top_k (int): Number of top results to return
            
        Returns:
            list: List of tuples (chunk, similarity_score, cluster_id)
        """
        if not self.is_trained:
            print("❌ Error: Model not trained yet. Please run train() first.")
            return []
        
        # Vectorize the question
        question_vector = self.vectorizer.transform([question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, self.chunk_vectors).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                results.append({
                    'chunk': self.chunks[idx],
                    'similarity': similarities[idx],
                    'cluster': self.cluster_labels[idx]
                })
        
        return results

    def save_model(self, model_path='ict_knowledge_ai.pkl'):
        """Save the trained model to disk."""
        if not self.is_trained:
            print("❌ Error: Cannot save untrained model")
            return False
        
        model_data = {
            'vectorizer': self.vectorizer,
            'kmeans': self.kmeans,
            'svd': self.svd,
            'chunks': self.chunks,
            'chunk_vectors': self.chunk_vectors,
            'cluster_labels': self.cluster_labels,
            'config': {
                'max_features': self.max_features,
                'n_clusters': self.n_clusters,
                'chunk_size': self.chunk_size
            }
        }
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def load_model(self, model_path='ict_knowledge_ai.pkl'):
        """Load a trained model from disk."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.kmeans = model_data['kmeans']
            self.svd = model_data['svd']
            self.chunks = model_data['chunks']
            self.chunk_vectors = model_data['chunk_vectors']
            self.cluster_labels = model_data['cluster_labels']
            
            config = model_data['config']
            self.max_features = config['max_features']
            self.n_clusters = config['n_clusters']
            self.chunk_size = config['chunk_size']
            
            self.is_trained = True
            print(f"✅ Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def interactive_query_session(ai_system):
    """Run an interactive query session with the AI system."""
    print("\n" + "="*60)
    print("🤖 ICT Trading Knowledge AI - Interactive Session")
    print("="*60)
    print("Ask questions about trading concepts from ICT's teachings.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', '']:
                print("👋 Session ended. Happy trading!")
                break
            
            print("\n🔍 Searching for relevant concepts...")
            results = ai_system.query(question, top_k=3)
            
            if not results:
                print("❌ No relevant information found. Try rephrasing your question.")
                continue
            
            print(f"\n📚 Found {len(results)} relevant concept(s):")
            print("-" * 50)
            
            for i, result in enumerate(results, 1):
                print(f"\n🎯 Result {i} (Similarity: {result['similarity']:.3f}, Cluster: {result['cluster']}):")
                print(f"   {result['chunk'][:400]}...")
                if len(result['chunk']) > 400:
                    print(f"   [Content truncated - {len(result['chunk'])} total characters]")
        
        except KeyboardInterrupt:
            print("\n\n👋 Session ended by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Initialize the AI system
    print("🚀 Initializing ICT Knowledge AI System...")
    ai = ICTKnowledgeAI(max_features=5000, n_clusters=10, chunk_size=250)
    
    # Train on the transcript file
    transcript_file = "playlist_transcripts.txt"
    
    if ai.train(transcript_file):
        # Save the trained model
        ai.save_model()
        
        # Start interactive session
        interactive_query_session(ai)
    else:
        print("\n💡 Note: The transcript file appears to be empty.")
        print("   After your transcription bot fills 'playlist_transcripts.txt',")
        print("   run this script again to train the AI system.")
        
        # Show example of how to use it
        print("\n📋 Example usage after transcripts are available:")
        print("   python ict_knowledge_ai.py")
        print("\n📋 Example questions you can ask:")
        print("   - When should I enter a trade?")
        print("   - How do I manage risk?")
        print("   - What are institutional order blocks?")
        print("   - How to identify market structure?")