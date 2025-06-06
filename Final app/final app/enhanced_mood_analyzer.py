import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional, Union
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings

class EnhancedMoodAnalyzer:
    """
    Advanced mood analyzer using pre-trained language models to extract mood vectors from text.
    
    This class provides three strategies for mood extraction:
    1. Sentence Transformers (primary strategy) - Uses pre-trained BERT-based models
    2. TensorFlow Hub Universal Sentence Encoder (secondary strategy)
    3. Keyword-based matching (fallback strategy)
    
    The class automatically selects the best available strategy based on installed dependencies.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the enhanced mood analyzer.
        
        Parameters:
        -----------
        model_name : str
            Name of the sentence-transformers model to use if available
        """
        # Define mood features we want to extract
        self.mood_features = [
            'mood_happy', 'mood_sad', 'mood_energetic', 'mood_relaxing', 
            'mood_nostalgic', 'mood_romantic', 'mood_angry', 
            'mood_confident', 'mood_workout', 'mood_party', 'mood_study'
        ]
        
        # Mood name mapping for convenience
        self.mood_names = [feat.replace('mood_', '') for feat in self.mood_features]
        
        # Activity to mood mapping - used to boost relevant moods
        self.activity_mood_mapping = {
            "workout": {"energetic": 0.9, "confident": 0.7, "workout": 1.0},
            "study": {"study": 1.0, "relaxing": 0.8},
            "sleep": {"relaxing": 1.0, "study": 0.6},
            "party": {"party": 1.0, "energetic": 0.8, "happy": 0.8}
        }
        
        # Mapping dictionaries for the fallback keyword approach
        self.mood_mapping = self._initialize_mood_mapping()
        
        # Strategy selection - try to load advanced models
        self.strategy = 'keyword'  # Default fallback strategy
        self.model = None
        self.mapping_matrix = None
        
        # Try to load sentence transformers (best option)
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading Sentence Transformers model...")
            self.model = SentenceTransformer(model_name)
            self.strategy = 'sentence_transformers'
            print(f"Using strategy: {self.strategy}")
            
            # Initialize mapping matrix for sentence embeddings
            self._initialize_mapping_matrix(embed_dim=self.model.get_sentence_embedding_dimension())
            
        except ImportError:
            # Try TensorFlow Hub as secondary strategy
            try:
                import tensorflow as tf
                import tensorflow_hub as hub
                print("Loading TensorFlow Hub Universal Sentence Encoder...")
                self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                self.strategy = 'tensorflow_hub'
                print(f"Using strategy: {self.strategy}")
                
                # Initialize mapping matrix for USE embeddings (default dim=512)
                self._initialize_mapping_matrix(embed_dim=512)
                
            except ImportError:
                print("Using fallback keyword-based mood extraction.")
                # Fallback to keyword strategy (already set as default)
    
    def _initialize_mapping_matrix(self, embed_dim: int = 384):
        """
        Initialize a mapping matrix to convert embeddings to mood vectors.
        
        Parameters:
        -----------
        embed_dim : int
            Dimension of the embedding vectors from the pre-trained model
        """
        # Try to load pre-computed mapping matrix
        mapping_file = f'mood_mapping_matrix_{embed_dim}.npy'
        if os.path.exists(mapping_file):
            try:
                self.mapping_matrix = np.load(mapping_file)
                print(f"Loaded mapping matrix from {mapping_file}")
                return
            except:
                pass
        
        # Create a simple mapping matrix based on random projections
        # In a production system, this would be learned from data
        np.random.seed(42)  # For reproducibility
        mapping = np.random.normal(0, 0.1, (embed_dim, len(self.mood_features)))
        
        # Bias the mapping matrix to recognize mood-related terms
        mood_terms = {}
        for i, mood in enumerate(self.mood_names):
            mood_terms[mood] = self._get_mood_keywords(mood)
        
        # Get embeddings for mood keywords if model is available
        if self.model is not None:
            try:
                if self.strategy == 'sentence_transformers':
                    for i, mood in enumerate(self.mood_names):
                        keywords = mood_terms[mood]
                        for keyword in keywords[:5]:  # Use top 5 keywords
                            keyword_embed = self.model.encode([keyword])[0]
                            # Strengthen the connection between this keyword and mood
                            mapping[:, i] += 0.1 * keyword_embed / np.linalg.norm(keyword_embed)
                
                elif self.strategy == 'tensorflow_hub':
                    for i, mood in enumerate(self.mood_names):
                        keywords = mood_terms[mood]
                        for keyword in keywords[:5]:  # Use top 5 keywords
                            keyword_embed = self.model([keyword]).numpy()[0]
                            # Strengthen the connection between this keyword and mood
                            mapping[:, i] += 0.1 * keyword_embed / np.linalg.norm(keyword_embed)
            except:
                print("Warning: Could not optimize mapping matrix with embeddings")
        
        # Normalize the mapping matrix
        for i in range(mapping.shape[1]):
            mapping[:, i] = mapping[:, i] / np.linalg.norm(mapping[:, i])
        
        self.mapping_matrix = mapping
        
        # Save the mapping matrix for future use
        try:
            np.save(mapping_file, mapping)
            print(f"Saved mapping matrix to {mapping_file}")
        except:
            pass
    
    def _initialize_mood_mapping(self) -> Dict[str, List[str]]:
        """Initialize mood keyword mappings"""
        return {
            "happy": ['happy', 'joy', 'excited', 'cheerful', 'positive', 'uplifting', 
                     'bright', 'fun', 'elated', 'upbeat', 'merry', 'delighted'],
            
            "sad": ['sad', 'depressed', 'down', 'blue', 'unhappy', 'lonely', 
                   'melancholy', 'gloomy', 'somber', 'heartbroken', 'tearful'],
            
            "energetic": ['energetic', 'active', 'lively', 'pumped', 'energy', 'dynamic',
                         'vigorous', 'powerful', 'intense', 'vibrant', 'spirited'],
            
            "relaxing": ['relaxing', 'calm', 'peaceful', 'serene', 'tranquil', 'quiet',
                        'soothing', 'gentle', 'soft', 'mellow', 'easy', 'chill', 'ambient'],
            
            "nostalgic": ['nostalgic', 'memories', 'remember', 'past', 'reminiscent',
                         'childhood', 'throwback', 'retro', 'classic', 'vintage'],
            
            "romantic": ['romantic', 'love', 'passion', 'intimate', 'tender', 'affection',
                        'sensual', 'dreamy', 'emotional', 'heartfelt', 'sentimental'],
            
            "angry": ['angry', 'furious', 'mad', 'rage', 'irritated', 'annoyed',
                     'aggressive', 'hostile', 'fierce', 'intense', 'rebellious'],
            
            "confident": ['confident', 'strong', 'powerful', 'capable', 'self-assured',
                         'bold', 'determined', 'fearless', 'unstoppable', 'brave'],
            
            "workout": ['workout', 'exercise', 'gym', 'fitness', 'training', 'run',
                       'jog', 'cardio', 'lift', 'sweat', 'pump', 'grind'],
            
            "party": ['party', 'celebration', 'fun', 'dance', 'enjoy', 'festive',
                     'club', 'groove', 'night', 'beat', 'rhythm', 'vibe'],
            
            "study": ['study', 'focus', 'concentrate', 'work', 'productive', 'think',
                     'learn', 'read', 'calm', 'attention', 'academic', 'brain']
        }
    
    def _get_mood_keywords(self, mood: str) -> List[str]:
        """Get keywords for a specific mood"""
        return self.mood_mapping.get(mood, [])
    
    def extract_mood_from_text(self, text: str) -> np.ndarray:
        """
        Extract mood features from text input using the best available strategy.
        
        Parameters:
        -----------
        text : str
            User input text describing their mood or preferences
            
        Returns:
        --------
        numpy.ndarray
            Mood feature vector normalized between 0 and 1
        """
        if not text:
            # Return a balanced mood vector if no text is provided
            return np.ones(len(self.mood_features)) / len(self.mood_features)
        
        # Convert to lowercase
        text = text.lower()
        
        # Choose strategy based on what's available
        if self.strategy == 'sentence_transformers' and self.model is not None:
            return self._extract_mood_with_sentence_transformers(text)
        
        elif self.strategy == 'tensorflow_hub' and self.model is not None:
            return self._extract_mood_with_tensorflow_hub(text)
        
        else:
            # Fallback to keyword-based approach
            return self._extract_mood_with_keywords(text)
    
    def _extract_mood_with_sentence_transformers(self, text: str) -> np.ndarray:
        """Extract mood using Sentence Transformers"""
        try:
            # Get embedding
            embedding = self.model.encode([text])[0]
            
            # Project embedding to mood space
            mood_scores = np.dot(embedding, self.mapping_matrix)
            
            # Apply sigmoid to get values between 0 and 1
            mood_scores = 1 / (1 + np.exp(-mood_scores))
            
            # Check for activities with direct mapping
            mood_scores = self._apply_activity_mapping(text, mood_scores)
            
            # Normalize
            if np.sum(mood_scores) > 0:
                mood_scores = mood_scores / np.sum(mood_scores)
                
            return mood_scores
            
        except Exception as e:
            print(f"Error with sentence_transformers: {e}")
            # Fall back to keyword method if there's an error
            return self._extract_mood_with_keywords(text)
    
    def _extract_mood_with_tensorflow_hub(self, text: str) -> np.ndarray:
        """Extract mood using TensorFlow Hub's Universal Sentence Encoder"""
        try:
            # Get embedding
            embedding = self.model([text]).numpy()[0]
            
            # Project embedding to mood space
            mood_scores = np.dot(embedding, self.mapping_matrix)
            
            # Apply sigmoid to get values between 0 and 1
            mood_scores = 1 / (1 + np.exp(-mood_scores))
            
            # Check for activities with direct mapping
            mood_scores = self._apply_activity_mapping(text, mood_scores)
            
            # Normalize
            if np.sum(mood_scores) > 0:
                mood_scores = mood_scores / np.sum(mood_scores)
                
            return mood_scores
            
        except Exception as e:
            print(f"Error with tensorflow_hub: {e}")
            # Fall back to keyword method if there's an error
            return self._extract_mood_with_keywords(text)
    
    def _extract_mood_with_keywords(self, text: str) -> np.ndarray:
        """
        Extract mood using keyword matching (fallback method).
        This is similar to the original implementation but with improvements.
        """
        # Initialize mood scores
        mood_scores = np.zeros(len(self.mood_features))
        
        # For each mood dimension, check for keyword matches
        for i, mood_name in enumerate(self.mood_names):
            keywords = self._get_mood_keywords(mood_name)
            
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text)
            
            # Base score + boost for matches
            mood_scores[i] = 0.1 + min(0.5, matches * 0.1)
        
        # Check for activities
        mood_scores = self._apply_activity_mapping(text, mood_scores)
        
        # Ensure scores are between 0 and 1
        mood_scores = np.clip(mood_scores, 0, 1)
        
        # Normalize to sum to 1
        if np.sum(mood_scores) > 0:
            mood_scores = mood_scores / np.sum(mood_scores)
        
        return mood_scores
    
    def _apply_activity_mapping(self, text: str, mood_scores: np.ndarray) -> np.ndarray:
        """Apply activity-based mood mapping to boost certain mood scores"""
        # Make a copy to avoid modifying the input
        scores = mood_scores.copy()
        
        # Check for activities
        for activity, mood_boosts in self.activity_mood_mapping.items():
            if activity in text:
                for mood_name, boost in mood_boosts.items():
                    try:
                        idx = self.mood_names.index(mood_name)
                        scores[idx] = max(scores[idx], boost)
                    except:
                        pass
        
        return scores
    
    def get_top_moods(self, mood_vector: np.ndarray, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Get the top moods from a mood vector.
        
        Parameters:
        -----------
        mood_vector : numpy.ndarray
            The mood vector to analyze
        top_n : int
            Number of top moods to return
            
        Returns:
        --------
        List[Tuple[str, float]]
            List of (mood_name, score) tuples for the top moods
        """
        # Create pairs of mood names and scores
        mood_scores = [(name, score) for name, score in zip(self.mood_names, mood_vector)]
        
        # Sort by score in descending order
        mood_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N moods
        return mood_scores[:top_n]
    
    def save_mapping_matrix(self, filename: str = 'custom_mapping.npy') -> bool:
        """Save the current mapping matrix to a file"""
        if self.mapping_matrix is not None:
            try:
                np.save(filename, self.mapping_matrix)
                return True
            except:
                return False
        return False
    
    def load_mapping_matrix(self, filename: str) -> bool:
        """Load a mapping matrix from a file"""
        try:
            self.mapping_matrix = np.load(filename)
            return True
        except:
            return False
def test_enhanced_mood_analyzer():
    """Test the enhanced mood analyzer with various inputs"""
    # Create analyzer
    analyzer = EnhancedMoodAnalyzer()
    
    # Test with various inputs
    test_inputs = [
        "I'm feeling really energetic and want to work out",
        "I'm sad and feeling down today",
        "Need some relaxing music to help me study",
        "I want to dance at a party tonight",
        "Feeling nostalgic about my childhood",
        "I'm angry and frustrated with everything"
    ]
    
    print("\nTesting Enhanced Mood Analyzer:")
    print("-" * 50)
    
    for text in test_inputs:
        print(f"\nInput: '{text}'")
        
        # Get mood vector
        mood_vector = analyzer.extract_mood_from_text(text)
        
        # Get top moods
        top_moods = analyzer.get_top_moods(mood_vector, top_n=3)
        
        # Print results
        print("Top moods:")
        for mood, score in top_moods:
            print(f"- {mood}: {score:.3f}")
    
    print("\nMood analyzer test completed.")

# Run the test
test_enhanced_mood_analyzer()