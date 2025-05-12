import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import nltk
import re
import json
import pickle
import os
import random
from collections import Counter
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class EnhancedMusicRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir='./cache'):
        """
        Initialize the enhanced music recommender with SBERT and other advanced features
        
        Parameters:
        - model_name: The sentence transformer model to use
            'all-MiniLM-L6-v2' - Fast and efficient model with good performance
            'all-mpnet-base-v2' - Higher quality but slower model
        - cache_dir: Directory to cache processed data and models
        """
        # Database connection
        self.conn = psycopg2.connect(
            host="localhost",
            dbname="youtube_data",
            user="postgres",
            password="postgres"
        )
        self.cursor = self.conn.cursor()
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Load SBERT model
        print(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Cache directory
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Data containers
        self.videos_df = None
        self.comments_df = None
        self.video_embeddings = None
        self.video_features = None
        self.genre_mapping = None
        self.clusters = None
        self.cluster_keywords = None
        
        # Initialize expanded mood, theme, and activity keywords
        self._initialize_keywords()
        
    def _initialize_keywords(self):
        """Initialize expanded keyword dictionaries for better classification"""
        # Mood keywords with weighted terms
        self.mood_keywords = {
            'happy': ['happy', 'joy', 'upbeat', 'cheerful', 'fun', 'excited', 'uplifting', 'celebration', 
                     'smile', 'laugh', 'positive', 'happiness', 'great', 'awesome', 'wonderful', 'fantastic'],
            'sad': ['sad', 'depression', 'depressing', 'cry', 'crying', 'tears', 'heartbreak', 'melancholy',
                   'sorrow', 'grief', 'painful', 'devastating', 'emotional', 'hurt', 'broken', 'lonely'],
            'energetic': ['energy', 'pump', 'workout', 'gym', 'exercise', 'energetic', 'party', 'dance',
                        'upbeat', 'fast', 'adrenaline', 'hyper', 'powerful', 'intense', 'motivation', 'action'],
            'relaxing': ['relax', 'chill', 'peaceful', 'calm', 'sleep', 'study', 'ambient', 'meditation',
                       'soothing', 'gentle', 'quiet', 'tranquil', 'serene', 'mellow', 'soft', 'slow'],
            'nostalgic': ['memory', 'memories', 'nostalgia', 'nostalgic', 'childhood', 'old', 'classic', 'remember',
                         'throwback', 'retro', 'vintage', 'back in the day', 'reminds me', 'youth', 'grew up'],
            'romantic': ['love', 'romance', 'romantic', 'couple', 'relationship', 'wedding', 'together', 'heart',
                       'passion', 'kiss', 'lover', 'beautiful', 'emotional', 'intimate', 'feelings', 'sweet'],
            'angry': ['angry', 'anger', 'rage', 'furious', 'hate', 'mad', 'frustration', 'resentment',
                     'bitter', 'revenge', 'fight', 'aggressive', 'violence', 'fierce', 'intense', 'upset'],
            'confident': ['confident', 'strong', 'power', 'empowered', 'boss', 'independent', 'brave', 'courage',
                        'pride', 'self-love', 'fierce', 'belief', 'determined', 'hero', 'champion', 'winner']
        }
        
        # Activity keywords with weighted terms
        self.activity_keywords = {
            'workout': ['gym', 'exercise', 'fitness', 'workout', 'running', 'training', 'cardio', 'muscles',
                       'weights', 'sweat', 'strength', 'burn', 'routine', 'fit', 'physical', 'body'],
            'party': ['party', 'club', 'dance', 'dancing', 'celebrating', 'celebration', 'night out', 'drinks',
                     'weekend', 'drunk', 'fun', 'friends', 'social', 'wild', 'night', 'vibe'],
            'study': ['study', 'studying', 'homework', 'concentration', 'focus', 'learning', 'exam', 'school',
                     'university', 'college', 'class', 'lecture', 'reading', 'library', 'quiet', 'work'],
            'travel': ['travel', 'road trip', 'driving', 'journey', 'adventure', 'vacation', 'trip', 'drive',
                      'explore', 'destination', 'tourism', 'abroad', 'flight', 'foreign', 'culture', 'world'],
            'gaming': ['game', 'gaming', 'play', 'fortnite', 'minecraft', 'xbox', 'playstation', 'streaming',
                      'twitch', 'level', 'controller', 'online', 'character', 'virtual', 'video game', 'gamer'],
            'relaxation': ['sleep', 'rest', 'relax', 'bed', 'nap', 'chill', 'weekend', 'vacation', 
                          'break', 'peaceful', 'stress-free', 'calm', 'quiet', 'tranquil', 'comfortable', 'ease'],
            'cooking': ['cook', 'cooking', 'baking', 'recipe', 'kitchen', 'food', 'meal', 'dinner',
                       'breakfast', 'lunch', 'ingredients', 'delicious', 'flavor', 'taste', 'chef', 'dish'],
            'cleaning': ['clean', 'cleaning', 'organize', 'tidy', 'chores', 'housework', 'vacuum', 'sweep',
                       'mop', 'dust', 'laundry', 'wash', 'scrub', 'fresh', 'spotless', 'neat']
        }
        
        # Thematic keywords for content type
        self.theme_keywords = {
            'motivational': ['motivation', 'inspire', 'believe', 'dreams', 'success', 'achieve', 'overcome', 'goals',
                           'determination', 'perseverance', 'strength', 'courage', 'willpower', 'drive', 'ambitious'],
            'spiritual': ['spiritual', 'faith', 'god', 'soul', 'belief', 'prayer', 'divine', 'religion',
                         'worship', 'blessing', 'meditation', 'holy', 'sacred', 'spirit', 'peace'],
            'political': ['politics', 'government', 'election', 'democracy', 'president', 'vote', 'rights', 'freedom',
                         'policy', 'candidate', 'party', 'campaign', 'debate', 'liberal', 'conservative'],
            'educational': ['learn', 'education', 'knowledge', 'facts', 'information', 'teach', 'school', 'study',
                          'academic', 'research', 'science', 'history', 'lesson', 'tutorial', 'lecture'],
            'rebellious': ['rebel', 'revolution', 'fight', 'against', 'system', 'authority', 'rules', 'breaking',
                         'anarchy', 'resistance', 'protest', 'defiance', 'disobedience', 'rebellion', 'radical']
        }
        
        # Create weighted mood scores for more natural recommendations
        self.mood_weights = {
            'happy': {
                'sad': -0.8,       # Happy videos rarely work for sad moods
                'energetic': 0.5,  # Happy often works for energetic 
                'relaxing': -0.3,  # Happy less suitable for relaxing
                'confident': 0.4   # Happy can be good for confidence
            },
            'sad': {
                'happy': -0.7,     # Sad videos rarely work for happy moods
                'energetic': -0.6, # Sad not good for energetic
                'relaxing': 0.2,   # Sad can sometimes be relaxing
                'nostalgic': 0.5   # Sad often has nostalgic elements
            },
            'energetic': {
                'relaxing': -0.8,  # Energetic is opposite of relaxing
                'party': 0.7,      # Energetic is great for parties
                'workout': 0.8     # Energetic is perfect for workouts
            }
            # More relationships can be added
        }

    def load_data(self, limit=None):
        """
        Load videos and comments from the database
        
        Parameters:
        - limit: Optional limit on number of videos to load (for testing)
        """
        cache_file = os.path.join(self.cache_dir, 'data_cache.pkl')
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                print("Loading data from cache...")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.videos_df = cache_data['videos_df']
                    self.comments_df = cache_data['comments_df']
                    print(f"Loaded {len(self.videos_df)} videos and {len(self.comments_df)} comments from cache")
                    return self
            except Exception as e:
                print(f"Error loading from cache: {e}")
                print("Loading from database instead...")
        
        print("Loading data from database...")
        
        # Query to get video genres
        genre_query = """
        SELECT DISTINCT channel_title FROM videos
        """
        self.cursor.execute(genre_query)
        channels = [channel[0] for channel in self.cursor.fetchall()]
        
        # Attempt to map channels to genres based on channel name patterns
        self.genre_mapping = self._map_channels_to_genres(channels)
        
        # Load videos with limit if specified
        limit_clause = f"LIMIT {limit}" if limit else ""
        self.cursor.execute(f"""
            SELECT v.video_id, v.title, v.channel_title, v.view_count, v.like_count,
                   v.comment_count, v.duration, v.playlist_id
            FROM videos v
            {limit_clause}
        """)
        
        video_columns = ['video_id', 'title', 'channel_title', 'view_count', 'like_count', 
                          'comment_count', 'duration', 'playlist_id']
        self.videos_df = pd.DataFrame(self.cursor.fetchall(), columns=video_columns)
        
        # Rename columns for consistency in the code
        self.videos_df = self.videos_df.rename(columns={
            'channel_title': 'artist', 
            'view_count': 'views',
            'like_count': 'likes'
        })
        
        # Add genre column based on channel mapping
        self.videos_df['genre'] = self.videos_df['artist'].map(
            lambda artist: self._get_genre_for_channel(artist)
        )
        
        # Convert duration to seconds
        self.videos_df['duration_seconds'] = self.videos_df['duration'].apply(self._parse_duration)
        
        # Load comments (only top-level comments to reduce noise)
        if limit:
            # If we're limiting videos, only get comments for those videos
            video_ids = ", ".join([f"'{vid}'" for vid in self.videos_df['video_id']])
            self.cursor.execute(f"""
                SELECT c.comment_id, c.video_id, c.text, c.like_count, c.published_at
                FROM comments c
                WHERE c.is_reply = false
                AND c.video_id IN ({video_ids})
                AND length(c.text) BETWEEN 20 AND 1000  -- Filter out very short/long comments
            """)
        else:
            self.cursor.execute("""
                SELECT c.comment_id, c.video_id, c.text, c.like_count, c.published_at
                FROM comments c
                WHERE c.is_reply = false
                AND length(c.text) BETWEEN 20 AND 1000  -- Filter out very short/long comments
            """)
        
        comment_columns = ['comment_id', 'video_id', 'text', 'like_count', 'published_at']
        self.comments_df = pd.DataFrame(self.cursor.fetchall(), columns=comment_columns)
        
        # Ensure we have quality data
        self.comments_df['text'] = self.comments_df['text'].fillna('')
        self.comments_df = self.comments_df[self.comments_df['text'].str.len() > 20]
        
        # Cache the data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'videos_df': self.videos_df,
                    'comments_df': self.comments_df
                }, f)
            print(f"Saved data to cache: {cache_file}")
        except Exception as e:
            print(f"Error saving to cache: {e}")
        
        print(f"Loaded {len(self.videos_df)} videos and {len(self.comments_df)} comments from database")
        return self

    def _map_channels_to_genres(self, channels):
        """Map channel names to likely genres based on keywords"""
        genre_patterns = {
            'Pop': ['pop', 'hit', 'top40', 'vevo'],
            'Rock': ['rock', 'alternative', 'indie', 'band', 'guitar'],
            'Hip-Hop': ['rap', 'hip hop', 'hiphop', 'trap'],
            'Electronic': ['edm', 'electronic', 'dj', 'dance', 'house', 'techno'],
            'R&B': ['rnb', 'r&b', 'soul'],
            'Country': ['country', 'western'],
            'Latin': ['latin', 'latino', 'reggaeton', 'spanish'],
            'Classical': ['classical', 'orchestra', 'symphony'],
            'Jazz': ['jazz', 'blues', 'swing'],
            'Metal': ['metal', 'heavy', 'hardcore', 'death'],
            'Folk': ['folk', 'acoustic']
        }
        
        mapping = {}
        
        for channel in channels:
            if not channel:
                mapping[channel] = 'Unknown'
                continue
                
            channel_lower = channel.lower()
            matched = False
            
            # Check if channel name contains genre keywords
            for genre, patterns in genre_patterns.items():
                if any(pattern in channel_lower for pattern in patterns):
                    mapping[channel] = genre
                    matched = True
                    break
            
            # Default to Unknown if no match
            if not matched:
                mapping[channel] = 'Unknown'
        
        return mapping
    
    def _get_genre_for_channel(self, channel):
        """Get genre for a channel, using the mapping or a default"""
        if channel in self.genre_mapping:
            return self.genre_mapping[channel]
        return 'Unknown'
        
    def _parse_duration(self, duration_str):
        """Convert ISO 8601 duration to seconds"""
        if not duration_str or not isinstance(duration_str, str):
            return 0
            
        duration_str = duration_str.replace('PT', '')
        seconds = 0
        
        # Hours
        if 'H' in duration_str:
            hours, duration_str = duration_str.split('H')
            seconds += int(hours) * 3600
            
        # Minutes
        if 'M' in duration_str:
            minutes, duration_str = duration_str.split('M')
            seconds += int(minutes) * 60
            
        # Seconds
        if 'S' in duration_str:
            s = duration_str.replace('S', '')
            if s:
                seconds += int(s)
                
        return seconds

    def preprocess_comments(self):
        """Clean and preprocess comments with enhanced techniques"""
        cache_file = os.path.join(self.cache_dir, 'processed_comments.pkl')
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                print("Loading preprocessed comments from cache...")
                with open(cache_file, 'rb') as f:
                    self.comments_df = pickle.load(f)
                print("Successfully loaded preprocessed comments from cache")
                return self
            except Exception as e:
                print(f"Error loading from cache: {e}")
                print("Processing comments from scratch...")
        
        print("Preprocessing comments...")
        
        def clean_text(text):
            """Enhanced text cleaning with better noise removal"""
            if not isinstance(text, str):
                return ""
                
            # Convert HTML breaks to spaces
            text = re.sub(r'<br>|<br/>', ' ', text)
            
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Remove emojis and special characters but keep sentence structure
            text = re.sub(r'[^\w\s.,!?]', '', text)
            
            # Remove repeated characters (e.g., "looooove" -> "love")
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        # Clean the comments
        self.comments_df['cleaned_text'] = self.comments_df['text'].apply(clean_text)
        
        # Filter out empty comments after cleaning
        self.comments_df = self.comments_df[self.comments_df['cleaned_text'].str.len() > 20]
        
        # Analyze sentiment
        print("Analyzing sentiment in comments...")
        self.comments_df['sentiment'] = self.comments_df['cleaned_text'].apply(
            lambda text: self.sia.polarity_scores(text)['compound']
        )
        
        # Extract mood, activity, and theme indicators
        print("Extracting themed keywords from comments...")
        
        # Combine all keywords for vectorized operations
        all_keywords = {}
        all_keywords.update(self.mood_keywords)
        all_keywords.update(self.activity_keywords)
        all_keywords.update(self.theme_keywords)
        
        # Create columns for each category
        for category, keywords in tqdm(all_keywords.items(), desc="Categorizing comments"):
            # Count weighted occurrences
            # More specific/rare terms get higher weight
            self.comments_df[category] = self.comments_df['cleaned_text'].apply(
                lambda text: sum(1 + (0.5 * (keywords.index(word) / len(keywords)))
                               for word in keywords if word.lower() in text.lower())
            )
        
        # Cache preprocessed comments
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.comments_df, f)
            print(f"Saved preprocessed comments to cache: {cache_file}")
        except Exception as e:
            print(f"Error saving to cache: {e}")
        
        return self

    def create_embeddings(self, batch_size=32):
        """Create video embeddings using SBERT on aggregated comments with advanced features"""
        cache_file = os.path.join(self.cache_dir, 'video_features.pkl')
        embeddings_file = os.path.join(self.cache_dir, 'video_embeddings.npy')
        
        # Try to load from cache first
        if os.path.exists(cache_file) and os.path.exists(embeddings_file):
            try:
                print("Loading video features and embeddings from cache...")
                with open(cache_file, 'rb') as f:
                    self.video_features = pickle.load(f)
                self.video_embeddings = np.load(embeddings_file)
                print(f"Loaded features and embeddings for {len(self.video_features)} videos from cache")
                return self
            except Exception as e:
                print(f"Error loading from cache: {e}")
                print("Creating embeddings from scratch...")
        
        print("Creating video-level features...")
        
        # Get the videos with comments (some might not have any)
        videos_with_comments = set(self.comments_df['video_id'].unique())
        
        # Filter out videos without comments to avoid processing them
        videos_with_data = self.videos_df[self.videos_df['video_id'].isin(videos_with_comments)].copy()
        
        # Aggregate comments by video with more advanced metrics
        agg_dict = {
            'cleaned_text': lambda texts: ' '.join(texts),
            'like_count': 'sum',
            'sentiment': 'mean',
        }
        
        # Add all category columns to aggregation
        all_categories = list(self.mood_keywords.keys()) + list(self.activity_keywords.keys()) + list(self.theme_keywords.keys())
        for cat in all_categories:
            if cat in self.comments_df.columns:
                agg_dict[cat] = 'sum'
        
        # Group comments by video and aggregate
        video_comments = self.comments_df.groupby('video_id').agg(agg_dict).reset_index()
        
        # Normalize the feature scores
        for cat in all_categories:
            if cat in video_comments.columns:
                max_val = video_comments[cat].max()
                if max_val > 0:  # Avoid division by zero
                    video_comments[cat] = video_comments[cat] / max_val
        
        # Create TF-IDF features from the aggregated comments
        print("Creating TF-IDF features from comments...")
        tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5
        )
        
        # Check if there are videos with comments
        if len(video_comments) > 0:
            tfidf_matrix = tfidf.fit_transform(video_comments['cleaned_text'])
            
            # Use SVD to reduce dimensionality of TF-IDF
            svd = TruncatedSVD(n_components=20)
            tfidf_svd = svd.fit_transform(tfidf_matrix)
            
            # Create DataFrame with TF-IDF SVD components
            tfidf_cols = [f'tfidf_svd_{i}' for i in range(20)]
            tfidf_df = pd.DataFrame(tfidf_svd, columns=tfidf_cols)
            tfidf_df['video_id'] = video_comments['video_id'].values
            
            # Add TF-IDF components to video_comments
            video_comments = video_comments.merge(tfidf_df, on='video_id')
            
            # Generate SBERT embeddings
            print("Generating SBERT embeddings for video comments...")
            text_to_embed = video_comments['cleaned_text'].tolist()
            
            # Process in batches to avoid memory issues
            embeddings = []
            for i in tqdm(range(0, len(text_to_embed), batch_size), desc="Embedding batches"):
                batch = text_to_embed[i:i+batch_size]
                batch_embeddings = self.model.encode(batch)
                embeddings.extend(batch_embeddings)
            
            video_comments['embedding'] = embeddings
            
            # Rename column for consistency
            video_comments = video_comments.rename(columns={'like_count': 'comment_likes'})
            
            # Merge with video information
            self.video_features = videos_with_data.merge(
                video_comments, 
                on='video_id', 
                how='inner'  # Only keep videos that have comments
            )
            
            # Extract embeddings as numpy array
            self.video_embeddings = np.array(self.video_features['embedding'].tolist())
            
            # Cache the results
            try:
                with open(cache_file, 'wb') as f:
                    # Save without embeddings to reduce size
                    video_features_cache = self.video_features.drop(columns=['embedding'])
                    pickle.dump(video_features_cache, f)
                np.save(embeddings_file, self.video_embeddings)
                print(f"Saved video features and embeddings to cache")
            except Exception as e:
                print(f"Error saving to cache: {e}")
            
            print(f"Created embeddings for {len(self.video_features)} videos")
        else:
            print("No videos with comments found!")
            self.video_features = pd.DataFrame()
            self.video_embeddings = np.array([])
        
        return self

    def extract_playlist_features(self):
        """Extract features from playlist relationships"""
        if self.video_features is None or len(self.video_features) == 0:
            print("No video features available. Run create_embeddings() first.")
            return self
            
        print("Extracting playlist relationship features...")
        
        # Count videos per playlist
        playlist_counts = self.video_features['playlist_id'].value_counts()
        
        # Add playlist size as a feature
        self.video_features['playlist_size'] = self.video_features['playlist_id'].map(playlist_counts)
        
        # Calculate playlist similarity matrix
        playlists = self.video_features['playlist_id'].unique()
        playlist_matrix = np.zeros((len(playlists), len(playlists)))
        
        playlist_to_idx = {playlist: i for i, playlist in enumerate(playlists)}
        
        # For each video, find its playlist and increment co-occurrence counts
        for video_id in self.video_features['video_id'].unique():
            # Get all playlists containing this video
            video_playlists = self.video_features[self.video_features['video_id'] == video_id]['playlist_id'].unique()
            
            # Increment co-occurrence for each playlist pair
            for i, p1 in enumerate(video_playlists):
                idx1 = playlist_to_idx[p1]
                for p2 in video_playlists[i:]:
                    idx2 = playlist_to_idx[p2]
                    playlist_matrix[idx1, idx2] += 1
                    playlist_matrix[idx2, idx1] += 1
        
        # Normalize by playlist size to get similarity
        for i, p1 in enumerate(playlists):
            for j, p2 in enumerate(playlists):
                if i != j:
                    size1 = playlist_counts[p1]
                    size2 = playlist_counts[p2]
                    denominator = min(size1, size2)
                    if denominator > 0:
                        playlist_matrix[i, j] /= denominator
        
        # Store playlist similarity matrix for later use
        self.playlist_similarity = {playlists[i]: {playlists[j]: playlist_matrix[i, j] 
                                                for j in range(len(playlists))}
                                  for i in range(len(playlists))}
        
        return self

    def cluster_videos(self, n_clusters=15):
        """
        Cluster videos based on their embeddings and metadata
        to provide better diversity in recommendations
        """
        if self.video_embeddings is None or len(self.video_embeddings) == 0:
            print("No video embeddings available. Run create_embeddings() first.")
            return self
            
        cache_file = os.path.join(self.cache_dir, 'clusters.pkl')
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                print("Loading clusters from cache...")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.clusters = cache_data['clusters']
                    self.cluster_keywords = cache_data['cluster_keywords']
                print(f"Loaded {len(set(self.clusters))} clusters from cache")
                return self
            except Exception as e:
                print(f"Error loading clusters from cache: {e}")
                print("Creating clusters from scratch...")
        
        print(f"Clustering videos into {n_clusters} groups...")
        
        # Adjust number of clusters based on data size
        n_clusters = min(n_clusters, len(self.video_embeddings) // 10)
        n_clusters = max(n_clusters, 5)  # At least 5 clusters
        
        # Apply K-means clustering to embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.video_embeddings)
        
        # Add cluster labels to video features
        self.video_features['cluster'] = self.clusters
        
        # Find most common words and categories for each cluster
        self.cluster_keywords = {}
        
        for cluster_id in range(n_clusters):
            # Get videos in this cluster
            cluster_videos = self.video_features[self.video_features['cluster'] == cluster_id]
            
            # Get all comments for these videos
            cluster_comments = self.comments_df[self.comments_df['video_id'].isin(cluster_videos['video_id'])]
            
            # Extract common words
            if len(cluster_comments) > 0:
                all_text = ' '.join(cluster_comments['cleaned_text'])
                words = re.findall(r'\b\w{3,}\b', all_text.lower())
                word_counts = Counter(words)
                
                # Remove common stop words not already filtered
                for word in ['the', 'and', 'this', 'that', 'with', 'for', 'you', 'she', 'him', 'her']:
                    if word in word_counts:
                        del word_counts[word]
                
                # Get top mood and activity features
                mood_scores = {}
                for mood in self.mood_keywords:
                    if mood in cluster_videos.columns:
                        mood_scores[mood] = cluster_videos[mood].mean()
                        
                activity_scores = {}
                for activity in self.activity_keywords:
                    if activity in cluster_videos.columns:
                        activity_scores[activity] = cluster_videos[activity].mean()
                
                # Store cluster information
                self.cluster_keywords[cluster_id] = {
                    'top_words': word_counts.most_common(10),
                    'mood_scores': mood_scores,
                    'activity_scores': activity_scores,
                    'video_count': len(cluster_videos),
                    'genres': cluster_videos['genre'].value_counts().to_dict()
                }
            else:
                self.cluster_keywords[cluster_id] = {
                    'top_words': [],
                    'mood_scores': {},
                    'activity_scores': {},
                    'video_count': 0,
                    'genres': {}
                }
        
        # Cache the clusters
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'clusters': self.clusters,
                    'cluster_keywords': self.cluster_keywords
                }, f)
            print(f"Saved clusters to cache: {cache_file}")
        except Exception as e:
            print(f"Error saving clusters to cache: {e}")
        
        print(f"Created {n_clusters} clusters with average size of {len(self.video_features) / n_clusters:.1f} videos")
        return self

    def find_similar_videos(self, seed_idx, n=10, diversity_weight=0.3, exclude_seed=True):
        """
        Find videos similar to the seed video with enhanced diversity
        
        Parameters:
        - seed_idx: Index of seed video in video_features
        - n: Number of videos to return
        - diversity_weight: Weight for diversity (0.0 to 1.0, higher means more diverse results)
        - exclude_seed: Whether to exclude the seed video from results
        
        Returns:
        - Indices of similar videos and their similarity scores
        """
        if self.video_embeddings is None or len(self.video_embeddings) == 0:
            return [], []
            
        # Get the embedding of the seed video
        seed_embedding = self.video_embeddings[seed_idx].reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(seed_embedding, self.video_embeddings)[0]
        
        # Get seed video information
        seed_video = self.video_features.iloc[seed_idx]
        seed_cluster = seed_video['cluster'] if 'cluster' in seed_video else None
        seed_genre = seed_video['genre']
        seed_playlist = seed_video['playlist_id']
        
        # Apply diversity penalty
        if diversity_weight > 0 and 'cluster' in self.video_features:
            # Penalize videos from the same cluster
            for i, cluster in enumerate(self.video_features['cluster']):
                if cluster == seed_cluster:
                    similarities[i] -= diversity_weight * 0.2
            
            # Penalize videos from the same genre
            for i, genre in enumerate(self.video_features['genre']):
                if genre == seed_genre:
                    similarities[i] -= diversity_weight * 0.1
            
            # Penalize videos from the same playlist
            for i, playlist in enumerate(self.video_features['playlist_id']):
                if playlist == seed_playlist:
                    similarities[i] -= diversity_weight * 0.15
        
        # Sort by similarity (highest first)
        if exclude_seed:
            # Set the seed's similarity to -1 to exclude it
            similarities[seed_idx] = -1
            
        # Get top indices
        top_indices = similarities.argsort()[::-1][:n]
        
        return top_indices, similarities[top_indices]

    def generate_from_text(self, user_text, size=10, diversity=0.3):
        """
        Generate a playlist based on user-provided text with enhanced features
        
        Parameters:
        - user_text: Text provided by the user
        - size: Number of songs to include in the playlist
        - diversity: Level of diversity in recommendations (0.0 to 1.0)
        
        Returns:
        - List of dictionaries containing video information
        """
        if not user_text or len(user_text.strip()) < 20:
            print("Text is too short to analyze. Please provide more detailed text.")
            return self._generate_popular(size)
            
        if self.video_embeddings is None or len(self.video_embeddings) == 0:
            print("No video embeddings available. Run create_embeddings() first.")
            return self._generate_popular(size)
            
        print("Generating playlist based on user text...")
        
        # Clean and process the user text
        user_text = self._clean_user_text(user_text)
        
        # Analyze the sentiment of the text
        sentiment_score = self.sia.polarity_scores(user_text)['compound']
        print(f"Text sentiment score: {sentiment_score:.2f} (range: -1 to +1)")
        
        # Extract mood and activity keywords
        mood_scores = self._extract_mood_scores(user_text)
        activity_scores = self._extract_activity_scores(user_text)
        theme_scores = self._extract_theme_scores(user_text)
        
        # Find the dominant mood and activity
        dominant_mood = max(mood_scores.items(), key=lambda x: x[1])[0] if mood_scores else None
        dominant_mood_score = mood_scores.get(dominant_mood, 0) if dominant_mood else 0
        
        dominant_activity = max(activity_scores.items(), key=lambda x: x[1])[0] if activity_scores else None
        dominant_activity_score = activity_scores.get(dominant_activity, 0) if dominant_activity else 0
        
        dominant_theme = max(theme_scores.items(), key=lambda x: x[1])[0] if theme_scores else None
        dominant_theme_score = theme_scores.get(dominant_theme, 0) if dominant_theme else 0
        
        print(f"Dominant mood: {dominant_mood} (score: {dominant_mood_score:.2f})")
        print(f"Dominant activity: {dominant_activity} (score: {dominant_activity_score:.2f})")
        print(f"Dominant theme: {dominant_theme} (score: {dominant_theme_score:.2f})")
        
        # Get BERT embedding for the user text
        user_embedding = self.model.encode([user_text])[0].reshape(1, -1)
        
        # Calculate similarity with video embeddings
        similarities = cosine_similarity(user_embedding, self.video_embeddings)[0]
        
        # Create a combined score for each video with enhanced weighting
        scores = []
        for i, video in enumerate(self.video_features.itertuples()):
            # Base score is the semantic similarity
            score = similarities[i] * 0.4  # 40% weight on semantic similarity
            
            # Add sentiment alignment (videos with similar sentiment get a boost)
            video_sentiment = getattr(video, 'sentiment', 0)
            sentiment_alignment = 1 - abs(sentiment_score - video_sentiment) / 2  # Normalize to [0,1]
            score += sentiment_alignment * 0.15  # 15% weight on sentiment alignment
            
            # Add mood alignment with relationships between moods
            if dominant_mood and hasattr(video, dominant_mood):
                mood_alignment = getattr(video, dominant_mood, 0)
                score += mood_alignment * 0.15  # 15% weight to dominant mood
                
                # Consider relationships between moods
                for other_mood, weight in self.mood_weights.get(dominant_mood, {}).items():
                    if hasattr(video, other_mood):
                        # Apply cross-mood weighting (can be positive or negative)
                        cross_mood_score = getattr(video, other_mood, 0) * weight
                        score += cross_mood_score * 0.05
            
            # Add activity alignment
            if dominant_activity and hasattr(video, dominant_activity):
                activity_alignment = getattr(video, dominant_activity, 0)
                score += activity_alignment * 0.15  # 15% weight
                
            # Add theme alignment
            if dominant_theme and hasattr(video, dominant_theme):
                theme_alignment = getattr(video, dominant_theme, 0)
                score += theme_alignment * 0.1  # 10% weight
                
            # Consider duration preferences based on activity
            if hasattr(video, 'duration_seconds'):
                duration_score = 0
                duration = getattr(video, 'duration_seconds', 0)
                
                # Activity-specific duration preferences
                if dominant_activity == 'workout' and 180 <= duration <= 300:
                    # Workout songs often 3-5 minutes
                    duration_score = 0.05
                elif dominant_activity == 'study' and duration >= 300:
                    # Study music often longer
                    duration_score = 0.05
                elif dominant_activity == 'party' and duration <= 240:
                    # Party songs often shorter
                    duration_score = 0.05
                    
                score += duration_score
                
            # Add genre diversity based on text content
            if hasattr(video, 'genre'):
                genre = getattr(video, 'genre')
                genre_score = 0
                
                # Genre-mood alignments
                if dominant_mood == 'energetic' and genre in ['Electronic', 'Hip-Hop', 'Rock']:
                    genre_score = 0.05
                elif dominant_mood == 'relaxing' and genre in ['R&B', 'Classical', 'Folk']:
                    genre_score = 0.05
                elif dominant_mood == 'sad' and genre in ['Pop', 'R&B', 'Folk']:
                    genre_score = 0.05
                    
                score += genre_score
                
            # Apply popularity as a small boost (to break ties)
            popularity = min(1.0, np.log1p(getattr(video, 'views', 0)) / 20)  # Logarithmic scaling
            score += popularity * 0.05  # 5% weight
            
            scores.append((i, score, video))
        
        # Sort by combined score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top candidates by score
        top_candidates = scores[:size*2]  # Get 2x candidates for diversity
        
        # Apply diversity balancing if requested
        if diversity > 0 and len(top_candidates) > size:
            selected = [top_candidates[0]]  # Always select top match
            candidates = top_candidates[1:]
            
            while len(selected) < size and candidates:
                # For each candidate, calculate its average similarity to already selected items
                candidate_scores = []
                
                for i, (idx, score, video) in enumerate(candidates):
                    # Get embedding for this candidate
                    embedding = self.video_embeddings[idx].reshape(1, -1)
                    
                    # Calculate average similarity to selected videos
                    similarities = []
                    for sel_idx, _, _ in selected:
                        sel_embedding = self.video_embeddings[sel_idx].reshape(1, -1)
                        sim = cosine_similarity(embedding, sel_embedding)[0][0]
                        similarities.append(sim)
                    
                    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                    
                    # Combine original score with diversity penalty
                    diversity_penalty = avg_similarity * diversity
                    adjusted_score = score - diversity_penalty
                    
                    candidate_scores.append((i, adjusted_score))
                
                # Select the candidate with the highest adjusted score
                best_idx = max(candidate_scores, key=lambda x: x[1])[0]
                selected.append(candidates[best_idx])
                candidates.pop(best_idx)
            
            # Replace scores with the diverse selection
            scores = selected
        else:
            # Just take the top scoring videos
            scores = scores[:size]
        
        # Create the playlist
        playlist = []
        for i, score, video in scores:
            video_dict = {
                'video_id': video.video_id,
                'title': video.title,
                'artist': video.artist,
                'match_score': float(score),
                'genre': getattr(video, 'genre', 'Unknown'),
                'views': int(getattr(video, 'views', 0))
            }
            playlist.append(video_dict)
        
        print(f"Generated playlist with {len(playlist)} videos matching your text")
        return playlist

    def generate_genre_mix(self, genres, size=10, mood=None):
        """
        Generate a playlist with a specific genre distribution
        
        Parameters:
        - genres: Dictionary mapping genres to their proportions (e.g., {'Pop': 0.6, 'Rock': 0.4})
        - size: Total number of songs in the playlist
        - mood: Optional mood to filter by
        
        Returns:
        - List of dictionaries containing video information
        """
        if self.video_features is None or len(self.video_features) == 0:
            print("No video features available. Run create_embeddings() first.")
            return []
            
        print(f"Generating genre mix playlist: {genres}")
        
        # Normalize genre proportions
        total = sum(genres.values())
        genres = {g: p/total for g, p in genres.items()}
        
        # Calculate number of songs per genre
        genre_counts = {g: max(1, int(size * p)) for g, p in genres.items()}
        
        # Adjust to exactly match size
        total_allocated = sum(genre_counts.values())
        if total_allocated < size:
            # Add the remaining to the highest proportion genre
            top_genre = max(genres.items(), key=lambda x: x[1])[0]
            genre_counts[top_genre] += size - total_allocated
        elif total_allocated > size:
            # Remove from the lowest proportion genres
            sorted_genres = sorted(genres.items(), key=lambda x: x[1])
            for g, _ in sorted_genres:
                if genre_counts[g] > 1 and total_allocated > size:
                    genre_counts[g] -= 1
                    total_allocated -= 1
                if total_allocated == size:
                    break
        
        # Get videos for each genre
        playlist = []
        
        for genre, count in genre_counts.items():
            # Filter by genre
            genre_videos = self.video_features[self.video_features['genre'] == genre]
            
            # Apply mood filter if specified
            if mood and mood in self.mood_keywords:
                if mood in genre_videos.columns:
                    # Sort by mood score (descending)
                    genre_videos = genre_videos.sort_values(by=[mood, 'views'], ascending=[False, False])
            else:
                # Sort by popularity
                genre_videos = genre_videos.sort_values(by='views', ascending=False)
            
            # Get top videos
            top_videos = genre_videos.head(count)
            
            # Add to playlist
            for _, video in top_videos.iterrows():
                playlist.append({
                    'video_id': video['video_id'],
                    'title': video['title'],
                    'artist': video['artist'],
                    'genre': video['genre'],
                    'views': int(video['views'])
                })
        
        # Shuffle the playlist to mix genres
        random.shuffle(playlist)
        
        print(f"Generated genre mix playlist with {len(playlist)} videos")
        return playlist

    def generate_playlist(self, mode='popular', seed_video_id=None, mood=None, 
                         activity=None, user_text=None, genres=None, size=10, diversity=0.3):
        """
        Generate a playlist based on various criteria with enhanced features
        
        Parameters:
        - mode: Method to generate playlist
            'popular': Most popular videos (default)
            'seed': Based on similarity to a seed video
            'mood': Based on a specific mood
            'activity': Based on a specific activity
            'text': Based on user-provided text
            'genre_mix': Based on specified genre distribution
            'random': Random selection
        - seed_video_id: Video ID to use as seed (required if mode='seed')
        - mood: Mood to base playlist on (required if mode='mood')
        - activity: Activity to base playlist on (required if mode='activity')
        - user_text: Text to analyze for playlist generation (required if mode='text')
        - genres: Dictionary of genre proportions (required if mode='genre_mix')
        - size: Number of videos to include in playlist
        - diversity: Level of diversity in recommendations (0.0 to 1.0)
        
        Returns:
        - List of dicts with video information
        """
        if not hasattr(self, 'video_features') or len(self.video_features) == 0:
            print("Features not created. Run load_data().preprocess_comments().create_embeddings() first.")
            return []
        
        if mode == 'seed' and seed_video_id:
            return self._generate_from_seed(seed_video_id, size, diversity)
        elif mode == 'mood' and mood:
            return self._generate_from_mood(mood, size, diversity)
        elif mode == 'activity' and activity:
            return self._generate_from_activity(activity, size, diversity)
        elif mode == 'text' and user_text:
            return self.generate_from_text(user_text, size, diversity)
        elif mode == 'genre_mix' and genres:
            return self.generate_genre_mix(genres, size, mood)
        elif mode == 'random':
            return self._generate_random(size)
        else:
            return self._generate_popular(size)
    
    def _generate_from_seed(self, seed_video_id, size=10, diversity=0.3):
        """Generate playlist based on similarity to seed video with diversity"""
        print(f"Generating playlist based on seed video: {seed_video_id}")
        
        # Find the index of the seed video
        seed_videos = self.video_features[self.video_features['video_id'] == seed_video_id]
        
        if len(seed_videos) == 0:
            print(f"Seed video {seed_video_id} not found. Generating popular playlist instead.")
            return self._generate_popular(size)
        
        seed_idx = seed_videos.index[0]
        seed_info = seed_videos.iloc[0]
        
        # Find similar videos with diversity
        similar_indices, similarities = self.find_similar_videos(seed_idx, n=size, 
                                                               diversity_weight=diversity,
                                                               exclude_seed=True)
        
        # Create the playlist
        playlist = []
        for idx, sim in zip(similar_indices, similarities):
            video = self.video_features.iloc[idx]
            playlist.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'artist': video['artist'],
                'similarity': float(sim),
                'genre': video['genre'],
                'views': int(video['views'])
            })
        
        print(f"Generated playlist of {len(playlist)} videos similar to '{seed_info['title']}'")
        return playlist
    
    def _generate_from_mood(self, mood, size=10, diversity=0.3):
        """Generate playlist based on a specific mood with enhanced features"""
        valid_moods = list(self.mood_keywords.keys())
        
        if mood not in valid_moods:
            print(f"Invalid mood '{mood}'. Choose from: {valid_moods}")
            return self._generate_popular(size)
        
        print(f"Generating '{mood}' playlist")
        
        # Check if the mood column exists
        if mood not in self.video_features.columns:
            print(f"Mood '{mood}' not found in video features. Generating popular playlist instead.")
            return self._generate_popular(size)
        
        # Sort videos by mood score and sentiment appropriately
        if mood in ['happy', 'energetic', 'confident']:
            # For positive moods, prefer positive sentiment
            sorted_videos = self.video_features.sort_values(
                by=[mood, 'sentiment', 'views'], 
                ascending=[False, False, False]
            )
        elif mood in ['sad', 'angry']:
            # For sad/angry moods, sentiment can be negative or positive based on preference
            # Here we sort with positive sentiment first
            sorted_videos = self.video_features.sort_values(
                by=[mood, 'views'], 
                ascending=[False, False]
            )
        else:
            # For neutral moods, sort just by the mood score
            sorted_videos = self.video_features.sort_values(
                by=[mood, 'views'], 
                ascending=[False, False]
            )
        
        # Apply genre diversity if diversity parameter is set
        selected_videos = []
        
        if diversity > 0:
            # Get top candidates (2x the requested size)
            candidates = sorted_videos.head(size * 3)
            seen_genres = set()
            
            # First, select the top match
            if len(candidates) > 0:
                selected_videos.append(candidates.iloc[0])
                if 'genre' in candidates.columns:
                    seen_genres.add(candidates.iloc[0]['genre'])
            
            # Choose remaining with genre diversity
            remaining = candidates.iloc[1:]
            
            while len(selected_videos) < size and len(remaining) > 0:
                # Prioritize unseen genres if we don't have many yet
                if len(seen_genres) < 3 and len(remaining) > size:
                    # Find the highest ranked video with a new genre
                    for i, video in remaining.iterrows():
                        if 'genre' in video and video['genre'] not in seen_genres:
                            selected_videos.append(video)
                            seen_genres.add(video['genre'])
                            remaining = remaining.drop(i)
                            break
                    else:
                        # If no new genres, just take the top one
                        selected_videos.append(remaining.iloc[0])
                        if 'genre' in remaining.columns:
                            seen_genres.add(remaining.iloc[0]['genre'])
                        remaining = remaining.iloc[1:]
                else:
                    # Just take the top one
                    selected_videos.append(remaining.iloc[0])
                    if 'genre' in remaining.columns and len(remaining) > 0:
                        seen_genres.add(remaining.iloc[0]['genre'])
                    remaining = remaining.iloc[1:]
        else:
            # Just take the top videos by mood score
            selected_videos = sorted_videos.head(size).to_dict('records')
        
        # Create the playlist
        playlist = []
        for video in selected_videos:
            if isinstance(video, pd.Series):
                video = video.to_dict()
                
            playlist.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'artist': video['artist'],
                'mood_score': float(video[mood]),
                'genre': video.get('genre', 'Unknown'),
                'views': int(video['views'])
            })
        
        print(f"Generated '{mood}' playlist with {len(playlist)} videos")
        return playlist
    
    def _generate_from_activity(self, activity, size=10, diversity=0.3):
        """Generate playlist based on a specific activity with enhanced features"""
        valid_activities = list(self.activity_keywords.keys())
        
        if activity not in valid_activities:
            print(f"Invalid activity '{activity}'. Choose from: {valid_activities}")
            return self._generate_popular(size)
        
        print(f"Generating '{activity}' playlist")
        
        # Check if the activity column exists
        if activity not in self.video_features.columns:
            print(f"Activity '{activity}' not found in video features. Generating popular playlist instead.")
            return self._generate_popular(size)
        
        # Sort videos by activity score
        sorted_videos = self.video_features.sort_values(
            by=[activity, 'views'], 
            ascending=[False, False]
        )
        
        # Apply activity-specific duration preferences
        if 'duration_seconds' in self.video_features.columns:
            # Create a duration score based on activity
            duration_scores = []
            
            for _, video in sorted_videos.iterrows():
                score = video[activity]  # Base score is the activity score
                duration = video['duration_seconds']
                
                # Activity-specific adjustments
                if activity == 'workout':
                    # Workout music: prefer 3-5 minute songs with high energy
                    if 180 <= duration <= 300:
                        score += 0.2
                    # Penalize very short or very long songs
                    elif duration < 120 or duration > 360:
                        score -= 0.1
                    
                    # Boost energetic songs
                    if 'energetic' in video and video['energetic'] > 0.5:
                        score += 0.1
                        
                elif activity == 'study':
                    # Study music: prefer longer, less distracting songs
                    if duration > 300:
                        score += 0.1
                    
                    # Boost relaxing songs
                    if 'relaxing' in video and video['relaxing'] > 0.5:
                        score += 0.2
                        
                elif activity == 'party':
                    # Party music: prefer shorter, upbeat songs
                    if 120 <= duration <= 240:
                        score += 0.1
                    
                    # Boost happy and energetic songs
                    if 'happy' in video and video['happy'] > 0.5:
                        score += 0.1
                    if 'energetic' in video and video['energetic'] > 0.5:
                        score += 0.1
                
                duration_scores.append((score, _))
            
            # Sort videos by the adjusted score
            duration_scores.sort(reverse=True)
            
            # Get indices in order
            indices = [idx for _, idx in duration_scores]
            
            # Reorder the videos
            sorted_videos = sorted_videos.loc[indices]
        
        # Apply diversity similar to mood-based playlist
        if diversity > 0:
            # Get top candidates
            candidates = sorted_videos.head(size * 3)
            selected_videos = []
            seen_genres = set()
            
            # Add diversity similar to mood-based playlist
            # (Same implementation as in _generate_from_mood)
            if len(candidates) > 0:
                selected_videos.append(candidates.iloc[0])
                if 'genre' in candidates.columns:
                    seen_genres.add(candidates.iloc[0]['genre'])
            
            remaining = candidates.iloc[1:]
            
            while len(selected_videos) < size and len(remaining) > 0:
                if len(seen_genres) < 3 and len(remaining) > size:
                    for i, video in remaining.iterrows():
                        if 'genre' in video and video['genre'] not in seen_genres:
                            selected_videos.append(video)
                            seen_genres.add(video['genre'])
                            remaining = remaining.drop(i)
                            break
                    else:
                        selected_videos.append(remaining.iloc[0])
                        if 'genre' in remaining.columns:
                            seen_genres.add(remaining.iloc[0]['genre'])
                        remaining = remaining.iloc[1:]
                else:
                    selected_videos.append(remaining.iloc[0])
                    if 'genre' in remaining.columns and len(remaining) > 0:
                        seen_genres.add(remaining.iloc[0]['genre'])
                    remaining = remaining.iloc[1:]
        else:
            # Just take the top videos by activity score
            selected_videos = sorted_videos.head(size).to_dict('records')
        
        # Create the playlist
        playlist = []
        for video in selected_videos:
            if isinstance(video, pd.Series):
                video = video.to_dict()
                
            playlist.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'artist': video['artist'],
                'activity_score': float(video[activity]),
                'genre': video.get('genre', 'Unknown'),
                'views': int(video['views']),
                'duration': video.get('duration', '')
            })
        
        print(f"Generated '{activity}' playlist with {len(playlist)} videos")
        return playlist
    
    def _generate_popular(self, size=10):
        """Generate playlist based on popularity with genre diversity"""
        print("Generating popular playlist")
        
        # Sort by views
        sorted_videos = self.video_features.sort_values(by='views', ascending=False)
        
        # Take the top videos, ensuring some genre diversity
        genres_seen = set()
        selected_videos = []
        
        for _, video in sorted_videos.iterrows():
            genre = video.get('genre', 'Unknown')
            
            # Limit to 3 videos per genre for diversity
            if genre in genres_seen and list(genres_seen).count(genre) >= 3:
                continue
                
            genres_seen.add(genre)
            selected_videos.append(video)
            
            if len(selected_videos) >= size:
                break
        
        # If we don't have enough videos with genre diversity, fill with most popular
        if len(selected_videos) < size:
            remaining = size - len(selected_videos)
            already_selected = {v['video_id'] for v in selected_videos}
            
            for _, video in sorted_videos.iterrows():
                if video['video_id'] not in already_selected:
                    selected_videos.append(video)
                    if len(selected_videos) >= size:
                        break
        
        # Create the playlist
        playlist = []
        for video in selected_videos:
            playlist.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'artist': video['artist'],
                'views': int(video['views']),
                'genre': video.get('genre', 'Unknown')
            })
        
        print(f"Generated popular playlist with {len(playlist)} videos")
        return playlist
    
    def _generate_random(self, size=10):
        """Generate a random playlist with genre diversity"""
        print("Generating random playlist")
        
        # Get random indices with genre diversity
        all_genres = self.video_features['genre'].unique() if 'genre' in self.video_features.columns else []
        
        playlist = []
        
        if len(all_genres) > 0:
            # Try to get videos from each genre
            videos_per_genre = max(1, size // len(all_genres))
            
            for genre in all_genres:
                genre_videos = self.video_features[self.video_features['genre'] == genre]
                if len(genre_videos) > 0:
                    # Get random videos from this genre
                    sampled = genre_videos.sample(min(videos_per_genre, len(genre_videos)))
                    for _, video in sampled.iterrows():
                        playlist.append({
                            'video_id': video['video_id'],
                            'title': video['title'],
                            'artist': video['artist'],
                            'genre': video['genre'],
                            'views': int(video['views'])
                        })
                        
                        if len(playlist) >= size:
                            break
                            
                if len(playlist) >= size:
                    break
        
        # If we don't have enough yet,
                # If we don't have enough yet, fill with random videos
        if len(playlist) < size:
            # Get random videos, excluding ones already selected
            selected_ids = {v['video_id'] for v in playlist}
            remaining_videos = self.video_features[~self.video_features['video_id'].isin(selected_ids)]
            
            if len(remaining_videos) > 0:
                # Sample the remaining videos
                sampled = remaining_videos.sample(min(size - len(playlist), len(remaining_videos)))
                for _, video in sampled.iterrows():
                    playlist.append({
                        'video_id': video['video_id'],
                        'title': video['title'],
                        'artist': video['artist'],
                        'genre': video.get('genre', 'Unknown'),
                        'views': int(video['views'])
                    })
        
        print(f"Generated random playlist with {len(playlist)} videos")
        return playlist

    def _clean_user_text(self, text):
        """Clean and normalize user-provided text"""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _extract_mood_scores(self, text):
        """Extract mood scores from text with enhanced sensitivity"""
        text_lower = text.lower()
        mood_scores = {}
        
        for mood, keywords in self.mood_keywords.items():
            # Count weighted occurrences
            count = 0
            for i, keyword in enumerate(keywords):
                # More specific words get higher weight
                weight = 1.0 - (0.5 * i / len(keywords))
                count += text_lower.count(keyword.lower()) * weight
                
            if count > 0:
                # Normalize by text length to avoid bias towards longer texts
                normalized_count = count / (len(text.split()) + 1)  # +1 to avoid division by zero
                mood_scores[mood] = normalized_count
        
        # Normalize the scores to [0,1] range
        if mood_scores:
            max_score = max(mood_scores.values())
            if max_score > 0:
                mood_scores = {mood: score/max_score for mood, score in mood_scores.items()}
        
        return mood_scores

    def _extract_activity_scores(self, text):
        """Extract activity scores from text with enhanced sensitivity"""
        text_lower = text.lower()
        activity_scores = {}
        
        for activity, keywords in self.activity_keywords.items():
            # Count weighted occurrences
            count = 0
            for i, keyword in enumerate(keywords):
                # More specific words get higher weight
                weight = 1.0 - (0.5 * i / len(keywords))
                count += text_lower.count(keyword.lower()) * weight
                
            if count > 0:
                # Normalize by text length to avoid bias towards longer texts
                normalized_count = count / (len(text.split()) + 1)  # +1 to avoid division by zero
                activity_scores[activity] = normalized_count
        
        # Normalize the scores to [0,1] range
        if activity_scores:
            max_score = max(activity_scores.values())
            if max_score > 0:
                activity_scores = {activity: score/max_score for activity, score in activity_scores.items()}
        
        return activity_scores
        
    def _extract_theme_scores(self, text):
        """Extract theme scores from text"""
        text_lower = text.lower()
        theme_scores = {}
        
        for theme, keywords in self.theme_keywords.items():
            # Count weighted occurrences
            count = 0
            for i, keyword in enumerate(keywords):
                # More specific words get higher weight
                weight = 1.0 - (0.5 * i / len(keywords))
                count += text_lower.count(keyword.lower()) * weight
                
            if count > 0:
                # Normalize by text length to avoid bias towards longer texts
                normalized_count = count / (len(text.split()) + 1)  # +1 to avoid division by zero
                theme_scores[theme] = normalized_count
        
        # Normalize the scores to [0,1] range
        if theme_scores:
            max_score = max(theme_scores.values())
            if max_score > 0:
                theme_scores = {theme: score/max_score for theme, score in theme_scores.items()}
        
        return theme_scores

    def get_video_info(self, video_id):
        """Get detailed information about a specific video with enhanced features"""
        if video_id not in self.video_features['video_id'].values:
            return None
            
        # Get video details
        video = self.video_features[self.video_features['video_id'] == video_id].iloc[0]
        
        # Get top comments for this video
        video_comments = self.comments_df[self.comments_df['video_id'] == video_id]
        top_comments = video_comments.sort_values(by='like_count', ascending=False).head(5)
        
        # Calculate most common words in comments
        all_text = ' '.join(video_comments['cleaned_text'])
        words = re.findall(r'\b\w+\b', all_text.lower())
        common_words = Counter(words).most_common(15)
        
        # Get mood profile
        mood_profile = {}
        for mood in self.mood_keywords:
            if mood in video:
                mood_profile[mood] = float(video[mood])
        
        # Get activity profile
        activity_profile = {}
        for activity in self.activity_keywords:
            if activity in video:
                activity_profile[activity] = float(video[activity])
        
        # Get theme profile
        theme_profile = {}
        for theme in self.theme_keywords:
            if theme in video:
                theme_profile[theme] = float(video[theme])
        
        # Format detailed information
        info = {
            'video_id': video_id,
            'title': video['title'],
            'artist': video['artist'],
            'genre': video.get('genre', 'Unknown'),
            'views': int(video['views']),
            'likes': int(video['likes']),
            'comment_count': int(video.get('comment_count', 0)),
            'duration': video.get('duration', ''),
            'duration_seconds': float(video.get('duration_seconds', 0)),
            'sentiment': float(video['sentiment']),
            'mood_profile': mood_profile,
            'activity_profile': activity_profile,
            'theme_profile': theme_profile,
            'top_comments': top_comments[['text', 'like_count']].values.tolist(),
            'common_words': common_words,
            'cluster': int(video['cluster']) if 'cluster' in video else None
        }
        
        # Get cluster information if available
        if 'cluster' in video and self.cluster_keywords:
            cluster_id = int(video['cluster'])
            if cluster_id in self.cluster_keywords:
                info['cluster_info'] = self.cluster_keywords[cluster_id]
        
        return info
    
    def get_genre_distribution(self):
        """Get the distribution of genres in the dataset"""
        if 'genre' not in self.video_features.columns:
            return {}
            
        genre_counts = self.video_features['genre'].value_counts().to_dict()
        total = sum(genre_counts.values())
        
        genre_distribution = {genre: count/total for genre, count in genre_counts.items()}
        
        return genre_distribution
        
    def recommend_mixed_moods(self, primary_mood, secondary_mood, size=10, ratio=0.7):
        """
        Generate a playlist that mixes two moods with a specified ratio
        
        Parameters:
        - primary_mood: Primary mood for the playlist
        - secondary_mood: Secondary mood to mix in
        - size: Number of songs to include
        - ratio: Proportion of primary mood (0.0 to 1.0)
        
        Returns:
        - List of dictionaries containing video information
        """
        valid_moods = list(self.mood_keywords.keys())
        
        if primary_mood not in valid_moods or secondary_mood not in valid_moods:
            print(f"Invalid mood. Choose from: {valid_moods}")
            return self._generate_popular(size)
            
        print(f"Generating mixed playlist: {primary_mood} ({ratio:.0%}) + {secondary_mood} ({(1-ratio):.0%})")
        
        # Calculate number of songs per mood
        primary_count = max(1, int(size * ratio))
        secondary_count = size - primary_count
        
        # Generate playlist for each mood
        primary_playlist = self._generate_from_mood(primary_mood, primary_count)
        secondary_playlist = self._generate_from_mood(secondary_mood, secondary_count)
        
        # Combine playlists
        combined_playlist = primary_playlist + secondary_playlist
        
        # Shuffle to mix the moods
        random.shuffle(combined_playlist)
        
        return combined_playlist
    
    def recommend_for_event(self, event_type, size=10):
        """
        Generate a playlist tailored for a specific type of event
        
        Parameters:
        - event_type: Type of event ('party', 'wedding', 'workout', 'dinner', 'roadtrip')
        - size: Number of songs to include
        
        Returns:
        - List of dictionaries containing video information
        """
        event_configurations = {
            'party': {
                'moods': {'happy': 0.4, 'energetic': 0.6},
                'genre_bias': {'Pop': 0.4, 'Electronic': 0.3, 'Hip-Hop': 0.3},
                'duration_range': (180, 240)  # 3-4 minutes
            },
            'wedding': {
                'moods': {'romantic': 0.6, 'happy': 0.4},
                'genre_bias': {'Pop': 0.5, 'R&B': 0.3, 'Classical': 0.2},
                'duration_range': (180, 300)  # 3-5 minutes
            },
            'workout': {
                'moods': {'energetic': 0.8, 'confident': 0.2},
                'genre_bias': {'Hip-Hop': 0.4, 'Rock': 0.3, 'Electronic': 0.3},
                'duration_range': (180, 240)  # 3-4 minutes
            },
            'dinner': {
                'moods': {'relaxing': 0.7, 'romantic': 0.3},
                'genre_bias': {'Jazz': 0.4, 'R&B': 0.3, 'Classical': 0.3},
                'duration_range': (180, 360)  # 3-6 minutes
            },
            'roadtrip': {
                'moods': {'happy': 0.4, 'energetic': 0.3, 'nostalgic': 0.3},
                'genre_bias': {'Rock': 0.4, 'Pop': 0.4, 'Country': 0.2},
                'duration_range': (180, 300)  # 3-5 minutes
            }
        }
        
        if event_type not in event_configurations:
            print(f"Unknown event type: {event_type}. Choose from: {list(event_configurations.keys())}")
            return self._generate_popular(size)
            
        config = event_configurations[event_type]
        print(f"Generating playlist for {event_type} event")
        
        # Select videos based on the event configuration
        # We'll score each video based on the event preferences
        
        scores = []
        for i, video in enumerate(self.video_features.itertuples()):
            score = 0
            
            # Score based on moods
            for mood, weight in config['moods'].items():
                if hasattr(video, mood):
                    mood_val = getattr(video, mood)
                    if isinstance(mood_val, (int, float)):  # Make sure it's a number
                        score += mood_val * weight
            
            # Score based on genre
            if hasattr(video, 'genre'):
                genre = getattr(video, 'genre')
                if genre in config['genre_bias']:
                    score += config['genre_bias'][genre] * 0.3
            
            # Score based on duration
            if hasattr(video, 'duration_seconds'):
                duration = getattr(video, 'duration_seconds')
                min_duration, max_duration = config['duration_range']
                if min_duration <= duration <= max_duration:
                    score += 0.2
            
            # Add a small popularity factor
            popularity = np.log1p(getattr(video, 'views', 0)) / 20
            score += popularity * 0.1
            
            scores.append((i, score, video))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create playlist
        playlist = []
        for i, score, video in scores[:size]:
            playlist.append({
                'video_id': video.video_id,
                'title': video.title,
                'artist': video.artist,
                'match_score': float(score),
                'genre': getattr(video, 'genre', 'Unknown'),
                'views': int(getattr(video, 'views', 0))
            })
        
        print(f"Generated {event_type} playlist with {len(playlist)} videos")
        return playlist
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("Database connection closed")


# Example usage
if __name__ == "__main__":
    # Initialize the recommender
    recommender = EnhancedMusicRecommender()
    
    # Load and process data
    recommender.load_data()  # Add limit=100 for testing with fewer videos
    recommender.preprocess_comments()
    recommender.create_embeddings()
    recommender.cluster_videos(n_clusters=15)
    
    # Example 1: Generate a playlist based on user text
    user_text = """
    I'm feeling really energetic today and want to go for a run in the park.
    The sun is shining and I'm in such a positive mood. I want some upbeat 
    music that will keep me motivated during my workout!
    """
    
    playlist = recommender.generate_from_text(user_text, size=5, diversity=0.3)
    print("\nWorkout Playlist based on your text:")
    for i, video in enumerate(playlist, 1):
        print(f"{i}. {video['title']} by {video['artist']} - {video['genre']} (match: {video['match_score']:.2f})")
    
    # Example 2: Generate a mixed-mood playlist
    mixed_playlist = recommender.recommend_mixed_moods('energetic', 'relaxing', size=5, ratio=0.6)
    print("\nMixed Mood Playlist (Energetic + Relaxing):")
    for i, video in enumerate(mixed_playlist, 1):
        mood_score = video.get('mood_score', 0)
        print(f"{i}. {video['title']} by {video['artist']} - {video['genre']}")
    
    # Example 3: Generate a playlist for a specific event
    party_playlist = recommender.recommend_for_event('party', size=5)
    print("\nParty Playlist:")
    for i, video in enumerate(party_playlist, 1):
        print(f"{i}. {video['title']} by {video['artist']} - {video['genre']} (match: {video['match_score']:.2f})")
    
    # Example 4: Generate a genre mix playlist
    genres = {'Pop': 0.4, 'Rock': 0.3, 'Hip-Hop': 0.3}
    genre_playlist = recommender.generate_genre_mix(genres, size=5)
    print("\nGenre Mix Playlist:")
    for i, video in enumerate(genre_playlist, 1):
        print(f"{i}. {video['title']} by {video['artist']} - {video['genre']}")
    
    # Close the connection
    recommender.close()