from enhanced_mood_analyzer import EnhancedMoodAnalyzer
from xgboost_feedback import XGBoostFeedbackEngine
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
class EnhancedMusicRecommender:
    """
    Advanced music recommender that incorporates user feedback, mood analysis,
    diversity mechanisms, and personalized recommendations.
    """
    
    def __init__(self, music_data_path="/Users/ishaangosain/Desktop/dataset ds3/archive (10)/final app/music_data_with_clusters.csv"):
        """
        Initialize the enhanced music recommender.
        
        Parameters:
        -----------
        music_data_path : str, optional
            Path to the CSV file containing music data with clusters
        """
        # Initialize components
        self.mood_analyzer = EnhancedMoodAnalyzer()
        self.feedback_engine = XGBoostFeedbackEngine()
        
        # Define mood features
        self.mood_features = [
            'mood_happy', 'mood_sad', 'mood_energetic', 'mood_relaxing', 
            'mood_nostalgic', 'mood_romantic', 'mood_angry', 
            'mood_confident', 'mood_workout', 'mood_party', 'mood_study'
        ]
        
        # Cluster descriptions for user-friendly output
        self.cluster_descriptions = {
            0: "Energetic & Confident - Perfect for workouts and confidence boosting",
            1: "Romantic & Nostalgic Study - Emotional music for focused study",
            2: "Intense & Angry - Powerful music for releasing energy",
            3: "Balanced & Versatile - Well-rounded music with moderate energy",
            4: "Relaxing & Melancholic - Calm music for deep focus or processing emotions"
        }
        
        # Load or create music dataset
        if music_data_path:
            try:
                self.music_df = pd.read_csv(music_data_path)
                print(f"Loaded {len(self.music_df)} songs from {music_data_path}")
            except:
                print(f"Could not load music data from {music_data_path}")
                self.music_df = self._create_sample_music_data()
        else:
            print("Creating sample music data...")
            self.music_df = self._create_sample_music_data()
        
        # Set cluster centers if not in data
        if 'cluster' in self.music_df.columns:
            self.cluster_centers = self._calculate_cluster_centers()
        else:
            # Default cluster centers if not available
            self.cluster_centers = np.array([
                [0.798, 0.229, 0.822, 0.146, 0.391, 0.410, 0.565, 0.803, 0.808, 0.774, 0.381],  # Cluster 0
                [0.486, 0.519, 0.537, 0.550, 0.689, 0.723, 0.471, 0.449, 0.559, 0.541, 0.788],  # Cluster 1
                [0.453, 0.545, 0.772, 0.142, 0.356, 0.272, 0.784, 0.532, 0.759, 0.573, 0.436],  # Cluster 2
                [0.651, 0.360, 0.694, 0.299, 0.543, 0.584, 0.530, 0.636, 0.693, 0.685, 0.563],  # Cluster 3
                [0.205, 0.766, 0.354, 0.727, 0.596, 0.531, 0.496, 0.186, 0.388, 0.304, 0.804]   # Cluster 4
            ])
        
        # Create a visualization of song clusters
        self._visualize_song_clusters()
    
    def _create_sample_music_data(self, num_songs=200):
        """
        Create a sample music dataset.
        
        Parameters:
        -----------
        num_songs : int
            Number of songs to generate
            
        Returns:
        --------
        pandas.DataFrame
            Synthetic music data
        """
        print(f"Generating sample music dataset with {num_songs} songs...")
        
        # Define popular artists
        artists = [
            'Taylor Swift', 'Drake', 'Ed Sheeran', 'Ariana Grande',
            'Post Malone', 'Billie Eilish', 'Justin Bieber', 'Lady Gaga',
            'The Weeknd', 'BTS', 'Dua Lipa', 'Beyoncé', 'Bad Bunny',
            'Harry Styles', 'Travis Scott', 'Olivia Rodrigo', 'Doja Cat',
            'Lil Nas X', 'Kendrick Lamar', 'SZA'
        ]
        
        # Generate random track names
        prefixes = ['Beautiful', 'Dark', 'Cold', 'Hot', 'Golden', 'Silver', 'Forever', 'Never', 'Time', 'Love']
        suffixes = ['Heart', 'Soul', 'Mind', 'Dreams', 'Nights', 'Days', 'Rhythm', 'Beat', 'Dance', 'Journey']
        
        track_names = []
        for _ in range(num_songs):
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            track_names.append(f"{prefix} {suffix}")
            
        # Generate track IDs
        track_ids = [f"T{i:06d}" for i in range(num_songs)]
        
        # Generate popularity scores (1-100)
        popularity = np.random.randint(30, 100, size=num_songs)
        
        # Generate audio features
        danceability = np.random.beta(5, 2, size=num_songs) * 0.8 + 0.1  # Between 0.1 and 0.9
        energy = np.random.beta(2, 2, size=num_songs) * 0.8 + 0.1
        valence = np.random.beta(2, 2, size=num_songs) * 0.8 + 0.1
        tempo = np.random.normal(120, 20, size=num_songs)  # Mean of 120 BPM
        acousticness = np.random.beta(2, 5, size=num_songs)  # Mostly lower values
        
        # Define cluster centers for different mood profiles
        cluster_centers = np.array([
            # Cluster 0: Energetic & Confident (workout, party)
            [0.8, 0.2, 0.9, 0.1, 0.3, 0.4, 0.5, 0.9, 0.9, 0.8, 0.3],
            
            # Cluster 1: Romantic & Nostalgic Study
            [0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.2, 0.5, 0.2, 0.3, 0.9],
            
            # Cluster 2: Intense & Angry
            [0.4, 0.6, 0.9, 0.1, 0.4, 0.2, 0.9, 0.6, 0.7, 0.5, 0.3],
            
            # Cluster 3: Balanced & Versatile
            [0.6, 0.4, 0.7, 0.3, 0.5, 0.6, 0.5, 0.6, 0.7, 0.7, 0.5],
            
            # Cluster 4: Relaxing & Melancholic
            [0.2, 0.8, 0.3, 0.9, 0.6, 0.5, 0.3, 0.2, 0.3, 0.3, 0.8]
        ])
        
        # Assign cluster randomly but with distribution
        cluster_probs = [0.2, 0.2, 0.15, 0.25, 0.2]  # Probability for each cluster
        clusters = np.random.choice(5, size=num_songs, p=cluster_probs)
        
        # Generate mood features based on cluster
        mood_features = []
        for i in range(num_songs):
            # Get base mood from cluster center
            base_mood = cluster_centers[clusters[i]]
            
            # Add random variation but keep within bounds
            variation = np.random.normal(0, 0.15, size=len(base_mood))
            mood = np.clip(base_mood + variation, 0.01, 0.99)
            
            # Correlation: more energetic songs have higher danceability
            if mood[2] > 0.7:  # mood_energetic
                danceability[i] = min(0.95, danceability[i] * 1.2)
            
            # Correlation: happier songs have higher valence
            if mood[0] > 0.7:  # mood_happy
                valence[i] = min(0.95, valence[i] * 1.3)
                
            # Correlation: more relaxing songs have higher acousticness
            if mood[3] > 0.7:  # mood_relaxing
                acousticness[i] = min(0.95, acousticness[i] * 1.5)
                
            mood_features.append(mood)
        
        mood_features = np.array(mood_features)
        
        # Determine primary mood for each song
        mood_names = ['happy', 'sad', 'energetic', 'relaxing', 'nostalgic', 
                     'romantic', 'angry', 'confident', 'workout', 'party', 'study']
        
        primary_moods = []
        for i in range(num_songs):
            # Find the highest mood value
            max_idx = np.argmax(mood_features[i])
            primary_moods.append(mood_names[max_idx])
        
        # Create DataFrame
        data = {
            'track_id': track_ids,
            'track_name': track_names,
            'artists': np.random.choice(artists, size=num_songs),
            'popularity': popularity,
            'danceability': danceability,
            'energy': energy,
            'valence': valence,
            'tempo': tempo,
            'acousticness': acousticness,
            'cluster': clusters,
            'primary_mood': primary_moods
        }
        
        # Add mood feature columns
        for i, mood in enumerate(self.mood_features):
            data[mood] = mood_features[:, i]
        
        # Create DataFrame
        music_df = pd.DataFrame(data)
        
        # Save to CSV file
        try:
            output_dir = '/home/user/output/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            music_df.to_csv(f"{output_dir}music_dataset.csv", index=False)
            print(f"Saved music dataset to {output_dir}music_dataset.csv")
        except Exception as e:
            print(f"Could not save dataset: {e}")
        
        return music_df
    
    def _calculate_cluster_centers(self):
        """Calculate cluster centers from the music data"""
        if 'cluster' not in self.music_df.columns:
            return None
        
        cluster_centers = []
        
        # Get unique clusters
        unique_clusters = sorted(self.music_df['cluster'].unique())
        
        # Calculate center for each cluster
        for cluster in unique_clusters:
            cluster_df = self.music_df[self.music_df['cluster'] == cluster]
            
            # Extract mood features
            mood_data = []
            for feature in self.mood_features:
                if feature in cluster_df.columns:
                    mood_data.append(cluster_df[feature].values)
            
            if mood_data:
                # Calculate mean across all mood features
                center = np.mean(np.array(mood_data).T, axis=0)
                cluster_centers.append(center)
        
        return np.array(cluster_centers)
    
    def _visualize_song_clusters(self):
        """Create a visualization of song clusters using PCA - disabled for web app"""
        print("Visualization of song clusters disabled for web app")
        return  # Skip visualization entirely
        
        # The code below won't run due to the return statement above
        try:
            from sklearn.decomposition import PCA
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Extract mood features
            mood_data = self.music_df[self.mood_features].values
            
            # Apply PCA to reduce dimensions to 2D
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(mood_data)
            
            # Get cluster assignments
            clusters = self.music_df['cluster'].values
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Plot each cluster with a different color
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i in range(5):  # Assuming 5 clusters
                # Plot songs in this cluster
                mask = clusters == i
                plt.scatter(
                    reduced_data[mask, 0], 
                    reduced_data[mask, 1], 
                    s=50, 
                    c=colors[i % len(colors)], 
                    label=f'Cluster {i}: {self.cluster_descriptions[i].split(" - ")[0]}'
                )
            
            # Plot cluster centers
            centers_reduced = pca.transform(self.cluster_centers)
            plt.scatter(
                centers_reduced[:, 0], 
                centers_reduced[:, 1], 
                s=200, 
                marker='*', 
                c='black', 
                label='Cluster Centers'
            )
            
            # Add labels and legend
            plt.title('Music Song Clusters Based on Mood Features', fontsize=16)
            plt.xlabel('PCA Dimension 1', fontsize=12)
            plt.ylabel('PCA Dimension 2', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save the figure
            output_dir = '/home/user/output/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(f"{output_dir}song_clusters.png", dpi=300, bbox_inches='tight')
            print(f"Saved cluster visualization to {output_dir}song_clusters.png")
            
        except Exception as e:
            print(f"Could not create cluster visualization: {e}")

    
    def recommend_songs(self, text_input, user, top_n=10, diversity_factor=0.3):
        """
        Recommend songs based on mood and user preferences.
        
        Parameters:
        -----------
        text_input : str
            User input text describing their mood
        user : User
            User object containing preferences and history
        top_n : int
            Number of songs to recommend
        diversity_factor : float
            Factor to control diversity in recommendations (0-1)
            
        Returns:
        --------
        pandas.DataFrame
            Recommended songs
        """
        print(f"Finding songs matching: '{text_input}'")
        
        # Extract mood vector from text using the enhanced analyzer
        mood_vector = self.mood_analyzer.extract_mood_from_text(text_input)
        
        # Print top moods for debugging
        top_moods = self.mood_analyzer.get_top_moods(mood_vector)
        print("Top moods detected:")
        for mood, score in top_moods:
            print(f"- {mood}: {score:.3f}")
            
        # Find matching songs
        recommendations = self._find_matching_songs(mood_vector, top_n*3)
        
        # Apply diversity - exclude previously recommended songs
        if hasattr(user, 'get_recommendation_history'):
            exclude_tracks = user.get_recommendation_history()
            if exclude_tracks and 'track_id' in recommendations.columns:
                recommendations = recommendations[~recommendations['track_id'].isin(exclude_tracks)]
        
        # If we have a user model, personalize recommendations
        personalize_weight = 0.5  # Default weight
        if hasattr(user, 'predict_song_preferences') and hasattr(user, 'get_personalization_weight'):
            # Predict preferences
            personalize_weight = user.get_personalization_weight()
            scored_recommendations = user.predict_song_preferences(recommendations, mood_vector)
            
            # Apply the feedback engine
            if hasattr(self, 'feedback_engine'):
                scored_recommendations = self.feedback_engine.update_song_weights(user, scored_recommendations)
            
            # Combine the base similarity score with personalized prediction
            scored_recommendations['combined_score'] = (
                (1 - personalize_weight) * scored_recommendations['similarity'] + 
                personalize_weight * (scored_recommendations['predicted_rating'] / 5.0)
            )
            
            # Sort by combined score
            recommendations = scored_recommendations.sort_values('combined_score', ascending=False)
        
        # Ensure diversity
        diverse_recommendations = self._ensure_diversity(recommendations, top_n, diversity_factor)
        
        # Track these recommendations
        if hasattr(user, 'track_recommendation') and 'track_id' in diverse_recommendations.columns:
            user.track_recommendation(diverse_recommendations['track_id'].tolist())
        
        # Enrich recommendations with explanations
        if len(diverse_recommendations) > 0:
            diverse_recommendations = self._add_recommendation_explanations(diverse_recommendations, mood_vector, user)
        
        return diverse_recommendations.head(top_n)
    
    def _find_matching_songs(self, mood_vector, top_n=10):
        """
        Find songs that match the given mood vector.
        
        Parameters:
        -----------
        mood_vector : numpy.ndarray
            Mood feature vector
        top_n : int
            Number of songs to return
            
        Returns:
        --------
        pandas.DataFrame
            Matching songs with similarity scores
        """
        # Extract mood features from songs
        song_mood_features = self.music_df[self.mood_features].values
        
        # Calculate cosine similarity between mood vector and each song
        similarities = cosine_similarity([mood_vector], song_mood_features)[0]
        
        # Add similarity scores to DataFrame
        result_df = self.music_df.copy()
        result_df['similarity'] = similarities
        
        # Sort by similarity
        result_df = result_df.sort_values('similarity', ascending=False)
        
        return result_df.head(top_n)
    
    def _ensure_diversity(self, recommendations, top_n=10, diversity_factor=0.3):
        """
        Ensure diversity in recommendations by balancing similarity with variety.
        
        Parameters:
        -----------
        recommendations : pandas.DataFrame
            Initial recommendations sorted by similarity
        top_n : int
            Number of songs to recommend
        diversity_factor : float
            Factor to control diversity (0-1), higher means more diverse
            
        Returns:
        --------
        pandas.DataFrame
            Diverse recommendations
        """
        if len(recommendations) <= top_n:
            return recommendations
        
        # Calculate how many songs to take from the top vs. diversify
        top_count = int(top_n * (1 - diversity_factor))
        diversity_count = top_n - top_count
        
        if top_count <= 0:
            top_count = 1
            diversity_count = top_n - 1
        
        # Get top songs based on similarity/score
        sorted_column = 'combined_score' if 'combined_score' in recommendations.columns else 'similarity'
        top_recommendations = recommendations.sort_values(sorted_column, ascending=False).head(top_count)
        
        # For diversity picks, use a combination of:
        # 1. Different clusters
        # 2. Different artists
        # 3. Different moods
        
        remaining_pool = recommendations.iloc[top_count:].copy()
        
        # Keep track of clusters and artists already in the recommendations
        selected_clusters = set(top_recommendations['cluster'].values) if 'cluster' in top_recommendations.columns else set()
        selected_artists = set(top_recommendations['artists'].values) if 'artists' in top_recommendations.columns else set()
        selected_moods = set(top_recommendations['primary_mood'].values) if 'primary_mood' in top_recommendations.columns else set()
        
        diverse_picks = []
        
        while len(diverse_picks) < diversity_count and len(remaining_pool) > 0:
            # Score each remaining song based on how different it is
            diversity_scores = []
            
            for i, row in remaining_pool.iterrows():
                score = 0.0
                
                # Boost score if from a new cluster
                if 'cluster' in row and row['cluster'] not in selected_clusters:
                    score += 2.0
                
                # Boost score if from a new artist
                if 'artists' in row and row['artists'] not in selected_artists:
                    score += 1.5
                
                # Boost score if from a new primary mood
                if 'primary_mood' in row and row['primary_mood'] not in selected_moods:
                    score += 1.0
                
                # Add some randomness to avoid always picking the same songs
                score += np.random.uniform(0, 0.5)
                
                # Still consider the original ranking (with a lower weight)
                original_score = row[sorted_column] if sorted_column in row else 0
                score += original_score * 0.3
                
                diversity_scores.append((i, score))
            
            # Sort by diversity score and pick the top one
            diversity_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx = diversity_scores[0][0]
            best_row = remaining_pool.loc[best_idx]
            
            # Add to diverse picks
            diverse_picks.append(best_row)
            
            # Update selected sets
            if 'cluster' in best_row:
                selected_clusters.add(best_row['cluster'])
            if 'artists' in best_row:
                selected_artists.add(best_row['artists'])
            if 'primary_mood' in best_row:
                selected_moods.add(best_row['primary_mood'])
            
            # Remove from remaining pool
            remaining_pool = remaining_pool.drop(best_idx)
        
        # Combine top recommendations with diverse picks
        diverse_recommendations = pd.concat([top_recommendations, pd.DataFrame(diverse_picks)])
        
        # Re-sort based on the original score to avoid too much disruption
        diverse_recommendations = diverse_recommendations.sort_values(sorted_column, ascending=False)
        
        return diverse_recommendations
    
    def _add_recommendation_explanations(self, recommendations, mood_vector, user):
        """
        Add explanations for why each song was recommended.
        
        Parameters:
        -----------
        recommendations : pandas.DataFrame
            Recommendations to explain
        mood_vector : numpy.ndarray
            Mood vector from the current query
        user : User
            User object containing preferences
            
        Returns:
        --------
        pandas.DataFrame
            Recommendations with explanations
        """
        # Create a copy to avoid modifying the original
        explained_df = recommendations.copy()
        
        # Add explanation column if it doesn't exist
        if 'explanation' not in explained_df.columns:
            explained_df['explanation'] = ""
        
        # Check what columns we have available
        has_similarity = 'similarity' in explained_df.columns
        has_predicted_rating = 'predicted_rating' in explained_df.columns
        has_combined_score = 'combined_score' in explained_df.columns
        has_cluster = 'cluster' in explained_df.columns
        has_primary_mood = 'primary_mood' in explained_df.columns
        
        # Generate explanations for each song
        for i, row in explained_df.iterrows():
            explanation_parts = []
            
            # Explain cluster
            if has_cluster and row['cluster'] in self.cluster_descriptions:
                cluster_desc = self.cluster_descriptions[row['cluster']].split(" - ")[0]
                explanation_parts.append(f"This song belongs to the {cluster_desc} cluster.")
            
            # Explain primary mood
            if has_primary_mood:
                explanation_parts.append(f"Its primary mood is {row['primary_mood']}.")
            
            # Explain mood match
            if has_similarity:
                match_percent = int(row['similarity'] * 100)
                explanation_parts.append(f"{match_percent}% match to your mood description.")
            
            # Explain predicted rating
            if has_predicted_rating:
                explanation_parts.append(f"Predicted rating: {row['predicted_rating']:.1f}/5.0 based on your preferences.")
            
            # Explain if user has rated songs by this artist before
            artist = row.get('artists', None)
            if artist and hasattr(user, 'feedback_history') and len(user.feedback_history) > 0:
                # Check if user has rated songs by this artist before
                artist_ratings = []
                for feedback in user.feedback_history:
                    if feedback.get('artists') == artist:
                        artist_ratings.append(feedback.get('rating', 0))
                
                if artist_ratings:
                    avg_artist_rating = sum(artist_ratings) / len(artist_ratings)
                    if avg_artist_rating >= 4.0:
                        explanation_parts.append(f"By {artist}, an artist you've liked before.")
                    elif avg_artist_rating <= 2.0:
                        explanation_parts.append(f"By {artist}, an artist you've previously rated lower.")
            
            # Explain diversity addition
            if i >= len(explained_df) * 0.7:  # Assuming later recommendations are for diversity
                explanation_parts.append("Added for variety.")
            
            # Combine explanations
            explained_df.at[i, 'explanation'] = " ".join(explanation_parts)
        
        return explained_df

    def visualize_user_preferences(self, users):
        """
        Create a visualization of user mood preferences.
        
        Parameters:
        -----------
        users : list
            List of User objects to visualize
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Check if we have users to visualize
            if not users or len(users) == 0:
                print("No users to visualize")
                return
            
            # Set up the figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Mood features to visualize
            moods = ['happy', 'sad', 'energetic', 'relaxing', 
                    'nostalgic', 'romantic', 'angry', 
                    'confident', 'workout', 'party', 'study']
            
            # Number of variables
            N = len(moods)
            
            # Angle of each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Plot each user
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
            
            for i, user in enumerate(users):
                # Get the user's mood profile
                if not hasattr(user, 'mood_profile') or user.mood_profile is None:
                    continue
                    
                values = user.mood_profile.tolist()
                values += values[:1]  # Close the loop
                
                # Plot the mood profile
                ax.plot(angles, values, linewidth=2, linestyle='solid', 
                    label=user.user_id, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
            
            # Set labels
            plt.xticks(angles[:-1], moods, size=12)
            
            # Set y labels
            ax.set_rlabel_position(0)
            plt.yticks([0.1, 0.2, 0.3], ['0.1', '0.2', '0.3'], color='grey', size=10)
            plt.ylim(0, 0.3)
            
            # Add title and legend
            plt.title('User Mood Preferences', size=16)
            plt.legend(loc='upper right')
            
            # Save the figure
            output_dir = '/home/user/output/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(f"{output_dir}user_preferences.png", dpi=300, bbox_inches='tight')
            print(f"Saved user preferences visualization to {output_dir}user_preferences.png")
            
        except Exception as e:
            print(f"Could not create user preferences visualization: {e}")

    def visualize_recommendation_flow(self):
        """Create a visualization of the recommendation flow"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.patches import Rectangle
            
            # Set up the figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define components and their positions
            components = [
                {'name': 'User Text Input', 'pos': (0.1, 0.85), 'width': 0.2, 'height': 0.1, 'color': 'lightblue'},
                {'name': 'User Feedback History', 'pos': (0.7, 0.85), 'width': 0.2, 'height': 0.1, 'color': 'lightgreen'},
                {'name': 'TF-IDF Mood Analyzer', 'pos': (0.1, 0.65), 'width': 0.2, 'height': 0.1, 'color': 'orange'},
                {'name': 'Mood Vector', 'pos': (0.1, 0.45), 'width': 0.2, 'height': 0.1, 'color': 'yellow'},
                {'name': 'XGBoost Personalization', 'pos': (0.7, 0.65), 'width': 0.2, 'height': 0.1, 'color': 'lightcoral'},
                {'name': 'User Preferences', 'pos': (0.7, 0.45), 'width': 0.2, 'height': 0.1, 'color': 'lightgreen'},
                {'name': 'Song Selection', 'pos': (0.4, 0.35), 'width': 0.2, 'height': 0.1, 'color': 'lightskyblue'},
                {'name': 'Diversity Mechanism', 'pos': (0.4, 0.15), 'width': 0.2, 'height': 0.1, 'color': 'plum'},
                {'name': 'Final Recommendations', 'pos': (0.4, 0.05), 'width': 0.2, 'height': 0.05, 'color': 'lightgreen'},
            ]
            
            # Define connection arrows
            arrows = [
                {'start': 'User Text Input', 'end': 'TF-IDF Mood Analyzer', 'color': 'gray'},
                {'start': 'TF-IDF Mood Analyzer', 'end': 'Mood Vector', 'color': 'gray'},
                {'start': 'User Feedback History', 'end': 'XGBoost Personalization', 'color': 'gray'},
                {'start': 'XGBoost Personalization', 'end': 'User Preferences', 'color': 'gray'},
                {'start': 'Mood Vector', 'end': 'Song Selection', 'color': 'gray'},
                {'start': 'User Preferences', 'end': 'Song Selection', 'color': 'gray'},
                {'start': 'Song Selection', 'end': 'Diversity Mechanism', 'color': 'gray'},
                {'start': 'Diversity Mechanism', 'end': 'Final Recommendations', 'color': 'gray'},
            ]
            
            # Create a component lookup
            component_lookup = {c['name']: c for c in components}
            
            # Draw arrows
            for arrow in arrows:
                start = component_lookup[arrow['start']]
                end = component_lookup[arrow['end']]
                
                # Calculate start and end positions
                start_x = start['pos'][0] + start['width'] / 2
                start_y = start['pos'][1]
                end_x = end['pos'][0] + end['width'] / 2
                end_y = end['pos'][1] + end['height']
                
                # Handle horizontal arrows
                if start_y == end_y + end['height']:
                    start_y -= start['height']
                    
                # Draw the arrow
                ax.annotate('', 
                        xy=(end_x, end_y), 
                        xytext=(start_x, start_y),
                        arrowprops=dict(facecolor=arrow['color'], shrink=0.05, width=1.5, headwidth=8))
            
            # Draw components
            for component in components:
                ax.add_patch(Rectangle(component['pos'], component['width'], component['height'], 
                                    fill=True, color=component['color'], alpha=0.7))
                
                # Add text
                ax.text(component['pos'][0] + component['width'] / 2, 
                    component['pos'][1] + component['height'] / 2,
                    component['name'], 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10, 
                    fontweight='bold')
            
            # Set plot properties
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Add title
            plt.title('Advanced Music Recommendation System', fontsize=16)
            
            # Save the figure
            output_dir = '/home/user/output/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(f"{output_dir}recommendation_flow.png", dpi=300, bbox_inches='tight')
            print(f"Saved recommendation flow visualization to {output_dir}recommendation_flow.png")
            
        except Exception as e:
            print(f"Could not create recommendation flow visualization: {e}")
            
    def create_system_summary(self):
        """Create a summary of the recommendation system"""
        try:
            import matplotlib.pyplot as plt
            import textwrap
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.axis('off')
            
            # Summary text
            summary = """
            # Music Recommendation System Summary

            ## Advanced Mood Analysis with TF-IDF
            • Uses TF-IDF vectorization to understand mood from text
            • Extracts nuanced mood vectors across 11 dimensions
            • More accurate than keyword matching approaches

            ## User Preference Tracking
            • Maintains individual user profiles
            • Tracks rating history and preferences
            • Evolves preferences based on feedback

            ## XGBoost-based Feedback Learning
            • Learns from user ratings and feedback
            • Continuously improves recommendations over time
            • Adapts to changing user preferences

            ## Diversity-Aware Recommendations
            • Prevents recommendation "echo chambers"
            • Ensures variety in artists, moods, and clusters
            • Balances similarity with exploration

            ## Weighted Song Selection
            • Combines mood matching with personalization
            • Uses weighted averages for better results
            • Adapts weights based on user feedback

            ## Personalization & Adaptation
            • Increases personalization as more feedback is collected
            • Provides transparent explanations for recommendations
            • Improves recommendations with each interaction
            """
            
            # Add text to figure
            ax.text(0.5, 0.5, summary, ha='center', va='center', 
                wrap=True, fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
            
            # Save the figure
            output_dir = '/home/user/output/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(f"{output_dir}system_summary.png", dpi=300, bbox_inches='tight')
            print(f"Saved system summary to {output_dir}system_summary.png")
            
        except Exception as e:
            print(f"Could not create system summary: {e}")
            
    def create_documentation(self):
        """Create comprehensive documentation for the recommendation system"""
        # Create system info text file
        output_dir = '/Users/ishaangosain/Desktop/dataset ds3/archive (10)/output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # System info text
        system_info = [
            "===== Advanced Music Recommendation System =====",
            "",
            "This system provides personalized music recommendations based on mood descriptions",
            "and continuously improves through user feedback using XGBoost learning.",
            "",
            "Dataset Statistics:",
            f"- Number of songs: {len(self.music_df)}",
            "- Number of mood clusters: 5",
            "- Number of mood dimensions: 11",
            "",
            "Key Components:",
            "",
            "1. TF-IDF Mood Analysis",
            "   Extracts mood vectors from natural language using TF-IDF vectorization",
            "   More accurate than simple keyword matching approaches",
            "",
            "2. User Preference Tracking",
            "   Maintains individual profiles for each user",
            "   Tracks feedback history and evolves preferences over time",
            "",
            "3. XGBoost Feedback Learning",
            "   Uses gradient boosting to learn from user ratings",
            "   Continuously improves recommendation quality",
            "",
            "4. Diversity-Aware Recommendations",
            "   Ensures variety in recommendations across artists, clusters, and moods",
            "   Prevents recommendation 'echo chambers'",
            "",
            "5. Personalization & Adaptation",
            "   Increases personalization as more feedback is collected",
            "   Provides transparent explanations for recommendations",
            "",
            "Usage:",
            "1. Enter a text description of your desired mood",
            "2. View and rate recommendations",
            "3. System learns from your ratings to improve future recommendations",
            "4. Get increasingly personalized suggestions over time",
            "",
            "===== End Documentation ====="
        ]
        
        # Write system info to file
        with open(f"{output_dir}system_info.txt", "w") as f:
            f.write("\n".join(system_info))
        print(f"Saved system information to {output_dir}system_info.txt")
        
        # Create README markdown file
        readme_md = [
            "# Advanced Music Recommendation System",
            "",
            "## Overview",
            "",
            "This system provides personalized music recommendations based on text descriptions of mood,",
            "using advanced TF-IDF vectorization, XGBoost-based personalization, and diversity-aware algorithms.",
            "It continuously improves through user feedback, building personalized models for each user.",
            "",
            "## Components",
            "",
            "### ImprovedMoodAnalyzer",
            "",
            "- Uses TF-IDF vectorization to extract mood vectors from text",
            "- Analyzes 11 mood dimensions: happy, sad, energetic, relaxing, nostalgic, romantic, angry, confident, workout, party, study",
            "- More nuanced understanding of mood compared to keyword matching",
            "",
            "### User Class",
            "",
            "- Tracks individual user preferences and feedback history",
            "- Maintains a personalized XGBoost model for each user",
            "- Handles weighted preferences and recommendation history",
            "- Evolves mood profile based on song ratings",
            "",
            "### XGBoostFeedbackEngine",
            "",
            "- Provides continuous learning from user feedback",
            "- Propagates errors from negative feedback to similar songs",
            "- Updates song weights based on user preferences",
            "- Provides global insights across all users",
            "",
            "### EnhancedMusicRecommender",
            "",
            "- Combines mood matching with personalization",
            "- Ensures diversity in recommendations",
            "- Adds explanations for transparency",
            "- Creates visualizations of song clusters and user preferences",
            "",
            "## How It Works",
            "",
            "1. User enters a text description of their desired mood",
            "2. System extracts a mood vector using TF-IDF analysis",
            "3. System finds songs that match this mood vector",
            "4. If the user has feedback history, personalization is applied using XGBoost",
            "5. Diversity mechanisms ensure varied recommendations",
            "6. User rates recommendations, which improves future suggestions",
            "7. System increasingly personalizes to each user's preferences over time",
            "",
            "## Output Files",
            "",
            "- `song_clusters.png`: Visualization of song clusters in 2D space",
            "- `user_preferences.png`: Radar chart of user mood preferences",
            "- `recommendation_flow.png`: Flowchart of the recommendation process",
            "- `system_summary.png`: Summary of system capabilities",
            "- `music_dataset.csv`: The synthetic music dataset used for recommendations",
            "- `system_info.txt`: Text information about the system",
            "",
            "## Implementation Details",
            "",
            "- The system uses cosine similarity to match mood vectors to songs",
            "- XGBoost models are trained for each user with sufficient feedback",
            "- Diversity is ensured by balancing similarity with variety in clusters, artists, and moods",
            "- The personalization weight increases as more user feedback is collected",
            "- Negative feedback propagates to similar songs to avoid repeated mistakes",
            "",
            "© 2024 Advanced Music Recommendation System"
        ]
        
        # Write README to file
        with open(f"{output_dir}README.md", "w") as f:
            f.write("\n".join(readme_md))
        print(f"Saved README to {output_dir}README.md")

