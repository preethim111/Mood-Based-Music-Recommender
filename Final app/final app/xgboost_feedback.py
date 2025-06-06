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
class XGBoostFeedbackEngine:
    """
    Advanced feedback engine using XGBoost to learn from user feedback
    and continually improve personalized recommendations.
    """
    
    def __init__(self):
        self.feature_importance = {}  # Track feature importance
        self.error_tracking = {}  # Track errors for propagation
    
    def analyze_user_feedback(self, user):
        """
        Analyze a user's feedback to extract insights.
        
        Parameters:
        -----------
        user : User
            User object containing feedback history
            
        Returns:
        --------
        dict
            Insights about user preferences
        """
        if not user.feedback_history or len(user.feedback_history) < 3:
            return {"status": "insufficient_data"}
        
        feedback_df = pd.DataFrame(user.feedback_history)
        
        # Calculate average rating
        avg_rating = feedback_df['rating'].mean()
        
        # Find favorite and least favorite clusters
        if 'cluster' in feedback_df.columns:
            cluster_ratings = feedback_df.groupby('cluster')['rating'].mean()
            favorite_cluster = cluster_ratings.idxmax()
            least_favorite = cluster_ratings.idxmin()
        else:
            favorite_cluster = None
            least_favorite = None
        
        # Get feature importance if model exists
        feature_importance = {}
        if user.preference_model:
            X, _ = user._prepare_training_data()
            if X is not None:
                feature_names = X.columns
                importance_values = user.preference_model.feature_importances_
                feature_importance = dict(zip(feature_names, importance_values))
                
                # Update global feature tracking
                for feature, importance in feature_importance.items():
                    if feature not in self.feature_importance:
                        self.feature_importance[feature] = []
                    self.feature_importance[feature].append(importance)
        
        # Return insights
        return {
            "status": "success",
            "avg_rating": avg_rating,
            "favorite_cluster": favorite_cluster,
            "least_favorite_cluster": least_favorite,
            "feature_importance": feature_importance,
            "feedback_count": len(user.feedback_history)
        }
    
    def propagate_errors(self, user, song_id, rating, threshold=3.0):
        """
        Propagate errors from negative feedback to similar songs.
        This helps the system learn from mistakes faster.
        
        Parameters:
        -----------
        user : User
            User object containing feedback history
        song_id : str
            ID of the song with negative feedback
        rating : float
            User rating (1-5)
        threshold : float
            Rating threshold below which feedback is considered negative
        """
        if rating >= threshold:
            return  # Only propagate errors for negative feedback
        
        # Get the song data
        song_data = None
        for entry in user.feedback_history:
            if entry['song_id'] == song_id:
                song_data = entry
                break
        
        if not song_data:
            return
            
        # Track error for this song
        if song_id not in self.error_tracking:
            self.error_tracking[song_id] = {'count': 0, 'ratings': []}
        
        self.error_tracking[song_id]['count'] += 1
        self.error_tracking[song_id]['ratings'].append(rating)
        
        # We can't propagate if we don't have cluster information
        if 'cluster' not in song_data:
            return
            
        # Mark the cluster as problematic for this user
        cluster = song_data['cluster']
        primary_mood = song_data.get('primary_mood', None)
        
        # For future recommendations, this will help avoid similar songs
        penalty_factor = (threshold - rating) / threshold  # Higher for worse ratings
        
        return {
            'user_id': user.user_id,
            'problematic_cluster': cluster,
            'problematic_mood': primary_mood,
            'penalty_factor': penalty_factor
        }
    
    def update_song_weights(self, user, song_df):
        """
        Update song weights based on user feedback and model predictions.
        
        Parameters:
        -----------
        user : User
            User object containing feedback history
        song_df : pandas.DataFrame
            DataFrame of songs with predicted ratings
            
        Returns:
        --------
        pandas.DataFrame
            Songs with updated weights
        """
        if len(user.feedback_history) < 3:
            # Not enough feedback to adjust weights
            song_df['weight'] = 1.0
            return song_df
            
        # Create a copy to avoid modifying the original
        updated_df = song_df.copy()
        
        # Add weight column if it doesn't exist
        if 'weight' not in updated_df.columns:
            updated_df['weight'] = 1.0
        
        # Gather feedback insights
        insights = self.analyze_user_feedback(user)
        
        if insights['status'] != 'success':
            return updated_df
            
        # Adjust weights based on cluster preferences
        if 'cluster' in updated_df.columns and insights['favorite_cluster'] is not None:
            # Boost songs in favorite clusters
            favorite_boost = 1.3  # 30% boost
            updated_df.loc[updated_df['cluster'] == insights['favorite_cluster'], 'weight'] *= favorite_boost
            
            # Penalize songs in disliked clusters
            if insights['least_favorite_cluster'] is not None:
                dislike_penalty = 0.7  # 30% penalty
                updated_df.loc[updated_df['cluster'] == insights['least_favorite_cluster'], 'weight'] *= dislike_penalty
        
        # Apply error propagation - reduce weight of similar songs to those rated poorly
        problematic_songs = [e for e in user.feedback_history if e.get('rating', 5) < 3.0]
        if problematic_songs:
            for entry in problematic_songs:
                if 'cluster' in entry and 'cluster' in updated_df.columns:
                    # Reduce weight of songs in the same cluster as poorly rated songs
                    problematic_cluster = entry['cluster']
                    rating = entry['rating']
                    penalty = 0.8 + (rating / 10.0)  # Higher rating = milder penalty
                    updated_df.loc[updated_df['cluster'] == problematic_cluster, 'weight'] *= penalty
        
        # Adjust weights based on predicted ratings
        if 'predicted_rating' in updated_df.columns:
            # Scale weights based on predicted ratings (1-5) to boost highly predicted songs
            rating_factor = updated_df['predicted_rating'] / 3.0  # Normalize around 3.0
            updated_df['weight'] = updated_df['weight'] * rating_factor
        
        # Check for problematic features based on feedback
        if insights.get('feature_importance') and len(insights['feature_importance']) > 0:
            # Find the most important features that predict poor ratings
            top_features = sorted(
                insights['feature_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for feature, importance in top_features:
                if feature.startswith('query_'):
                    # This is a mood feature, might be worth considering
                    mood_name = feature.replace('query_', '')
                    
                    # Check if this mood consistently results in poor ratings
                    # This requires more complex analysis we'll skip for now
                    pass
        
        # Ensure weights are reasonable values
        updated_df['weight'] = np.clip(updated_df['weight'], 0.1, 5.0)
        
        return updated_df
    
    def get_global_insights(self):
        """
        Get insights from all user feedback across the system.
        
        Returns:
        --------
        dict
            Global insights about feature importance and error patterns
        """
        if not self.feature_importance:
            return {'status': 'no_data'}
        
        # Calculate average importance for each feature
        avg_importance = {}
        for feature, values in self.feature_importance.items():
            avg_importance[feature] = sum(values) / len(values)
        
        # Find most important features
        top_features = sorted(
            avg_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Analyze error patterns
        error_patterns = {}
        if self.error_tracking:
            # Count total errors by song
            song_error_counts = {song_id: data['count'] for song_id, data in self.error_tracking.items()}
            
            # Find most problematic songs
            problematic_songs = sorted(
                song_error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            error_patterns = {
                'problematic_songs': problematic_songs,
                'total_error_count': sum(data['count'] for data in self.error_tracking.values())
            }
        
        return {
            'status': 'success',
            'top_features': top_features,
            'error_patterns': error_patterns,
            'feature_count': len(self.feature_importance)
        }