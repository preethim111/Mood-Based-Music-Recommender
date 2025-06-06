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
import xgboost as xgb
class User:
    """
    User class to track individual preferences, history, and personalized recommendations.
    Each user maintains their own feedback history and XGBoost model.
    """
    
    def __init__(self, user_id):
        """
        Initialize a user object.
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        """
        self.user_id = user_id
        self.feedback_history = []
        self.recommendation_history = set()  # Track recommended song IDs
        self.weighted_preferences = {}  # Track weighted song preferences
        self.last_mood_vector = None
        
        # XGBoost model for personalized recommendations
        self.preference_model = None
        self.feature_encoders = {}
        self.min_feedback_for_model = 5  # Minimum feedback needed to train model
        
        # User's mood profile - evolves as user interacts with system
        self.mood_profile = np.zeros(11)  # 11 mood dimensions
        
        # Load existing data if available
        self._load_user_data()
    
    def _load_user_data(self):
        """Load user data from file if it exists"""
        user_file = f'./user_data/{self.user_id}_data.json'
        
        if os.path.exists(user_file):
            try:
                with open(user_file, 'r') as f:
                    user_data = json.load(f)
                
                self.feedback_history = user_data.get('feedback_history', [])
                self.recommendation_history = set(user_data.get('recommendation_history', []))
                self.weighted_preferences = user_data.get('weighted_preferences', {})
                self.mood_profile = np.array(user_data.get('mood_profile', [0] * 11))
                
                print(f"Loaded data for user {self.user_id}: {len(self.feedback_history)} feedback entries")
                
                # Train preference model if enough feedback
                if len(self.feedback_history) >= self.min_feedback_for_model:
                    self._train_preference_model()
            except Exception as e:
                print(f"Error loading user data: {e}")
    
    def save_user_data(self):
        """Save user data to file"""
        user_file = f'./user_data/{self.user_id}_data.json'
        
        try:
            user_data = {
                'feedback_history': self.feedback_history,
                'recommendation_history': list(self.recommendation_history),
                'weighted_preferences': self.weighted_preferences,
                'mood_profile': self.mood_profile.tolist()
            }
            
            with open(user_file, 'w') as f:
                json.dump(user_data, f)
                
            print(f"Saved data for user {self.user_id}")
        except Exception as e:
            print(f"Error saving user data: {e}")
    
    def record_feedback(self, song_id, rating, mood_vector, song_data):
        """
        Record user feedback for a song.
        
        Parameters:
        -----------
        song_id : str
            ID of the rated song
        rating : float
            User rating (1-5)
        mood_vector : numpy.ndarray
            Mood vector from the query that led to this recommendation
        song_data : dict
            Song metadata and features
        """
        # Save the mood vector for later use
        self.last_mood_vector = mood_vector
        
        # Record timestamp
        timestamp = datetime.now().isoformat()
        
        # Create feedback entry
        feedback_entry = {
            'timestamp': timestamp,
            'song_id': song_id,
            'rating': float(rating),
            'mood_vector': mood_vector.tolist() if isinstance(mood_vector, np.ndarray) else mood_vector,
            'track_name': song_data.get('track_name', ''),
            'artists': song_data.get('artists', ''),
            'cluster': song_data.get('cluster', -1),
            'primary_mood': song_data.get('primary_mood', '')
        }
        
        # Add audio features if available
        for feature in ['danceability', 'energy', 'valence', 'tempo', 'acousticness']:
            if feature in song_data:
                feedback_entry[feature] = float(song_data[feature])
        
        # Add to feedback history
        self.feedback_history.append(feedback_entry)
        
        # Update weighted preferences
        if song_id not in self.weighted_preferences:
            self.weighted_preferences[song_id] = {'rating': rating, 'count': 1, 'last_updated': timestamp}
        else:
            # Update using exponential weighted average (recent ratings matter more)
            old_rating = self.weighted_preferences[song_id]['rating']
            old_count = self.weighted_preferences[song_id]['count']
            decay_factor = 0.8  # How much to weigh the old rating
            
            new_rating = (old_rating * decay_factor * old_count + rating) / (old_count * decay_factor + 1)
            new_count = old_count + 1
            
            self.weighted_preferences[song_id] = {
                'rating': new_rating, 
                'count': new_count,
                'last_updated': timestamp
            }
        
        # Update mood profile based on rating and mood of song
        # Higher ratings increase preference for that mood, lower ratings decrease it
        self._update_mood_profile(song_data, rating)
        
        # Save updated user data
        self.save_user_data()
        
        # Train or update preference model if we have enough feedback
        if len(self.feedback_history) >= self.min_feedback_for_model:
            self._train_preference_model()
            
        return True
    
    def _update_mood_profile(self, song_data, rating):
        """
        Update user's mood profile based on song rating.
        Higher ratings increase preference for that mood, lower ratings decrease it.
        
        Parameters:
        -----------
        song_data : dict
            Song data including mood features
        rating : float
            User rating (1-5)
        """
        # Extract mood features from song
        mood_features = []
        for i in range(11):
            feature_name = f'mood_{["happy", "sad", "energetic", "relaxing", "nostalgic", "romantic", "angry", "confident", "workout", "party", "study"][i]}'
            if feature_name in song_data:
                mood_features.append(float(song_data[feature_name]))
            else:
                mood_features.append(0.0)
        
        mood_features = np.array(mood_features)
        
        # Calculate adjustment factor based on rating
        # 1-2 stars: decrease preference, 3: neutral, 4-5 stars: increase preference
        if rating <= 2:
            adjustment = -0.05  # Decrease preference
        elif rating == 3:
            adjustment = 0.01   # Slight increase for neutral
        else:
            adjustment = 0.1    # Increase preference
        
        # Update profile by adjusting in the direction of the song's mood features
        # Higher rated songs pull the profile more toward their mood
        adjusted_features = mood_features * adjustment
        
        # Update mood profile
        self.mood_profile = np.clip(self.mood_profile + adjusted_features, 0, 1)
        
        # Normalize to sum to 1
        if np.sum(self.mood_profile) > 0:
            self.mood_profile = self.mood_profile / np.sum(self.mood_profile)
    
    def _prepare_training_data(self):
        """Prepare training data from feedback history for the XGBoost model"""
        if len(self.feedback_history) < self.min_feedback_for_model:
            return None, None
        
        # Create a dataframe from feedback
        df = pd.DataFrame(self.feedback_history)
        
        # Convert mood vector from list to separate columns
        mood_features = [
            'mood_happy', 'mood_sad', 'mood_energetic', 'mood_relaxing', 
            'mood_nostalgic', 'mood_romantic', 'mood_angry', 
            'mood_confident', 'mood_workout', 'mood_party', 'mood_study'
        ]
        
        # Convert JSON string to list if needed
        if isinstance(df['mood_vector'].iloc[0], str):
            df['mood_vector'] = df['mood_vector'].apply(json.loads)
        
        # Expand mood_vector into separate columns
        for i, feature in enumerate(mood_features):
            feature_name = f"query_{feature}"
            df[feature_name] = df['mood_vector'].apply(lambda x: x[i] if i < len(x) else 0)
        
        # Encode categorical features
        categorical_features = ['artists', 'primary_mood', 'cluster']
        for feature in categorical_features:
            if feature in df.columns:
                if feature not in self.feature_encoders:
                    self.feature_encoders[feature] = LabelEncoder()
                    df[f"{feature}_encoded"] = self.feature_encoders[feature].fit_transform(df[feature])
                else:
                    # Handle new categories
                    df[f"{feature}_encoded"] = df[feature].apply(
                        lambda x: -1 if x not in self.feature_encoders[feature].classes_ 
                                else self.feature_encoders[feature].transform([x])[0]
                    )
        
        # Select features for training
        features = [f"query_{f}" for f in mood_features]  # Mood query features
        
        # Add encoded categorical features
        for feature in categorical_features:
            if f"{feature}_encoded" in df.columns:
                features.append(f"{feature}_encoded")
        
        # Add audio features if available
        audio_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
        for feature in audio_features:
            if feature in df.columns:
                features.append(feature)
        
        # Only select columns that exist
        features = [f for f in features if f in df.columns]
        
        # Extract features and target
        X = df[features]
        y = df['rating'].values
        
        return X, y
    
    def _train_preference_model(self):
        """Train an XGBoost model to predict user preferences"""
        print(f"Training preference model for user {self.user_id}...")
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        if X is None or len(X) < self.min_feedback_for_model:
            print(f"Not enough feedback data to train model (need at least {self.min_feedback_for_model})")
            return
        
        # Split data for training and validation
        try:
            # Use a small test size when we have limited data
            test_size = min(0.2, 1/len(X))
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Initialize and train XGBoost model
            self.preference_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            # Train the model
            self.preference_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate the model
            train_preds = self.preference_model.predict(X_train)
            val_preds = self.preference_model.predict(X_val)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            
            print(f"Preference model trained. Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
            
        except Exception as e:
            print(f"Error training preference model: {e}")
            self.preference_model = None
    
    def predict_song_preferences(self, songs_df, mood_vector):
        """
        Predict preference scores for songs based on user's model.
        
        Parameters:
        -----------
        songs_df : pandas.DataFrame
            DataFrame of songs to score
        mood_vector : numpy.ndarray
            Mood vector from the current query
            
        Returns:
        --------
        pandas.DataFrame
            Songs with predicted ratings
        """
        if self.preference_model is None or len(self.feedback_history) < self.min_feedback_for_model:
            # Return default score of 3.0 if no model is available
            songs_df['predicted_rating'] = 3.0
            return songs_df
        
        # Create features for prediction
        prediction_data = songs_df.copy()
        
        # Add mood query features
        mood_features = [
            'mood_happy', 'mood_sad', 'mood_energetic', 'mood_relaxing', 
            'mood_nostalgic', 'mood_romantic', 'mood_angry', 
            'mood_confident', 'mood_workout', 'mood_party', 'mood_study'
        ]
        
        for i, feature in enumerate(mood_features):
            feature_name = f"query_{feature}"
            prediction_data[feature_name] = mood_vector[i] if i < len(mood_vector) else 0
        
        # Encode categorical features
        categorical_features = ['artists', 'primary_mood', 'cluster']
        for feature in categorical_features:
            if feature in prediction_data.columns and feature in self.feature_encoders:
                # Handle new categories
                prediction_data[f"{feature}_encoded"] = prediction_data[feature].apply(
                    lambda x: -1 if x not in self.feature_encoders[feature].classes_ 
                            else self.feature_encoders[feature].transform([x])[0]
                )
        
        # Select features for prediction
        features = [f"query_{f}" for f in mood_features]  # Mood query features
        
        # Add encoded categorical features
        for feature in categorical_features:
            if f"{feature}_encoded" in prediction_data.columns:
                features.append(f"{feature}_encoded")
        
        # Add audio features if available
        audio_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
        for feature in audio_features:
            if feature in prediction_data.columns:
                features.append(feature)
        
        # Only select columns that exist
        features = [f for f in features if f in prediction_data.columns]
        
        # Check if we have sufficient features
        if len(features) == 0:
            print("Warning: No matching features for prediction")
            prediction_data['predicted_rating'] = 3.0
            return prediction_data
        
        # Predict ratings
        try:
            X_pred = prediction_data[features]
            prediction_data['predicted_rating'] = self.preference_model.predict(X_pred)
            
            # Apply adjustments based on weighted preferences
            for i, row in prediction_data.iterrows():
                song_id = row['track_id']
                if song_id in self.weighted_preferences:
                    # Blend XGBoost prediction with weighted preference
                    xgb_weight = min(0.7, len(self.feedback_history) / 30)  # Increases as we get more feedback
                    pref_weight = 1 - xgb_weight
                    
                    xgb_pred = prediction_data.at[i, 'predicted_rating']
                    pref_rating = self.weighted_preferences[song_id]['rating']
                    
                    # Weighted blend
                    prediction_data.at[i, 'predicted_rating'] = (
                        xgb_weight * xgb_pred + pref_weight * pref_rating
                    )
            
        except Exception as e:
            print(f"Error predicting preferences: {e}")
            prediction_data['predicted_rating'] = 3.0
        
        return prediction_data
    
    def track_recommendation(self, song_ids):
        """
        Track songs that have been recommended to this user.
        
        Parameters:
        -----------
        song_ids : list
            List of song IDs that were recommended
        """
        for song_id in song_ids:
            self.recommendation_history.add(song_id)
        
        # Save updated user data
        self.save_user_data()
    
    def get_recommendation_history(self):
        """Get the set of songs previously recommended to this user"""
        return self.recommendation_history
    
    def reset_recommendation_history(self):
        """Reset the recommendation history"""
        self.recommendation_history = set()
        self.save_user_data()
        print(f"Reset recommendation history for user {self.user_id}")
    
    def get_personalization_weight(self):
        """
        Calculate how much to weigh personalization vs base recommendations.
        Returns higher values as more feedback is collected.
        """
        # Start with minimal personalization, increase with feedback
        base_weight = 0.3
        feedback_boost = min(0.5, len(self.feedback_history) * 0.05)
        return base_weight + feedback_boost
