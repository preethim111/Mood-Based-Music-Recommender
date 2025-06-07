# from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
# import os
# import numpy as np
# import pandas as pd
# import json
# from datetime import datetime
# import threading
# import time
# import uuid
# from werkzeug.serving import run_simple

# # Import your existing classes - make sure these files are in the same directory
# from enhanced_mood_analyzer import EnhancedMoodAnalyzer
# from user_class import User
# from music_recommender import EnhancedMusicRecommender

# app = Flask(__name__)
# app.secret_key = os.urandom(24)  # For session management

# # Global variables
# recommender = None
# loading_state = {
#     'is_loading': True,
#     'status': 'Initializing...',
#     'progress': 0
# }

# # User data directory
# USER_DATA_DIR = './user_data'
# os.makedirs(USER_DATA_DIR, exist_ok=True)

# # Initialize the recommender in the background
# def init_recommender():
#     global recommender, loading_state
    
#     try:
#         loading_state = {
#             'is_loading': True,
#             'status': 'Loading music data...',
#             'progress': 10
#         }
        
#         # Initialize recommender with path to your dataset
#         data_path = 'music_data_with_clusters.csv'  # Update this path to your dataset
#         recommender = EnhancedMusicRecommender(data_path)
        
#         loading_state = {
#             'is_loading': True,
#             'status': 'Initializing mood analyzer...',
#             'progress': 50
#         }
        
#         # Wait for recommender to be fully initialized
#         time.sleep(2)  # Add a small delay to ensure all components are ready
        
#         loading_state = {
#             'is_loading': False,
#             'status': 'Ready',
#             'progress': 100
#         }
        
#         print("Recommender initialized successfully!")
        
#     except Exception as e:
#         loading_state = {
#             'is_loading': False,
#             'status': f'Error: {str(e)}',
#             'progress': 100
#         }
#         print(f"Error initializing recommender: {e}")

# # Helper function to get or create user ID
# def get_user_id():
#     if 'user_id' not in session:
#         session['user_id'] = str(uuid.uuid4())
#     return session['user_id']

# # Format recommendations for display
# def format_recommendations_html(recommendations):
#     """Format recommendations for the web UI"""
#     results = []
#     for _, row in recommendations.iterrows():
#         song = {
#             'track_id': row.get('track_id', ''),
#             'track_name': row.get('track_name', ''),
#             'artists': row.get('artists', ''),
#             'primary_mood': row.get('primary_mood', ''),
#             'cluster': int(row.get('cluster', 0)),
#             'popularity': row.get('popularity', 0),
#         }
        
#         # Add additional fields if available
#         for field in ['similarity', 'predicted_rating', 'combined_score']:
#             if field in row and not pd.isna(row[field]):
#                 song[field] = round(float(row[field]), 2)
        
#         # Add explanation if available
#         if 'explanation' in row:
#             song['explanation'] = row['explanation']
            
#         results.append(song)
    
#     return results

# # Flask routes follow the same as before...
# # (include all the routes from the previous app.py)

# # Start initialization thread
# init_thread = threading.Thread(target=init_recommender)
# init_thread.daemon = True
# init_thread.start()

# if __name__ == '__main__':
#     # Run the app
#     run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
import threading
import time
import uuid
import traceback

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Global variables
recommender = None
loading_state = {
    'is_loading': True,
    'status': 'Initializing...',
    'progress': 0
}

# User data directory
USER_DATA_DIR = './user_data'
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

# Print debug info about the environment
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print(f"Python version: {sys.version}")

# Import the necessary classes with error handling
try:
    # First, make sure we can see what files are available
    print("Available Python files in directory:")
    python_files = [f for f in os.listdir('.') if f.endswith('.py')]
    print(python_files)
    
    # Now try to import our classes
    print("\nAttempting to import classes...")
    
    # Import the EnhancedMoodAnalyzer
    from enhanced_mood_analyzer import EnhancedMoodAnalyzer
    print("Successfully imported EnhancedMoodAnalyzer")
    
    # Test the analyzer
    test_analyzer = EnhancedMoodAnalyzer()
    print(f"Created analyzer with strategy: {test_analyzer.strategy}")
    test_result = test_analyzer.extract_mood_from_text("test mood")
    print(f"Test extraction successful, shape: {test_result.shape}")
    
    # Import the User class
    from user_class import User
    print("Successfully imported User class")
    
    # Import the EnhancedMusicRecommender
    from music_recommender import EnhancedMusicRecommender
    print("Successfully imported EnhancedMusicRecommender")
    
except Exception as e:
    print(f"Error during imports: {e}")
    traceback.print_exc()
    
    # Define fallback simple versions for development
    print("Using fallback classes")
    
    class EnhancedMoodAnalyzer:
        def __init__(self):
            self.strategy = "fallback"
            self.mood_features = ['mood_happy', 'mood_sad', 'mood_energetic', 'mood_relaxing', 
                             'mood_nostalgic', 'mood_romantic', 'mood_angry', 
                             'mood_confident', 'mood_workout', 'mood_party', 'mood_study']
            self.mood_names = [f.replace('mood_', '') for f in self.mood_features]
            
        def extract_mood_from_text(self, text):
            # Simple fallback that returns balanced mood vector
            return np.ones(len(self.mood_features)) / len(self.mood_features)
            
        def get_top_moods(self, mood_vector, top_n=3):
            # Return some dummy moods
            return [('happy', 0.3), ('energetic', 0.2), ('confident', 0.1)]
    
    class User:
        def __init__(self, user_id):
            self.user_id = user_id
            self.feedback_history = []
            self.recommendation_history = set()
            
        def record_feedback(self, track_id, rating, mood_vector, song_data):
            self.feedback_history.append({
                'timestamp': datetime.now().isoformat(),
                'song_id': track_id,
                'track_name': song_data.get('track_name', ''),
                'artists': song_data.get('artists', ''),
                'rating': rating
            })
            return True
            
        def get_recommendation_history(self):
            return self.recommendation_history
            
        def track_recommendation(self, song_ids):
            for song_id in song_ids:
                self.recommendation_history.add(song_id)
            self.last_recommendations_time = datetime.now().isoformat()
    
    # Save updated user data
            self.save_user_data()
                
        def reset_recommendation_history(self):
            self.recommendation_history = set()
    
    class EnhancedMusicRecommender:
        def __init__(self, music_data_path=None):
            self.music_df = pd.DataFrame()
            self.mood_analyzer = EnhancedMoodAnalyzer()
            self.mood_features = ['mood_happy', 'mood_sad', 'mood_energetic', 'mood_relaxing', 
                             'mood_nostalgic', 'mood_romantic', 'mood_angry', 
                             'mood_confident', 'mood_workout', 'mood_party', 'mood_study']
            # Create sample data
            if music_data_path and os.path.exists(music_data_path):
                try:
                    self.music_df = pd.read_csv(music_data_path)
                except:
                    self._create_sample_music_data()
            else:
                self._create_sample_music_data()
                
        def _create_sample_music_data(self, num_songs=20):
            # Create minimal sample data
            data = {
                'track_id': [f"T{i:06d}" for i in range(num_songs)],
                'track_name': [f"Sample Song {i}" for i in range(num_songs)],
                'artists': ['Sample Artist'] * num_songs,
                'cluster': [i % 5 for i in range(num_songs)],
                'primary_mood': ['happy', 'sad', 'energetic', 'relaxing', 'study'] * 4,
            }
            self.music_df = pd.DataFrame(data)
            
        def recommend_songs(self, text_input, user, top_n=10, diversity_factor=0.3):
            # Simple recommendation - just return first N songs
            results = self.music_df.head(top_n).copy()
            results['similarity'] = 0.8
            results['explanation'] = "Sample recommendation"
            
            # Track these recommendations
            if hasattr(user, 'track_recommendation') and 'track_id' in results.columns:
                user.track_recommendation(results['track_id'].tolist())
                
            return results

# Initialize the recommender in the background
def init_recommender():
    global recommender, loading_state
    
    try:
        loading_state = {
            'is_loading': True,
            'status': 'Loading mood analyzer...',
            'progress': 10
        }
        
        # First create and test the mood analyzer
        analyzer = EnhancedMoodAnalyzer()
        print(f"Created mood analyzer with strategy: {analyzer.strategy}")
        
        loading_state = {
            'is_loading': True,
            'status': 'Loading music data...',
            'progress': 30
        }
        
        # Initialize recommender with path to your dataset
        data_path = '/Users/preethimanne/Desktop/dataset ds3/archive (10)/final app/music_data_with_clusters.csv'
        if not os.path.exists(data_path):
            print(f"Warning: Dataset file {data_path} not found.")
            data_path = None  # Let the recommender handle the missing file
            
        loading_state = {
            'is_loading': True,
            'status': 'Creating recommender...',
            'progress': 50
        }
            
        recommender = EnhancedMusicRecommender(data_path)
        print("Created recommender successfully")
        
        loading_state = {
            'is_loading': True,
            'status': 'Finalizing setup...',
            'progress': 80
        }
        
        # Add a small delay to ensure all components are ready
        time.sleep(1)
        
        loading_state = {
            'is_loading': False,
            'status': 'Ready',
            'progress': 100
        }
        
        print("Recommender initialized successfully!")
        
    except Exception as e:
        loading_state = {
            'is_loading': False,
            'status': f'Error: {str(e)}',
            'progress': 100
        }
        print(f"Error initializing recommender: {e}")
        traceback.print_exc()

# Helper function to get or create user ID
def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

# Format recommendations for display
def format_recommendations_html(recommendations):
    """Format recommendations for the web UI"""
    results = []
    for _, row in recommendations.iterrows():
        song = {
            'track_id': row.get('track_id', ''),
            'track_name': row.get('track_name', ''),
            'artists': row.get('artists', ''),
            'primary_mood': row.get('primary_mood', ''),
        }
        
        # Add cluster if available
        if 'cluster' in row:
            try:
                song['cluster'] = int(row['cluster'])
            except:
                song['cluster'] = 0
        else:
            song['cluster'] = 0
            
        # Add popularity if available
        if 'popularity' in row:
            try:
                song['popularity'] = int(row['popularity'])
            except:
                song['popularity'] = 50
        else:
            song['popularity'] = 50
        
        # Add additional fields if available
        for field in ['similarity', 'predicted_rating', 'combined_score']:
            if field in row and not pd.isna(row[field]):
                try:
                    song[field] = round(float(row[field]), 2)
                except:
                    song[field] = 0.5
        
        # Add explanation if available
        if 'explanation' in row:
            song['explanation'] = row['explanation']
            
        results.append(song)
    
    return results

# Routes
@app.route('/')
def index():
    """Main page with mood input"""
    if loading_state['is_loading']:
        return render_template('loading.html', 
                              status=loading_state['status'], 
                              progress=loading_state['progress'])
    
    # Check if user is logged in
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user info
    user_id = get_user_id()
    user = User(user_id)
    feedback_count = len(user.feedback_history) if hasattr(user, 'feedback_history') else 0
    
    return render_template('index.html', 
                          username=session['username'], 
                          feedback_count=feedback_count,
                          personalization=feedback_count >= 5)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        if not username:
            flash('Please enter a username', 'danger')
            return redirect(url_for('login'))
        
        session['username'] = username
        session['user_id'] = f"user_{username.lower().replace(' ', '_')}"
        return redirect(url_for('index'))
    
    return render_template('login.html')
@app.route('/create_spotify_playlist')
def create_spotify_playlist():
    """Create a Spotify playlist with user's top-rated songs"""
    # Check if user is logged in
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user history
    user_id = get_user_id()
    user = User(user_id)
    
    if not hasattr(user, 'feedback_history') or len(user.feedback_history) < 3:
        flash('You need at least 3 song ratings to create a playlist.', 'info')
        return redirect(url_for('history'))
    
    # Get highly-rated songs (4-5 stars)
    top_songs = [item for item in user.feedback_history if item.get('rating', 0) >= 4]
    
    if not top_songs:
        flash('You need to rate some songs 4 or 5 stars to create a playlist.', 'info')
        return redirect(url_for('history'))
    
    # Extract track IDs
    track_ids = []
    for song in top_songs:
        if 'track_id' in song:
            track_ids.append(song['track_id'])
        elif 'song_id' in song:
            track_ids.append(song['song_id'])
    
    # Create a Spotify playlist link using their URI scheme
    spotify_track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]
    
    # Spotify has a URI limit, so we'll limit to 50 tracks
    spotify_track_uris = spotify_track_uris[:50]
    
    # Build a playlist URL that will work with Spotify's web interface
    spotify_url = f"https://open.spotify.com/track/{track_ids[0]}"
    
    return render_template('spotify_playlist.html',
                          top_songs=top_songs,
                          track_ids=track_ids,
                          spotify_url=spotify_url)

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/recommend', methods=['POST'])
def recommend():
    """Process mood input and generate recommendations"""
    if loading_state['is_loading'] or recommender is None:
        return jsonify({'error': 'Recommender not ready yet'}), 503
    
    # Get mood text from form
    mood_text = request.form.get('mood_text', '')
    
    if not mood_text or len(mood_text) < 3:
        flash('Please enter a valid mood description', 'danger')
        return redirect(url_for('index'))
    
    # try:
    #     # Get user
    #     user_id = get_user_id()
    #     user = User(user_id)
        
    #     # Get recommendations
    #     recommendations = recommender.recommend_songs(mood_text, user, top_n=10)
        
    #     # Store in session for reference
    #     session['last_recommendations'] = recommendations.to_dict('records')
    #     session['last_mood_text'] = mood_text
        
    #     # Format for display
    #     formatted_recommendations = format_recommendations_html(recommendations)
        
    #     return render_template('recommendations.html',
    #                           mood_text=mood_text,
    #                           recommendations=formatted_recommendations,
    #                           username=session.get('username', 'User'))
    try:
        # Get user
        user_id = get_user_id()
        user = User(user_id)
        
        # Get recommendations
        recommendations = recommender.recommend_songs(mood_text, user, top_n=10)
        
        # Store minimal data in session to reduce cookie size
        session['last_recommendations_ids'] = recommendations['track_id'].tolist()
        session['last_mood_text'] = mood_text
        
        # Store recommendations on the server side instead of in the session
        recommendation_file = f'./user_data/{user_id}_last_recommendations.json'
        with open(recommendation_file, 'w') as f:
            json.dump(recommendations.to_dict('records'), f)
        
        # Format for display
        formatted_recommendations = format_recommendations_html(recommendations)
        
        return render_template('recommendations.html',
                              mood_text=mood_text,
                              recommendations=formatted_recommendations,
                              username=session.get('username', 'User'))
    
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        traceback.print_exc()
        flash(f'Error generating recommendations. Please try a different description.', 'danger')
        return redirect(url_for('index'))

@app.route('/feedback', methods=['POST'])
def feedback():
    """Record user feedback on recommendations"""
    if loading_state['is_loading'] or recommender is None:
        return jsonify({'error': 'Recommender not ready yet'}), 503
    
    try:
        # Get form data
        track_id = request.form.get('track_id')
        rating = int(request.form.get('rating'))
        
        if not track_id or rating < 1 or rating > 5:
            flash('Invalid feedback data', 'danger')
            return redirect(url_for('index'))
        
        # Get user and last mood
        user_id = get_user_id()
        user = User(user_id)
        mood_text = session.get('last_mood_text', '')
        
        # Extract mood vector using the recommender's analyzer
        mood_vector = recommender.mood_analyzer.extract_mood_from_text(mood_text)
        
        # Find the song data from last recommendations
        song_data = None
        last_recs = session.get('last_recommendations', [])
        for rec in last_recs:
            if rec.get('track_id') == track_id:
                song_data = rec
                break
        
        if not song_data:
            # Try to get song data from the recommender's dataset
            song_df = recommender.music_df[recommender.music_df['track_id'] == track_id]
            if len(song_df) > 0:
                song_data = song_df.iloc[0].to_dict()
        
        if not song_data:
            song_data = {'track_id': track_id, 'track_name': 'Unknown Song', 'artists': 'Unknown Artist'}
            flash('Limited song data available for feedback', 'warning')
        
        # Record feedback
        user.record_feedback(track_id, rating, mood_vector, song_data)
        
        flash(f'Thank you for your rating of {rating}/5!', 'success')
        
        # Check if we still have recommendations file
        recommendation_file = f'./user_data/{user_id}_last_recommendations.json'
        if os.path.exists(recommendation_file):
            # Redirect back to recommendations page
            return redirect(url_for('recommendations'))
        else:
            # Redirect to home if recommendations are no longer available
            return redirect(url_for('index'))
    
    except Exception as e:
        print(f"Error recording feedback: {e}")
        traceback.print_exc()
        flash(f'Error recording feedback. Please try again.', 'danger')
        return redirect(url_for('index'))

@app.route('/recommendations')
def recommendations():
    """Display current recommendations"""
    # Check if user is logged in
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user ID
    user_id = get_user_id()
    
    # Check if we have recommendations
    recommendation_file = f'./user_data/{user_id}_last_recommendations.json'
    mood_text = session.get('last_mood_text', '')
    
    if not os.path.exists(recommendation_file):
        flash('No recommendations available. Try entering a mood first.', 'info')
        return redirect(url_for('index'))
    
    # Load recommendations from file
    try:
        with open(recommendation_file, 'r') as f:
            last_recs = json.load(f)
        
        # Format recommendations for display
        formatted_recommendations = []
        for rec in last_recs:
            # Convert any numpy or non-serializable types to standard Python types
            recommendation = {}
            for key, value in rec.items():
                if key in ['similarity', 'predicted_rating', 'combined_score']:
                    recommendation[key] = float(value) if value is not None else 0.0
                else:
                    recommendation[key] = value
            formatted_recommendations.append(recommendation)
        
        return render_template('recommendations.html',
                              mood_text=mood_text,
                              recommendations=formatted_recommendations,
                              username=session.get('username', 'User'))
    except Exception as e:
        print(f"Error loading recommendations: {e}")
        flash('Error loading recommendations. Please try a new search.', 'danger')
        return redirect(url_for('index'))

@app.route('/history')
def history():
    """Display user's feedback history"""
    # Check if user is logged in
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user history
    user_id = get_user_id()
    user = User(user_id)
    
    if not hasattr(user, 'feedback_history') or len(user.feedback_history) == 0:
        flash('No feedback history available yet.', 'info')
        return redirect(url_for('index'))
    
    # Format history for display
    history_items = []
    for item in user.feedback_history:
        history_items.append({
            'track_name': item.get('track_name', 'Unknown'),
            'artists': item.get('artists', 'Unknown'),
            'rating': item.get('rating', 0),
            'timestamp': item.get('timestamp', ''),
            'primary_mood': item.get('primary_mood', 'Unknown'),
        })
    
    return render_template('history.html',
                          history=history_items,
                          username=session.get('username', 'User'))

@app.route('/status')
def status():
    """API endpoint to check loading status"""
    return jsonify(loading_state)

@app.route('/reset_history', methods=['POST'])
def reset_history():
    """Reset the user's recommendation history"""
    # Check if user is logged in
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user and reset history
    user_id = get_user_id()
    user = User(user_id)
    
    if hasattr(user, 'reset_recommendation_history'):
        user.reset_recommendation_history()
        flash('Your recommendation history has been reset', 'success')
    
    return redirect(url_for('index'))

@app.errorhandler(404)
@app.errorhandler(500)
def handle_error(e):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error {e.code}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="card shadow">
                <div class="card-body text-center p-5">
                    <h1 class="text-danger mb-4">Error {e.code}</h1>
                    <p class="lead">{e.name}</p>
                    <a href="/" class="btn btn-primary mt-3">Go Home</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """, e.code
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content

# Start initialization thread
init_thread = threading.Thread(target=init_recommender)
init_thread.daemon = True
init_thread.start()

if __name__ == '__main__':
    # Run the app without the reloader to avoid termios issues
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)

