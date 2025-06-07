# 🎵 Mood-Based Music Recommendation System

This project is an intelligent, emotion-aware music recommendation system that generates personalized song suggestions by analyzing user mood, feedback, and audio features. It combines natural language processing, unsupervised learning, and adaptive machine learning to recommend music that matches how the user feels and evolves with their preferences over time.

---

## 🚀 Key Features

- **Emotion Extraction**: Extracts 11-dimensional mood vectors from user input using Sentence Transformers or fallback keyword matching.
- **Clustering for Mood Matching**: Uses K-Means to group songs into mood-based clusters for efficient mood-aligned recommendations.
- **Fast Intra-Cluster Search**: Matches user mood to a cluster centroid and returns similar songs via cosine similarity, achieving fast response time.
- **Adaptive Personalization**: Tracks user feedback (ratings, listening behavior) and uses XGBoost to personalize future recommendations.
- **User Profiling**: Maintains an evolving mood profile and weighted preferences per user for long-term customization.

---

## 🧠 Architecture Overview

1. **Input**: User provides a mood description or activity (e.g., “I want chill music to study”).
2. **Mood Analysis**: The `EnhancedMoodAnalyzer` class generates a mood vector.
3. **Cluster Matching**: User mood is matched to the closest song cluster using Euclidean distance to centroids.
4. **Candidate Ranking**:
   - **Cosine similarity** measures alignment with mood.
   - **XGBoost prediction** uses user history to estimate ratings.
   - A weighted `combined_score` blends both.
5. **Feedback Loop**: When a user rates a song, the system:
   - Updates weighted preference scores.
   - Adjusts the user's mood profile.
   - Retrains the XGBoost model if enough feedback is available.

---

[![Watch the demo]()](https://youtu.be/tufQCsaO4QQ)

