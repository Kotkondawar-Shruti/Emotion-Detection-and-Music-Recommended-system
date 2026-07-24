# Emotion Detection and Music Recommendation System

An Artificial Intelligence and Machine Learning-based application that detects a user's emotional state from text input and recommends music based on the predicted emotion.

This project combines Natural Language Processing (NLP), Machine Learning, and content-based music recommendation techniques to create an intelligent and interactive music recommendation system.

---

## Overview

Music can have a significant impact on human emotions and mood. This project aims to provide personalized music recommendations by analyzing the emotional context of a user's text.

The system follows a two-stage process:

1. The user's text is preprocessed using Natural Language Processing techniques.
2. A trained emotion classification model predicts the emotion expressed in the text.
3. Based on the predicted emotion, suitable songs are filtered using audio features such as valence and energy.
4. The system displays personalized song recommendations through an interactive Streamlit web application.

---

## Features

- Emotion detection from user-provided text
- Natural Language Processing-based text preprocessing
- Lowercase conversion
- URL removal
- User mention removal
- Special character removal
- Stopword removal
- Porter Stemming
- Text tokenization and sequence padding
- Machine learning-based emotion classification
- Emotion-based music recommendation
- Audio feature-based song filtering
- Up to five song recommendations
- Spotify audio preview support where available
- Interactive Streamlit web application

---

## System Workflow

```text
User Enters Text
        |
        v
Text Preprocessing
        |
        v
Stopword Removal and Stemming
        |
        v
Tokenization
        |
        v
Sequence Padding
        |
        v
Emotion Prediction Model
        |
        v
Predicted Emotion
        |
        v
Emotion-Based Music Filtering
        |
        v
Recommended Songs
        |
        v
Audio Preview

The processed text is converted into numerical sequences using a tokenizer and padded to a fixed maximum length before being passed to the trained emotion classification model.

Supported Emotions

The system supports the following emotion categories:

Joy
Love
Surprise
Sadness
Anger
Neutral
Music Recommendation Logic

After predicting the user's emotion, the system filters songs based on audio features.

Joy, Love and Surprise

For positive emotions, songs with higher valence and energy are recommended.

Valence > 0.6
Energy > 0.6
Sadness and Anger

For sadness and anger, songs with lower valence are selected.

Valence < 0.4
Other Emotions

For other emotional states, songs with moderate valence are selected.

0.4 <= Valence <= 0.6

The system then selects up to five songs from the filtered results and displays the song name, artist, and available audio preview.

Technologies Used
Component	Technology
Programming Language	Python
Machine Learning	TensorFlow, Keras
Natural Language Processing	NLTK
Data Processing	Pandas, NumPy
Machine Learning Utilities	Scikit-learn
Web Application	Streamlit
Data Format	CSV
Project Structure
Emotion-Detection-and-Music-Recommendation-System/
|
|-- app.py
|-- Music_Recommendation_System_Source_code.ipynb
|-- OVERSAMPLING.ipynb
|-- README.md
File Description
File	Description
app.py	Main Streamlit application
Music_Recommendation_System_Source_code.ipynb	Main project development and machine learning notebook
OVERSAMPLING.ipynb	Notebook used for handling class imbalance through oversampling
README.md	Project documentation
Dataset

The datasets used in this project are not included in this repository because of their large file sizes.

The project uses the following datasets:

Tweet Emotions Dataset

This dataset is used for emotion classification from text.

It contains text data associated with different emotional categories and is used during the emotion detection model development process.

Music Information Dataset

This dataset is used for music recommendation.

It contains song information and audio features used to filter and recommend songs based on the predicted emotion.

The datasets are required to reproduce the complete training and recommendation pipeline.

Installation and Setup
1. Clone the Repository
git clone https://github.com/Kotkondawar-Shruti/Emotion-Detection-and-Music-Recommendation-System.git

Navigate to the project directory:

cd Emotion-Detection-and-Music-Recommendation-System
2. Create a Virtual Environment

Creating a virtual environment is recommended to keep project dependencies isolated.

Windows
python -m venv venv
venv\Scripts\activate
macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Required Libraries

Install the required dependencies:

pip install streamlit pandas numpy nltk tensorflow scikit-learn
Running the Application

Start the Streamlit application using:

streamlit run app.py

The application will be available at:

http://localhost:8501
