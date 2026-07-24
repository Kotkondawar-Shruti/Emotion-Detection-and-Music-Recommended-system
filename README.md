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

User Enters Text

↓

Text Preprocessing

↓

Stopword Removal and Stemming

↓

Tokenization

↓

Sequence Padding

↓

Emotion Prediction Model

↓

Predicted Emotion

↓

Emotion-Based Music Filtering

↓

Recommended Songs

↓

Audio Preview

---

## Machine Learning Pipeline

The input text passes through the following preprocessing steps:

Raw Text

↓

Convert to Lowercase

↓

Remove URLs

↓

Remove User Mentions

↓

Remove Special Characters

↓

Remove Extra Spaces

↓

Remove Stopwords

↓

Porter Stemming

↓

Tokenization

↓

Sequence Padding

↓

Emotion Prediction

The processed text is converted into numerical sequences using a tokenizer and padded to a fixed maximum length before being passed to the trained emotion classification model.

---
