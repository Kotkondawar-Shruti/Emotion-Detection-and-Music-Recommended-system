import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av # Required for video frame handling

# --- App Configuration ---
st.set_page_config(
    page_title="MoodMate | Music Recommender",
    page_icon="🎵",
    layout="wide" # Use a wider layout for tabs
)

# --- Backend Code ---
MODEL_PATH = 'models/final_tuned_vgg16_model.h5'
MUSIC_DATA_PATH = 'data/music_processed/processed_music_tags.csv'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml' # Path to your downloaded file
EMOTION_MAP = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# --- Caching Models and Data ---

@st.cache_resource
def load_emotion_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Emotion detection model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"🔴 Error loading emotion model: {e}")
        return None

@st.cache_resource
def load_face_detector():
    try:
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            st.error(f"🔴 Error loading Haar Cascade file. Make sure '{HAAR_CASCADE_PATH}' is in the correct folder.")
            return None
        print("✅ Face detector (Haar Cascade) loaded successfully.")
        return face_cascade
    except Exception as e:
        st.error(f"🔴 Error loading Haar Cascade: {e}")
        return None

@st.cache_data
def load_music_data():
    try:
        df = pd.read_csv(MUSIC_DATA_PATH)
        print("✅ Music data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"🔴 Error loading music data: {e}")
        return None

# --- Backend Functions ---

def recommend_songs(emotion, music_df, num_recommendations=5):
    """Recommends songs based on the 'tags' column."""
    if music_df is None: return pd.DataFrame()
    CORRECT_COLUMN_NAME = 'tags' # Using the correct column name
    
    if CORRECT_COLUMN_NAME not in music_df.columns:
        st.error(f"Column '{CORRECT_COLUMN_NAME}' not found in music data. Available columns: {list(music_df.columns)}")
        return pd.DataFrame()

    playlist = music_df[music_df[CORRECT_COLUMN_NAME] == emotion]
    if playlist.empty:
        return music_df.sample(n=num_recommendations) # Fallback to random songs
    return playlist.sample(n=min(len(playlist), num_recommendations))

# --- Real-Time Webcam Class ---

class EmotionVideoTransformer(VideoTransformerBase):
    """Processes video frames to detect faces and predict emotions."""
    def __init__(self):
        self.face_cascade = load_face_detector()
        self.emotion_model = load_emotion_model()

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.face_cascade is None or self.emotion_model is None:
            return frame # Return frame unmodified if models failed to load

        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract the face ROI (Region of Interest)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Preprocess for the emotion model
            roi_norm = roi_gray / 255.0
            roi_3ch = np.repeat(np.expand_dims(roi_norm, axis=-1), 3, axis=-1)
            roi_final = np.expand_dims(roi_3ch, axis=0)
            
            # Predict emotion
            prediction = self.emotion_model.predict(roi_final)
            emotion_index = np.argmax(prediction)
            emotion = EMOTION_MAP.get(emotion_index, "Unknown")
            
            # Put the emotion text above the rectangle
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert numpy array back to VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---

st.title("🎵 MoodMate | Emotion-Based Music Recommender")

# Create tabs
tab1, tab2 = st.tabs(["📁 Upload an Image", "📷 Live Webcam Detection"])

# --- Tab 1: Upload an Image ---
with tab1:
    st.header("Get a Playlist from an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        image_np = np.array(image)

        with st.spinner('Analyzing emotion and curating your playlist...'):
            emotion_model = load_emotion_model()
            music_df = load_music_data()

            if emotion_model is not None and music_df is not None:
                # --- This is the prediction function from your previous code ---
                if len(image_np.shape) > 2 and image_np.shape[2] in [3, 4]:
                    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = image_np
                
                img_resized = cv2.resize(gray_img, (48, 48))
                img_normalized = img_resized / 255.0
                img_3_channel = np.repeat(np.expand_dims(img_normalized, axis=-1), 3, axis=-1)
                img_final = np.expand_dims(img_3_channel, axis=0)
                
                prediction = emotion_model.predict(img_final)
                emotion_index = np.argmax(prediction)
                detected_emotion = EMOTION_MAP.get(emotion_index, "Unknown")
                # --- End of prediction logic ---
                
                st.success(f"Emotion Detected: **{detected_emotion.upper()}**")

                playlist = recommend_songs(detected_emotion, music_df)
                
                if not playlist.empty:
                    st.header("🎶 Here's Your Personalized Playlist:")
                    st.dataframe(
                        playlist[['artist_name', 'title']],
                        use_container_width=True,
                        hide_index=True
                    )

# --- Tab 2: Live Webcam Detection ---
with tab2:
    st.header("Live Emotion Detection")
    st.info("Click 'Start' to turn on your webcam. The app will detect your face and display your emotion in real-time. Allow camera permissions when prompted.")

    webrtc_streamer(
        key="webcam",
        video_processor_factory=EmotionVideoTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False}
    )