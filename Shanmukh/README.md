# 🎧 AI MoodMate — Emotion Detection + Music Recommendation

This project detects a person's **emotion from a face image** using a deep learning model (TensorFlow/Keras) and recommends songs that match the detected mood.

## 🧠 Features
* Upload an image or capture a new one via **webcam**.
* Detects 7 primary emotions (Happy, Sad, Angry, Fear, Neutral, etc.).
* Recommends mood-matched songs from a processed Spotify dataset.
* Built using **TensorFlow**, **Streamlit**, **Pandas**, and **OpenCV**.

## 🚀 Live App
👉 [Click here to open the project](https://emotion-detection-and-music-recommended-system-vzedurgxgdecpdy.streamlit.app/)

## 📂 Project Files
* **`app.py`** → The main Streamlit application code.
* **`Emotion_Detection_Final_Project.ipynb`** → The Google Colab notebook used for model experimentation and training.
* **`emotion_model_mobilenet_v2_final.keras`** → The final, best-performing saved model (MobileNetV2).
* **`emotion_model_custom_cnn_sgd.keras`** → The saved model from the custom CNN experiment.
* **`processed_music.csv`** → The final music database used for recommendations.
* **`requirements.txt`** → A list of required Python packages for deployment.

## 🧰 Tech Stack
* **Core Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Web Framework:** Streamlit
* **Data & ML:** Scikit-learn (for classification reports), Pandas, NumPy
* **Image Processing:** OpenCV, Pillow (PIL)
* **Plotting:** Matplotlib, Seaborn
* **Tools & Environment:** Google Colab, GitHub

## 🧑‍💻 Author
**Shanmukh Akkala** Built as part of the **Emotion Detection and Music Recommendation System** project.

## 🪪 License
This project is licensed under the **MIT License** – see the LICENSE file for details.
