# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import emoji

# Load the pre-trained emotion classification model with caching
@st.cache_resource
def load_model():
    MODEL_PATH = "models/emotion_classifier_pipeline_lr_15_aug_2021.pkl"
    return joblib.load(open(MODEL_PATH, "rb"))

pipeline_lr = load_model()

# Function to predict emotions
def predict_emotions(docx):
    return pipeline_lr.predict([docx])[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    return pipeline_lr.predict_proba([docx])

# Function to get emoji for emotion
def get_emoji(emotion):
    emojis = {
        "happy": "😊", "sad": "😢", "angry": "😡", "surprised": "😲", "neutral": "😐"
    }
    return emojis.get(emotion, "❓")

# Apply custom CSS for enhanced UI/UX
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f4f4f4;
        }
        .main-title {
            font-size: 40px;
            font-weight: 600;
            color: #FF4500;
            text-align: center;
            padding: 20px;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        }
        .sub-header {
            font-size: 24px;
            font-weight: bold;
            color: #4682B4;
            text-align: center;
            margin-bottom: 20px;
        }
        .stTextArea {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton button {
            background: linear-gradient(135deg, #FF6347, #FF4500);
            color: white;
            font-size: 18px;
            padding: 12px 25px;
            border-radius: 10px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #FF4500, #FF6347);
            transform: scale(1.05);
        }
        .result-card {
            background: red;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App Main Function
def main():
    st.markdown("<h1 class='main-title'>🔥 Emotion Classifier App 🔥</h1>", unsafe_allow_html=True)
    menu = ['Home', 'About']
    choice = st.sidebar.selectbox("📌 Menu", menu)

    if choice == "Home":
        st.markdown("<h2 class='sub-header'>📝 Analyze Emotion in Text</h2>", unsafe_allow_html=True)
        
        # User input form
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area(label="", placeholder="Type your text here...", height=150)
            example_text = "I am so happy today!"
            use_example = st.form_submit_button("🔄 Use Example")
            if use_example:
                raw_text = example_text
            submit_text = st.form_submit_button(label="🔍 Analyze Emotion")
        
        if submit_text:
            with st.spinner("Analyzing emotion..."):
                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<div class='result-card'><h3>📜 Original Text:</h3><p>{raw_text}</p></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-card'><h3>🔮 Prediction:</h3><p>Emotion: <b>{prediction}</b> {get_emoji(prediction)}</p><p>Confidence: <b>{np.max(probability) * 100:.2f}%</b></p></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='result-card'><h3>📊 Prediction Probability Distribution:</h3></div>", unsafe_allow_html=True)
                proba_df = pd.DataFrame(probability, columns=pipeline_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotion", "Probability"]
                proba_df_clean = proba_df_clean.sort_values(by="Probability", ascending=False)
                
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='Emotion',
                    y='Probability',
                    color='Emotion'
                ).properties(width=400, height=300)
                st.altair_chart(fig, use_container_width=True)
                
        # File upload option
        st.markdown("### 📂 Upload a text file for analysis")
        uploaded_file = st.file_uploader("Choose a file", type=['txt'])

        if uploaded_file is not None:
            # Read and decode file content
            file_text = uploaded_file.getvalue().decode("utf-8")

            # Display file content
            st.text_area("📄 File Content", file_text, height=200)

            # Make predictions on the file content
            file_prediction = predict_emotions(file_text)
            file_probability = get_prediction_proba(file_text)

            # Display results
            st.success(f"Predicted Emotion: **{file_prediction}** {get_emoji(file_prediction)}")
            st.write(f"Confidence Score: **{np.max(file_probability) * 100:.2f}%**")

            # Show probability distribution
            proba_df = pd.DataFrame(file_probability, columns=pipeline_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotion", "Probability"]
            proba_df_clean = proba_df_clean.sort_values(by="Probability", ascending=False)

            # Show probability distribution as a chart
            st.markdown("### 📊 Emotion Probability Distribution")
            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x="Emotion",
                y="Probability",
                color="Emotion"
            ).properties(width=500, height=300)
            st.altair_chart(fig, use_container_width=True)

    else:
        st.markdown("<h2 class='sub-header'>ℹ️ About</h2>", unsafe_allow_html=True)
        st.info("🚀 This emotion classification project is created by Kashish Girdhar (2210991767). It uses an advanced ML model to analyze text emotions.")
        
if __name__ == '__main__':
    main()
