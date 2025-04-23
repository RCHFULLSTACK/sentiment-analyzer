import streamlit as st
from transformers import pipeline

# Konfigurera sidan
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

# Stil och utseende
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# App titel
st.title("🔍 Textanalys: Sentiment Detector")
st.markdown("Analysera känsloläget i din text med hjälp av AI")

# Ladda sentiment-modellen
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_model()

# Text input
user_input = st.text_area("Skriv eller klistra in text att analysera:", height=150)

# Analys-knapp
if st.button("Analysera text"):
    if user_input:
        with st.spinner('Analyserar texten...'):
            result = sentiment_analyzer(user_input)
            
            # Visa resultat
            sentiment = result[0]['label']
            score = result[0]['score']
            
            if sentiment == "POSITIVE":
                emoji = "😊"
                color = "green"
                sentiment_text = "POSITIV"
            else:
                emoji = "😔"
                color = "red"
                sentiment_text = "NEGATIV"
            
            st.markdown(f"## Resultat: {emoji}")
            st.markdown(f"<h3 style='color: {color};'>Texten är <b>{sentiment_text}</b> med {score:.1%} säkerhet</h3>", unsafe_allow_html=True)
            
            # Visualisering
            st.progress(score if sentiment == "POSITIVE" else 1-score)
    else:
        st.error("Vänligen ange text för analys.")

st.markdown("---")
st.markdown("Powered by Hugging Face Transformers")