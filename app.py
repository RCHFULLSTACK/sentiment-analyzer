import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
import time
import re
from collections import Counter

# Konfigurera sidan
st.set_page_config(page_title="Avancerad Sentiment Analyzer", page_icon="🧠", layout="wide")

# CSS styling
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton>button {
    background-color: #4361ee;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    border: none;
}
.stButton>button:hover {
    background-color: #3a56d4;
}
.title-container {
    background-color: #4361ee;
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
}
.result-positive {
    background-color: rgba(76, 175, 80, 0.1);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
    margin: 1rem 0;
}
.result-negative {
    background-color: rgba(244, 67, 54, 0.1);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #F44336;
    margin: 1rem 0;
}
.highlight-positive {
    background-color: rgba(76, 175, 80, 0.2);
    padding: 2px;
    border-radius: 3px;
}
.highlight-negative {
    background-color: rgba(244, 67, 54, 0.2);
    padding: 2px;
    border-radius: 3px;
}
.metric-card {
    background-color: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    text-align: center;
}
.history-item {
    background-color: white;
    padding: 0.8rem;
    border-radius: 5px;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Funktioner för analys
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def get_sentiment_words(text, sentiment):
    """Identifiera ord som bidrar till sentiment"""
    # Enkel heuristik för demo - i verkligheten skulle detta använda mer avancerad NLP
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "love", "best", "beautiful", "enjoy"]
    negative_words = ["bad", "terrible", "awful", "horrible", "sad", "hate", "worst", "ugly", "disappointing", "poor"]
    
    words = re.findall(r'\b\w+\b', text.lower())
    if sentiment == "POSITIVE":
        return [word for word in words if word in positive_words]
    else:
        return [word for word in words if word in negative_words]

def highlight_sentiment_words(text, sentiment_words):
    """Markera ord som bidrar till sentiment"""
    text_lower = text.lower()
    for word in sentiment_words:
        # Hitta alla förekomster av ordet med hänsyn till ord-gränser
        for match in re.finditer(r'\b' + re.escape(word) + r'\b', text_lower):
            start, end = match.span()
            actual_word = text[start:end]  # Bevara original skiftläge
            highlight_class = "highlight-positive" if word in positive_words else "highlight-negative"
            text = text[:start] + f'<span class="{highlight_class}">{actual_word}</span>' + text[end:]
            # Justera text_lower för att matcha ny längd
            text_lower = text_lower[:start] + actual_word + text_lower[end:]
    return text

def analyze_text_sentiment(text):
    """Utför fullständig sentiment-analys"""
    # Grundläggande sentiment
    result = sentiment_model(text)
    sentiment = result[0]['label']
    score = result[0]['score']
    
    # Emotioner
    emotions = emotion_model(text)
    top_emotion = emotions[0]['label']
    
    # Identifiera nyckelord
    sentiment_words = get_sentiment_words(text, sentiment)
    
    # Räkna ord
    word_count = len(re.findall(r'\b\w+\b', text))
    
    return {
        'sentiment': sentiment,
        'score': score,
        'emotion': top_emotion,
        'sentiment_words': sentiment_words,
        'word_count': word_count,
        'text': text
    }

# Initiera session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Ladda modeller
sentiment_model = load_sentiment_model()
emotion_model = load_emotion_model()

# Definiera positiva/negativa ord för highlighting
positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "love", "best", "beautiful", "enjoy"]
negative_words = ["bad", "terrible", "awful", "horrible", "sad", "hate", "worst", "ugly", "disappointing", "poor"]

# App rubrik
st.markdown('<div class="title-container"><h1>🧠 Avancerad Sentiment Analyzer</h1><p>Analysera känsloläget och emotioner i din text med djup AI</p></div>', unsafe_allow_html=True)

# Skapa två kolumner - en för input och en för historik
col1, col2 = st.columns([2, 1])

with col1:
    # Inmatning och analys
    st.subheader("Textanalys")
    user_input = st.text_area("Skriv eller klistra in text att analysera:", height=150)
    
    # Språkval
    language = st.selectbox("Välj språk (Notera: bäst resultat för engelska)", ["Engelska", "Svenska"])
    
    # Avancerade inställningar
    with st.expander("Avancerade inställningar"):
        show_word_cloud = st.checkbox("Visa ordmoln", value=True)
        show_emotion = st.checkbox("Visa emotionell analys", value=True)
    
    # Analys-knapp
    if st.button("Analysera text"):
        if user_input:
            with st.spinner('Analyserar texten...'):
                # Simulera lite bearbetningstid för bättre UX
                time.sleep(0.8)
                
                # Genomför analys
                analysis = analyze_text_sentiment(user_input)
                
                # Spara till historik
                st.session_state.analysis_history.append(analysis)
                
                # Visa resultat
                sentiment = analysis['sentiment']
                score = analysis['score']
                
                # Bestäm färg och emoji
                if sentiment == "POSITIVE":
                    emoji = "😊"
                    color = "green"
                    sentiment_text = "POSITIV"
                    result_class = "result-positive"
                else:
                    emoji = "😔"
                    color = "red"
                    sentiment_text = "NEGATIV"
                    result_class = "result-negative"
                
                # Visa huvud-resultat
                st.markdown(f'<div class="{result_class}">', unsafe_allow_html=True)
                st.markdown(f"### Resultat: {emoji}")
                st.markdown(f"<h3 style='color: {color};'>Texten är <b>{sentiment_text}</b> med {score:.1%} säkerhet</h3>", unsafe_allow_html=True)
                
                # Visualisera resultat
                st.progress(score if sentiment == "POSITIVE" else 1-score)
                
                # Visa emotion om aktiverad
                if show_emotion:
                    st.markdown(f"**Dominerande emotion:** {analysis['emotion'].capitalize()}")
                
                # Visa wordcount
                st.markdown(f"**Ordantal:** {analysis['word_count']}")
                
                # Visa nyckelord
                if analysis['sentiment_words']:
                    st.markdown("**Nyckelord som påverkar sentiment:**")
                    st.write(", ".join(analysis['sentiment_words']))
                
                # Visa text med markerade sentiment-ord
                st.markdown("**Din text med sentiment-analys:**")
                highlighted_text = highlight_sentiment_words(user_input, analysis['sentiment_words'])
                st.markdown(highlighted_text, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Visa ordmoln om aktiverat
                if show_word_cloud and analysis['word_count'] > 5:
                    st.subheader("Ordfrekvensanalys")
                    
                    # Räkna ord
                    words = re.findall(r'\b\w+\b', user_input.lower())
                    word_freq = Counter(words)
                    
                    # Skapa en enkel graf
                    top_words = dict(word_freq.most_common(10))
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.barh(list(top_words.keys()), list(top_words.values()), color='#4361ee')
                    ax.set_xlabel('Frekvens')
                    ax.set_title('Vanligaste orden i texten')
                    st.pyplot(fig)
        else:
            st.error("Vänligen ange text för analys.")

with col2:
    # Historik-sektion
    st.subheader("Analyshistorik")
    
    if st.session_state.analysis_history:
        # Visa statistik
        total_analyses = len(st.session_state.analysis_history)
        positive_count = sum(1 for item in st.session_state.analysis_history if item['sentiment'] == "POSITIVE")
        negative_count = total_analyses - positive_count
        
        # Visa statistik i tre kolumner
        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**{total_analyses}**")
            st.markdown("Analyser")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with stat2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**{positive_count}**")
            st.markdown("Positiva")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with stat3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**{negative_count}**")
            st.markdown("Negativa")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visa historikens element
        st.markdown("### Tidigare analyser")
        for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
            if i >= 5:  # Begränsa till senaste 5 för enkelhetens skull
                break
                
            sentiment_emoji = "😊" if analysis['sentiment'] == "POSITIVE" else "😔"
            st.markdown(f'<div class="history-item">', unsafe_allow_html=True)
            st.markdown(f"**{sentiment_emoji} {analysis['sentiment']} ({analysis['score']:.1%})**")
            
            # Visa en kort version av texten
            short_text = analysis['text'][:50] + "..." if len(analysis['text']) > 50 else analysis['text']
            st.markdown(f"{short_text}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Lägg till en knapp för att rensa historik
        if st.button("Rensa historik"):
            st.session_state.analysis_history = []
            st.experimental_rerun()
    else:
        st.info("Inga tidigare analyser. Börja genom att analysera text i panelen till vänster.")

# Footer
st.markdown("---")
st.markdown("Powered by Hugging Face Transformers | Utvecklad av Sentiment Analyzer Team")