import streamlit as st
import time
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from urllib.parse import urlparse, parse_qs

# --- PAGE CONFIG ---
st.set_page_config(page_title="Video Intel AI", page_icon="🤖", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4150; }
    </style>
    """, unsafe_allow_html=True)

# API Setup
api_key = st.secrets.get("GEMINI_API_KEY", "YOUR_KEY_HERE")
client = genai.Client(api_key=api_key)

def get_video_id(url):
    parsed = urlparse(url)
    if parsed.netloc == 'youtu.be': return parsed.path[1:]
    if 'v=' in parsed.query: return parse_qs(parsed.query)['v'][0]
    return None

def fetch_transcript(v_id):
    try:
        data = YouTubeTranscriptApi().fetch(v_id).to_raw_data()
        return " ".join([i['text'] for i in data])
    except Exception: return None

# --- UI LAYOUT ---
st.title("📺 Video Intel: Professional AI Agent")
url = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Start AI Analysis", use_container_width=True):
    v_id = get_video_id(url)
    if v_id:
        # Create columns for the Dashboard Look
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.video(url)
            st.info("💡 Pro Tip: Sentiment analysis helps determine if the video content is generally critical, objective, or promotional.")
            
        with col2:
            with st.spinner("🕵️ AI Agent is scanning transcript..."):
                transcript = fetch_transcript(v_id)
                
                if transcript:
                    # ML Feature: Sentiment Analysis
                    blob = TextBlob(transcript)
                    polarity = blob.sentiment.polarity
                    sentiment_label = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
                    
                    # Display ML Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Emotional Tone", sentiment_label)
                    m2.metric("Subjectivity Score", f"{blob.sentiment.subjectivity:.2f}")

                    # AI Summary Generation
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.0-flash", 
                            contents=f"Provide a professional summary with bulleted action items for: {transcript[:8000]}"
                        )
                        st.subheader("📝 AI Summary & Notes")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"AI Quota Reached: {e}")
                else:
                    st.error("Transcript not found. Ensure the video has closed captions enabled.")
    else:
        st.error("Invalid YouTube URL.")