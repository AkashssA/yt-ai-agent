import streamlit as st
import time
from google import genai
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from urllib.parse import urlparse, parse_qs

# --- PAGE CONFIG ---
st.set_page_config(page_title="Video Intel AI", page_icon="🤖", layout="wide")

# Custom UI Styling for a Professional Dashboard
st.markdown("""
    <style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4150; color: white; }
    .main { background-color: #0e1117; }
    .stButton>button { background-color: #ff4b4b; color: white; font-weight: bold; width: 100%; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- API CLIENTS ---
# Securely loading keys from Streamlit Secrets
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("API Keys missing! Add them to 'Advanced Settings > Secrets' in Streamlit Cloud.")
    st.stop()

gemini_client = genai.Client(api_key=GEMINI_KEY)
groq_client = Groq(api_key=GROQ_KEY)

# --- FUNCTIONS ---

def get_video_id(url):
    parsed = urlparse(url)
    if parsed.netloc == 'youtu.be': return parsed.path[1:]
    if 'v=' in parsed.query: return parse_qs(parsed.query)['v'][0]
    return None

def fetch_transcript(v_id):
    try:
        data = YouTubeTranscriptApi.get_transcript(v_id)
        return " ".join([i['text'] for i in data])
    except Exception as e:
        return f"Error: {str(e)}"

def generate_ai_notes(transcript_text):
    """Try Gemini first, fallback to Groq if Quota (429) is hit."""
    safe_text = transcript_text[:9000] # Stay under free tier token limits
    prompt = f"Summarize this YouTube transcript into professional notes with key takeaways: {safe_text}"
    
    try:
        # 1. Attempt Gemini 2.0 Flash
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        return response.text, "Gemini 2.0"
    
    except Exception as e:
        # Check for Resource Exhausted (429)
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            st.warning("🔄 Gemini Quota Hit. Switching to Groq Fallback...")
            try:
                # 2. Fallback to Groq (Llama 3.3 70B)
                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                return chat_completion.choices[0].message.content, "Groq (Llama 3.3)"
            except Exception as groq_e:
                return f"Both APIs failed. Error: {str(groq_e)}", "Fail"
        else:
            return f"Error: {str(e)}", "Fail"

# --- DASHBOARD UI ---

st.title("📺 Video Intel: AI Analysis Agent")
st.write("Professional insights from YouTube transcripts with automatic failover.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Content"):
    v_id = get_video_id(url)
    if v_id:
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.video(url)
            
        with col2:
            with st.spinner("🕵️ Fetching Transcript..."):
                transcript = fetch_transcript(v_id)
                
                if "Error" not in transcript:
                    # ML Feature: Sentiment Analysis
                    blob = TextBlob(transcript)
                    polarity = blob.sentiment.polarity
                    tone = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
                    
                    # Display ML Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Emotional Tone", tone)
                    m2.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")
                    
                    # AI Note Generation
                    final_notes, provider = generate_ai_notes(transcript)
                    
                    st.success(f"Generated via {provider}")
                    st.subheader("📝 Professional Notes")
                    st.markdown(final_notes)
                else:
                    st.error("No transcript available for this video.")
    else:
        st.error("Invalid YouTube URL.")
