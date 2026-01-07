import streamlit as st
import time
import os
import yt_dlp
from google import genai
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from urllib.parse import urlparse, parse_qs

# --- PAGE CONFIG ---
st.set_page_config(page_title="Video Intel AI Pro", page_icon="🤖", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4150; color: white; }
    .main { background-color: #0e1117; }
    .stButton>button { background-color: #ff4b4b; color: white; font-weight: bold; width: 100%; border-radius: 8px; height: 3em; }
    .stTextInput>div>div>input { background-color: #1e2130; color: white; border: 1px solid #3e4150; }
    </style>
    """, unsafe_allow_html=True)

# --- API CLIENTS ---
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("API Keys missing! Add GEMINI_API_KEY and GROQ_API_KEY to Secrets in Streamlit Cloud.")
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
    """Attempt to get official YouTube captions."""
    try:
        data = YouTubeTranscriptApi.get_transcript(v_id)
        return " ".join([i['text'] for i in data])
    except Exception:
        return None

def transcribe_audio_with_groq(url):
    """Fallback: Download audio and use Groq Whisper for videos with no transcripts."""
    st.warning("⚠️ No transcript found. Groq is 'listening' to the video audio...")
    
    # yt-dlp options to extract high-quality audio
    audio_filename = "temp_audio.m4a"
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Send audio file to Groq Whisper
        with open(audio_filename, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_filename, file.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        
        # Cleanup file
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
            
        return transcription
    except Exception as e:
        if os.path.exists(audio_filename): os.remove(audio_filename)
        return f"Error during audio transcription: {str(e)}"

def generate_ai_notes(transcript_text):
    """Generate notes using Gemini with a Groq Llama fallback."""
    safe_text = transcript_text[:10000] # Limit context for free tier
    prompt = f"Summarize this YouTube content into professional notes with key takeaways and action items: {safe_text}"
    
    try:
        # Attempt Gemini 2.0 Flash
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        return response.text, "Gemini 2.0 Flash"
    
    except Exception as e:
        # Fallback to Groq Llama 3.3 if Gemini fails (429/Quota)
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            st.info("🔄 Gemini busy. Using Groq Llama 3.3...")
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                return chat_completion.choices[0].message.content, "Groq Llama 3.3"
            except Exception as groq_e:
                return f"Both AI models failed. Error: {str(groq_e)}", "Fail"
        else:
            return f"AI Error: {str(e)}", "Fail"

# --- DASHBOARD UI ---

st.title("📺 Video Intel AI: Professional Agent")
st.write("Analyze any video—even those without transcripts—using Multi-Model AI.")

url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Content"):
    v_id = get_video_id(url)
    if v_id:
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.video(url)
            
        with col2:
            with st.spinner("🕵️ Agent is processing..."):
                # 1. Try to get transcript
                transcript = fetch_transcript(v_id)
                
                # 2. If no transcript, listen to audio manually
                if not transcript:
                    transcript = transcribe_audio_with_groq(url)
                
                if transcript and "Error" not in transcript:
                    # ML Feature: Sentiment Analysis
                    blob = TextBlob(transcript)
                    polarity = blob.sentiment.polarity
                    tone = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
                    
                    # Dashboard Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Emotional Tone", tone)
                    m2.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")
                    
                    # Generate AI Summary
                    final_notes, provider = generate_ai_notes(transcript)
                    
                    st.success(f"Analysis Complete via {provider}")
                    st.subheader("📝 Professional Notes")
                    st.markdown(final_notes)
                else:
                    st.error(f"Could not process video. {transcript}")
    else:
        st.error("Please enter a valid YouTube URL.")
