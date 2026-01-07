import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai

load_dotenv()

# Initialize the Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_video_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([item['text'] for item in transcript_data])

def summarize_with_gemini(transcript):
    prompt = f"""
    You are a professional note-taker. Below is a transcript from a YouTube video. 
    Please provide:
    1. A concise 3-sentence summary.
    2. Key Takeaways in bullet points.
    3. A 'Deep Dive' section explaining the technical concepts mentioned.
    
    Transcript: {transcript}
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    print("🤖 Extraction in progress...")
    text = get_video_transcript(url)
    
    print("🧠 Gemini is thinking...")
    summary = summarize_with_gemini(text)
    
    print("\n--- FINAL NOTES ---\n")
    print(summary)