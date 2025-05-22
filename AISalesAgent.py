from SttServer import FasterWhisperTranscriber
import logging
from groq import Groq
import os
import subprocess
import tempfile
import numpy as np
from termcolor import colored
from dotenv import load_dotenv
import requests

load_dotenv()

#Define API keys and voice ID
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
groq_api_key = os.getenv("GROQ_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
gclient = Groq(api_key=groq_api_key)

def text_to_speech(elevenlabs_api_key, voice_id, text):
    """
    Convert text to speech using Elevenlabs API and return the audio data.
    
    Args:
    - elevenlabs_api_key (str): The API key for Elevenlabs.
    - voice_id (str): The voice ID to use for text-to-speech.
    - text (str): The text to convert to speech.
    
    Returns:
    - bytes: The audio data of the spoken text.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": f"{elevenlabs_api_key}"
    }
    data = {
        "model_id": "eleven_monolingual_v1",
        "text": text,
        "voice_settings": {
            "similarity_boost": 0.8,
            "stability": 0.5,
            "use_speaker_boost": True
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to get speech from ElevenLabs API: {response.text}")

def play_audio(audio_data):
    """
    Play the given audio data using the mpv player.
    
    Args:
    - audio_data (bytes): The audio data to play.
    """
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmpfile:
        tmpfile.write(audio_data)
        tmpfile.flush()
        subprocess.run(["mpv", "--no-video", tmpfile.name], check=True)

def main():
    whisper = FasterWhisperTranscriber()
    message_history = [
        {
            "role": "system",
            "content": "You are Noel. You work as a sales representative."
        }
    ]

    while True:
        try:
            print("\nPress and hold the SPACE to start Talking with the AI Sales Agent...")
            recording = whisper.record_audio()
            file_path = whisper.save_temp_audio(recording)
            full_transcript = whisper.transcribe_audio(file_path)
            print("Transcription:", full_transcript)
            message_history.append({"role": "user", "content": full_transcript})

            chat_completion = gclient.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=message_history,
                temperature=0.5,
                max_tokens=200,
                stream=False,
                top_p=1
            )
            assistant_response = chat_completion.choices[0].message.content
            print("Response from Groq client:", assistant_response)
            message_history.append({"role": "assistant", "content": assistant_response})

            #Simulate sending response text to EvelenLabs for TTS and getting audio data
            audio_data = text_to_speech(ELEVENLABS_API_KEY, VOICE_ID, assistant_response)
            #Simulate play audio
            play_audio(audio_data)

        except KeyboardInterrupt:
            print("\nExiting due to KeyboardInterrupt...")
            break

if __name__ == "__main__":
    main()