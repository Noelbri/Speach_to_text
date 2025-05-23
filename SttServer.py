import sounddevice as sd
import numpy as np 
from pynput import keyboard
from scipy.io.wavfile import write
import tempfile
import os 
from faster_whisper import WhisperModel

class FasterWhisperTranscriber:
    def __init__(self, model_size="small", sample_rate=44100):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, compute_type="float32")
        self.is_recording = False

    def on_press_space(self, key):
        if key == keyboard.Key.space:
            if not self.is_recording:
                self.is_recording = True
                print("Recording started.")
        
    def on_release_space(self, key):
        if key == keyboard.Key.space:
            if self.is_recording:
                self.is_recording = False
                print("Recording stopped.")
                return False
            
    def record_audio(self):
        recording = np.array([], dtype='float64').reshape(0,2)
        frames_per_buffer = int(self.sample_rate * 0.1)
        with keyboard.Listener(on_press=self.on_press_space, on_release=self.on_release_space) as listener:
            while True:
                if self.is_recording:
                    chunk = sd.rec(frames_per_buffer, samplerate=self.sample_rate, channels=2, dtype='float64')
                    sd.wait()
                    recording = np.vstack([recording, chunk])
                if not self.is_recording and len(recording) > 0:
                    break
            listener.join()
        return recording
    
    def save_temp_audio(self, recording):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        write(temp_file.name, self.sample_rate, recording)
        return temp_file.name
    

    def transcribe_audio(self, file_path):
        segments, info = self.model.transcribe(file_path, beam_size=5)
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        full_transcription = ""
        for segment in segments:
            print(segment.text)
            full_transcription += segment.text + ""
        os.remove(file_path)
        return full_transcription
    def run(self):
        print("Hold the spacebar to start Talking...")
        while True:
            recording = self.record_audio()
            file_path = self.save_temp_audio(recording)
            transcription = self.transcribe_audio(file_path)
            print(transcription)
            print("\nPress the spacebar to start recording again, or press Ctrl+C to exit.")


if __name__ == "__main__":
    transcriber = FasterWhisperTranscriber()
    transcriber.run()