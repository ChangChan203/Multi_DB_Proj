import librosa
import soundfile as sf
from scipy import signal
from pydub import AudioSegment

def cut_audio(input_path, output_path, min_keep_sec=10, start_sec=0, end_sec=10):
    audio = AudioSegment.from_file(input_path)
    duration = len(audio) / 1000.0

    if duration > min_keep_sec:
        audio = audio[start_sec * 1000: end_sec * 1000]

    audio.export(output_path, format="wav")
    return output_path

def clean_audio(input_path, output_path, target_sr=16000):
    audio, sr = librosa.load(input_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    b, a = signal.butter(4, 0.5, 'low')
    audio_filtered = signal.filtfilt(b, a, audio)
    audio_normalized = librosa.util.normalize(audio_filtered)

    sf.write(output_path, audio_normalized, target_sr)
    return output_path

def prepare_uploaded_file(uploaded_path):
    temp_cut = "temp_cut.wav"
    temp_clean = "temp_clean.wav"
    cut_audio(uploaded_path, temp_cut)
    clean_audio(temp_cut, temp_clean)
    return temp_clean
