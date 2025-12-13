import sys
import os
import shutil
import numpy as np
import librosa
import scipy.io.wavfile as wavfile  # שימוש ב-scipy לכתיבה בטוחה
from scipy.signal import medfilt, butter, sosfilt
from pydub import AudioSegment
from pathlib import Path
import demucs.separate

class PhonkCleaner:
    def __init__(self, input_file_path):
        # המרה לאובייקט Path לטיפול בטוח בנתיבים
        self.input_path = Path(input_file_path).resolve()
        
        if not self.input_path.exists():
            print(f"[!] Critical Error: Input file not found: {self.input_path}")
            sys.exit(1)

        self.track_name = self.input_path.stem
        self.output_dir = Path("phonk_cleaned_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # נתיב היעד ש-Demucs יוצר
        self.demucs_out_dir = self.output_dir / "htdemucs_ft" / self.track_name

    def separate_stems(self):
        """ מריץ את Demucs להפרדת ערוצים """
        print(f"[*] Starting Demucs separation for: {self.input_path.name}")

        cmd = [
            "-n", "htdemucs_ft",
            str(self.input_path),
            "-o", str(self.output_dir),
            "--mp3"
        ]

        try:
            demucs.separate.main(cmd)
        except Exception as e:
            print(f"[!] Demucs separation failed: {e}")
            sys.exit(1)
        
        if not self.demucs_out_dir.exists():
            print(f"[!] Error: Expected output directory not found: {self.demucs_out_dir}")
            sys.exit(1)
            
        return self.demucs_out_dir

    def synthesize_clean_kick(self, sr=44100):
        duration = 0.4
        t = np.linspace(0, duration, int(sr * duration))
        freq_sweep = np.linspace(150, 40, len(t))
        waveform = np.sin(2 * np.pi * np.cumsum(freq_sweep) / sr)
        envelope = np.exp(-10 * t)
        return waveform * envelope

    def synthesize_clean_snare(self, sr=44100):
        duration = 0.25
        t = np.linspace(0, duration, int(sr * duration))
        noise = np.random.normal(0, 0.8, len(t))
        noise_env = np.exp(-12 * t)
        tone = np.sin(2 * np.pi * 180 * t / sr)
        tone_env = np.exp(-20 * t)
        return (noise * noise_env * 0.7) + (tone * tone_env * 0.3)

    def save_wav_safe(self, path, data, sr):
        """ פונקציית עזר לשמירה בטוחה עם Scipy """
        # נרמול למניעת דיסטורשן דיגיטלי
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val * 0.95
        
        # המרה ל-16 ביט (סטנדרט תעשייה, הכי תואם)
        data_int16 = (data * 32767).astype(np.int16)
        
        try:
            wavfile.write(str(path), sr, data_int16)
            print(f"    -> Saved: {path.name}")
        except Exception as e:
            print(f"[!] Failed to save {path.name}: {e}")

    def process_drums(self):
        print("[*] Cleaning Drums...")
        drums_file = self.demucs_out_dir / "drums.mp3"
        if not drums_file.exists(): drums_file = self.demucs_out_dir / "drums.wav"
            
        if not drums_file.exists():
            print("[!] Could not find separated drums file.")
            return None

        # טעינה עם librosa (תמיד מחזיר float32)
        y, sr = librosa.load(str(drums_file), sr=None)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='samples')
        
        y_clean = np.zeros_like(y)
        clean_kick = self.synthesize_clean_kick(sr)
        clean_snare = self.synthesize_clean_snare(sr)
        
        for onset in onsets:
            if onset + int(sr*0.05) >= len(y): continue
            segment = y[onset : onset + int(sr*0.05)]
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
            
            drum_sample = clean_kick if centroid < 700 else clean_snare
            
            end = min(onset + len(drum_sample), len(y_clean))
            length = end - onset
            y_clean[onset:end] += drum_sample[:length]

        output_path = self.demucs_out_dir / "drums_clean_synth.wav"
        self.save_wav_safe(output_path, y_clean, sr)
        return output_path

    def process_bass(self):
        print("[*] Cleaning Bass (Resynthesis)...")
        bass_file = self.demucs_out_dir / "bass.mp3"
        if not bass_file.exists(): bass_file = self.demucs_out_dir / "bass.wav"
            
        if not bass_file.exists(): return None

        y, sr = librosa.load(str(bass_file), sr=None)

        print("    Running pitch detection (High Precision Mode)...")
        # תיקון קריטי: הגדלת frame_length ל-4096 כדי לתפוס תדרי סאב-באס נמוכים
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C1'), 
            fmax=librosa.note_to_hz('C4'), 
            sr=sr, 
            frame_length=4096, # הוגדל מ-2048
            fill_na=0.0
        )
        
        f0 = np.nan_to_num(f0, nan=0.0)
        f0 = medfilt(f0, kernel_size=11)
        
        phase = np.cumsum(f0 / sr)
        sine_wave = np.sin(2 * np.pi * phase)
        saturated_wave = np.tanh(sine_wave * 3.0)
        
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        rms = librosa.util.fix_length(rms, size=len(saturated_wave))
        
        sos = butter(4, 200, 'lp', fs=sr, output='sos')
        clean_bass = sosfilt(sos, saturated_wave * rms)
        
        output_path = self.demucs_out_dir / "bass_clean_synth.wav"
        self.save_wav_safe(output_path, clean_bass, sr)
        return output_path

    def mix_final_track(self, clean_drums_path, clean_bass_path):
        print("[*] Mixing final track...")
        try:
            # מנסה לטעון את הווקאל וה-Other
            vocals_path = self.demucs_out_dir / "vocals.mp3"
            other_path = self.demucs_out_dir / "other.mp3"
            
            # אם לא קיים mp3, מנסה wav
            if not vocals_path.exists(): vocals_path = vocals_path.with_suffix('.wav')
            if not other_path.exists(): other_path = other_path.with_suffix('.wav')

            vocals = AudioSegment.from_file(str(vocals_path))
            other = AudioSegment.from_file(str(other_path))
            
            if clean_drums_path and clean_drums_path.exists():
                new_drums = AudioSegment.from_wav(str(clean_drums_path))
            else:
                new_drums = AudioSegment.silent(duration=len(vocals))

            if clean_bass_path and clean_bass_path.exists():
                new_bass = AudioSegment.from_wav(str(clean_bass_path))
            else:
                new_bass = AudioSegment.silent(duration=len(vocals))

            final_mix = vocals.overlay(other).overlay(new_drums - 2).overlay(new_bass - 2)
            
            out_file = self.output_dir / f"{self.track_name}_CLEANED_PHONK.mp3"
            final_mix.export(str(out_file), format="mp3")
            
            print(f"\n Track saved at: {out_file.absolute()}")
            
        except Exception as e:
            print(f"[!] Mixing failed: {e}")

if __name__ == "__main__":
    # וודא שהקובץ קיים ליד הסקריפט
    INPUT_FILE = "Montagem_Xonada.mp3"
    
    cleaner = PhonkCleaner(INPUT_FILE)
    cleaner.separate_stems()
    
    clean_drums = cleaner.process_drums()
    clean_bass = cleaner.process_bass()
    
    cleaner.mix_final_track(clean_drums, clean_bass)