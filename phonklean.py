import sys
import shutil
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import medfilt, butter, sosfilt
from pydub import AudioSegment
from pathlib import Path
import demucs.separate

class PhonkCleaner:
    def __init__(self, input_file_path):
        # המרה לאובייקט Path - פותר את כל בעיות ה-Tuple/String
        self.input_path = Path(input_file_path).resolve()
        
        if not self.input_path.exists():
            print(f"[!] Critical Error: Input file not found: {self.input_path}")
            sys.exit(1)

        self.track_name = self.input_path.stem  # השם ללא סיומת (למשל "song")
        self.output_dir = Path("phonk_cleaned_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # נתיב היעד ש-Demucs ייצור
        # Demucs v4 יוצר תיקייה בשם המודל (htdemucs_ft) ובתוכה תיקייה עם שם השיר
        self.demucs_out_dir = self.output_dir / "htdemucs_ft" / self.track_name

    def separate_stems(self):
        """ מריץ את Demucs להפרדת ערוצים """
        print(f"[*] Starting Demucs separation for: {self.input_path.name}")
        print("    (This might take a while depending on your GPU/CPU...)")

        # בניית הפקודה כרשימת מחרוזות
        cmd = [
            "-n", "htdemucs_ft",      # המודל האיכותי ביותר
            str(self.input_path),     # נתיב הקלט
            "-o", str(self.output_dir), # תיקיית הפלט הראשית
            "--mp3"                   # שמירה כ-mp3 לחסכון במקום (אפשר להוריד אם רוצים wav)
        ]

        try:
            # קריאה ישירה לפונקציה של Demucs
            demucs.separate.main(cmd)
        except Exception as e:
            print(f"[!] Demucs failed to run: {e}")
            sys.exit(1)
        
        # וידוא שהקבצים נוצרו
        if not self.demucs_out_dir.exists():
            print(f"[!] Error: Expected output directory not found: {self.demucs_out_dir}")
            sys.exit(1)
            
        print(f"[*] Separation complete. Stems located at: {self.demucs_out_dir}")
        return self.demucs_out_dir

    def synthesize_clean_kick(self, sr=44100):
        """ יצירת קיק נקי (Sine Sweep) """
        duration = 0.4
        t = np.linspace(0, duration, int(sr * duration))
        # תדר יורד מ-150 הרץ ל-40 הרץ
        freq_sweep = np.linspace(150, 40, len(t))
        waveform = np.sin(2 * np.pi * np.cumsum(freq_sweep) / sr)
        envelope = np.exp(-10 * t) # מעטפת דעיכה
        return waveform * envelope

    def synthesize_clean_snare(self, sr=44100):
        """ יצירת סנר נקי (Noise + Tone) """
        duration = 0.25
        t = np.linspace(0, duration, int(sr * duration))
        
        # רכיב רעש (הסנר עצמו)
        noise = np.random.normal(0, 0.8, len(t))
        noise_env = np.exp(-12 * t)
        
        # רכיב טונאלי (הגוף של התוף)
        tone = np.sin(2 * np.pi * 180 * t / sr)
        tone_env = np.exp(-20 * t)
        
        return (noise * noise_env * 0.7) + (tone * tone_env * 0.3)

    def process_drums(self):
        """ החלפת תופים מלוכלכים בנקיים """
        print("[*] Cleaning Drums...")
        
        # ניסיון לטעון mp3 או wav
        drums_file = self.demucs_out_dir / "drums.mp3"
        if not drums_file.exists():
            drums_file = self.demucs_out_dir / "drums.wav"
            
        if not drums_file.exists():
            print("[!] Could not find separated drums file.")
            return None

        y, sr = librosa.load(str(drums_file), sr=None)
        
        # זיהוי מכות (Onsets)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='samples')
        
        y_clean = np.zeros_like(y)
        clean_kick = self.synthesize_clean_kick(sr)
        clean_snare = self.synthesize_clean_snare(sr)
        
        for onset in onsets:
            # בדיקה קצרה כדי לא לחרוג מגבולות המערך
            if onset + int(sr*0.05) >= len(y): continue
            
            # ניתוח ספקטרלי כדי להבדיל בין קיק לסנר
            segment = y[onset : onset + int(sr*0.05)]
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
            
            # סף פשוט: נמוך = קיק, גבוה = סנר
            drum_sample = clean_kick if centroid < 700 else clean_snare
            
            # הוספת הדגימה הנקייה למיקס החדש
            end = min(onset + len(drum_sample), len(y_clean))
            length = end - onset
            y_clean[onset:end] += drum_sample[:length]

        # שמירת הקובץ החדש
        output_path = self.demucs_out_dir / "drums_clean_synth.wav"
        sf.write(str(output_path), y_clean, sr)
        return output_path

    def process_bass(self):
        """ שחזור באס נקי באמצעות זיהוי תדרים (Resynthesis) """
        print("[*] Cleaning Bass (Resynthesis)...")
        
        bass_file = self.demucs_out_dir / "bass.mp3"
        if not bass_file.exists():
            bass_file = self.demucs_out_dir / "bass.wav"
            
        if not bass_file.exists():
            print("[!] Could not find separated bass file.")
            return None

        y, sr = librosa.load(str(bass_file), sr=None)

        # PYIN - אלגוריתם זיהוי פיץ' חזק
        print("    Running pitch detection (this may take a moment)...")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C4'), sr=sr, frame_length=2048
        )
        
        # החלפת ערכים חסרים (NaN) ב-0
        f0 = np.nan_to_num(f0, nan=0.0)
        # החלקה של הפיץ' כדי למנוע קפיצות
        f0 = medfilt(f0, kernel_size=11)
        
        # סינתזה: יצירת גל סינוס לפי התדר המזוהה בכל רגע
        # יצירת וקטור פאזה
        phase = np.cumsum(f0 / sr)
        sine_wave = np.sin(2 * np.pi * phase)
        
        # רוויה (Saturation) עדינה כדי לתת לבאס "בשר" כמו 808
        saturated_wave = np.tanh(sine_wave * 3.0)
        
        # התאמת העוצמה (Volume Matching) למקור
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        rms = librosa.util.fix_length(rms, size=len(saturated_wave))
        
        # Low Pass Filter לניקוי רעשים דיגיטליים גבוהים
        sos = butter(4, 200, 'lp', fs=sr, output='sos')
        clean_bass = sosfilt(sos, saturated_wave * rms)
        
        output_path = self.demucs_out_dir / "bass_clean_synth.wav"
        sf.write(str(output_path), clean_bass, sr)
        return output_path

    def mix_final_track(self, clean_drums_path, clean_bass_path):
        """ איחוד כל הערוצים לקובץ סופי """
        print("[*] Mixing final track...")
        
        # טעינת הערוצים המקוריים (שירה וכלים אחרים)
        try:
            # מנסה לטעון mp3 קודם (כי הרצנו עם --mp3)
            vocals = AudioSegment.from_file(self.demucs_out_dir / "vocals.mp3")
            other = AudioSegment.from_file(self.demucs_out_dir / "other.mp3")
        except:
            # fallback ל-wav
            vocals = AudioSegment.from_file(self.demucs_out_dir / "vocals.wav")
            other = AudioSegment.from_file(self.demucs_out_dir / "other.wav")

        # טעינת הערוצים החדשים שיצרנו (WAV)
        if clean_drums_path and clean_drums_path.exists():
            new_drums = AudioSegment.from_wav(str(clean_drums_path))
        else:
            new_drums = AudioSegment.silent(duration=len(vocals))

        if clean_bass_path and clean_bass_path.exists():
            new_bass = AudioSegment.from_wav(str(clean_bass_path))
        else:
            new_bass = AudioSegment.silent(duration=len(vocals))

        # מיקס: שירה + כלים אחרים + תופים חדשים (-2dB) + באס חדש (-2dB)
        final_mix = vocals.overlay(other).overlay(new_drums - 2).overlay(new_bass - 2)
        
        out_file = self.output_dir / f"{self.track_name}_CLEANED_PHONK.mp3"
        final_mix.export(str(out_file), format="mp3")
        
        print("\n" + "="*40)
        print(f"DONE! File saved at:\n{out_file.absolute()}")
        print("="*40)

# --- אזור ההרצה ---
if __name__ == "__main__":
    # שנה את השורה הזו לנתיב של השיר שלך
    # דוגמה: "C:/Music/drift_phonk.mp3" או פשוט "song.mp3"
    INPUT_FILE = "Montagem_Xonada.mp3"
    
    cleaner = PhonkCleaner(INPUT_FILE)
    
    # שלב 1: הפרדה
    cleaner.separate_stems()
    
    # שלב 2+3: עיבוד
    clean_drums = cleaner.process_drums()
    clean_bass = cleaner.process_bass()
    
    # שלב 4: מיקס
    cleaner.mix_final_track(clean_drums, clean_bass)