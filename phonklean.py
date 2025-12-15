import sys
import os
import shutil
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt, correlate
from pydub import AudioSegment
from pathlib import Path
import demucs.separate
import torch

# נסיון לייבא את VoiceFixer
try:
    from voicefixer import VoiceFixer
    HAS_VOICEFIXER = True
    print("[*] VoiceFixer detected! High-Fidelity Parallel Processing Enabled.")
except ImportError:
    HAS_VOICEFIXER = False
    print("[*] VoiceFixer not found. Using standard EQ fallback.")
    print("    (To enable AI restoration: pip install voicefixer)")

class PhonkCleaner:
    def __init__(self, input_file_path):
        self.input_path = Path(input_file_path).resolve()
        
        if not self.input_path.exists():
            print(f"[!] Critical Error: Input file not found: {self.input_path}")
            sys.exit(1)

        self.track_name = self.input_path.stem
        self.output_dir = Path("phonk_cleaned_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.demucs_out_dir = self.output_dir / "htdemucs_ft" / self.track_name

    def separate_stems(self):
        print(f"[*] Starting Demucs separation for: {self.input_path.name}")
        
        # אם כבר קיימת הפרדה, דלג כדי לחסוך זמן
        if self.demucs_out_dir.exists() and (self.demucs_out_dir / "vocals.mp3").exists():
            print("    -> Stems already exist, skipping separation.")
            return self.demucs_out_dir

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

    def save_wav_safe(self, path, data, sr):
        # נרמול למניעת דיסטורשן דיגיטלי
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val * 0.95

        # טיפול במימדים (מונו/סטריאו)
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                if data.shape == 1 and data.shape[1] > 1: # (1, N)
                    data = data.flatten()
                elif data.shape[1] == 1: # (N, 1)
                    data = data[:, 0]
                elif data.shape == 2 and data.shape[1] > 2: # (2, N) -> Transpose
                    data = data.T
            # הפיכה לסטריאו אם זה מונו
            if data.ndim == 1:
                data = np.column_stack((data, data))
        else:
            data = np.array(data)
            if data.ndim == 1:
                data = np.column_stack((data, data))
        
        data_int16 = (data * 32767).astype(np.int16)
        
        try:
            wavfile.write(str(path), sr, data_int16)
            print(f"    -> Saved Stereo WAV: {path.name}")
        except Exception as e:
            print(f"[!] Failed to save {path.name}: {e}")

    # --- יישור פאזה (הסוד לעיבוד מקבילי) ---
    def align_signals(self, ref_sig, target_sig):
        """
        מסנכרן את הפאזה בין האות המקורי לאות המעובד (AI) כדי למנוע ביטול תדרים.
        """
        # לוקחים חתיכה קצרה לחישוב כדי לא להעמיס על המעבד
        # מספיק 30 שניות כדי להבין את הדיליי
        n = min(len(ref_sig), len(target_sig), 44100 * 30)
        
        # חישוב קורלציה
        correlation = correlate(ref_sig[:n], target_sig[:n], mode='full', method='fft')
        lag = correlation.argmax() - (n - 1)
        
        # אם מצאנו דיליי משמעותי, נתקן אותו
        if abs(lag) > 0:
            print(f"    -> Aligning phase: shifting by {lag} samples")
            if lag > 0:
                # Target מקדים, צריך להזיז ימינה
                aligned = np.pad(target_sig, (0, lag), mode='constant')[lag:]
            else:
                # Target מאחר, צריך לחתוך התחלה
                lag = abs(lag)
                aligned = np.pad(target_sig, (lag, 0), mode='constant')[:len(target_sig)]
        else:
            aligned = target_sig

        # וידוא אורך זהה למקור
        if len(aligned) > len(ref_sig):
            aligned = aligned[:len(ref_sig)]
        elif len(aligned) < len(ref_sig):
            aligned = np.pad(aligned, (0, len(ref_sig) - len(aligned)), mode='constant')
            
        return aligned

   # we want to find all the time stamps (tuples of (start, end) for parts) where drums or bass play, so we only process vocals on those parts
    def find_drum_bass_timestamps(self, threshold=0.02, min_silence_len=0.3, sr=44100) -> list[tuple[float, float]]:
        drums_file = self.demucs_out_dir / "drums.mp3"
        if not drums_file.exists(): drums_file = self.demucs_out_dir / "drums.wav"
        bass_file = self.demucs_out_dir / "bass.mp3"
        if not bass_file.exists(): bass_file = self.demucs_out_dir / "bass.wav" 
        if not drums_file.exists() and not bass_file.exists():
            return []
        y_drums, _ = librosa.load(str(drums_file), sr=sr, mono=True) if drums_file.exists() else np.zeros(1)
        y_bass, _ = librosa.load(str(bass_file), sr=sr, mono=True) if bass_file.exists() else np.zeros(1)
        y_combined = y_drums + y_bass
        energy = librosa.feature.rms(y=y_combined, frame_length=2048, hop_length=512)[0]
        frames = np.where(energy > threshold)[0]
        if len(frames) == 0:
            return []
        times = librosa.frames_to_time(frames, sr=sr, hop_length=512)
        timestamps = []
        start = times[0]
        for i in range(1, len(times)):
            if times[i] - times[i-1] > min_silence_len:
                end = times[i-1]
                timestamps.append((start, end))
                start = times[i]
        timestamps.append((start, times[-1]))
        return timestamps

    def process_drums(self):
        print("[*] Enhancing Drums...")
        drums_file = self.demucs_out_dir / "drums.mp3"
        if not drums_file.exists(): drums_file = self.demucs_out_dir / "drums.wav"
        if not drums_file.exists(): return None

        y, sr = librosa.load(str(drums_file), sr=None, mono=True)
        # Soft Clipper
        y = np.tanh(y * 0.9)
        # EQ
        sos_hp = butter(2, 40, 'hp', fs=sr, output='sos')
        y = sosfilt(sos_hp, y)
        sos_lp = butter(2, 15000, 'lp', fs=sr, output='sos')
        y = sosfilt(sos_lp, y)
        
        output_path = self.demucs_out_dir / "drums_clean_synth.wav"
        self.save_wav_safe(output_path, y, sr)
        return output_path

    def process_bass(self):
        print("[*] Enhancing Bass...")
        bass_file = self.demucs_out_dir / "bass.mp3"
        if not bass_file.exists(): bass_file = self.demucs_out_dir / "bass.wav"
        if not bass_file.exists(): return None

        y, sr = librosa.load(str(bass_file), sr=None, mono=True)
        # Saturation & EQ
        y = np.tanh(y * 0.85)
        sos_hp = butter(2, 20, 'hp', fs=sr, output='sos')
        y = sosfilt(sos_hp, y)
        sos_lp = butter(2, 300, 'lp', fs=sr, output='sos')
        bass_low = sosfilt(sos_lp, y) * 1.2
        # מיקס של באס נמוך מודגש עם המקור
        y = y * 0.8 + bass_low * 0.2
        
        output_path = self.demucs_out_dir / "bass_clean_synth.wav"
        self.save_wav_safe(output_path, y, sr)
        return output_path

    def process_vocals(self):
        print("[*] Restoring Vocals (Parallel Strategy)...")
        vocals_path = self.demucs_out_dir / "vocals.mp3"
        if not vocals_path.exists(): vocals_path = vocals_path.with_suffix(".wav")
        if not vocals_path.exists():
            print("[!] No vocals stem found.")
            return None

        out_path = self.demucs_out_dir / "vocals_restored.wav"
        
        # טעינת המקור (חשוב לטעון ב-44100 לסטנדרטיזציה)
        y_original, sr = librosa.load(str(vocals_path), sr=44100, mono=True)

        if HAS_VOICEFIXER and NO_VOCALS is False:
            try:
                print("    -> Generating Clean Body with VoiceFixer...")
                vf = VoiceFixer()
                # קובץ זמני לפלט של המודל
                temp_vf = self.demucs_out_dir / "temp_vf.wav"
                
                # Mode 0 נותן את הניקוי החזק ביותר (אבל מאבד פרטים)
                vf.restore(input=str(vocals_path), output=str(temp_vf), cuda=torch.cuda.is_available(), mode=0)
                
                # טעינת התוצאה הנקייה
                y_clean, _ = librosa.load(str(temp_vf), sr=sr, mono=True)
                
                # מחיקת קובץ זמני
                if temp_vf.exists(): os.remove(temp_vf)

                # --- שלב 1: סנכרון ---
                y_clean = self.align_signals(y_original, y_clean)

                # --- שלב 2: חילוץ ה"אוויר" מהמקור ---
                print("    -> Extracting High-End Detail from original...")
                # פילטר High-Pass אגרסיבי ב-9kHz
                # הדיסטורשן של הפונק לרוב נמצא ב-Mids, הגבוהים יחסית נקיים
                sos_high = butter(6, 9000, 'hp', fs=sr, output='sos')
                y_air = sosfilt(sos_high, y_original)

                # --- שלב 3: המיקס המקבילי ---
                # 80% גוף נקי + 60% אוויר מקורי (קצת חפיפה לפצות על חיתוכים)
                y_final = (y_clean * 0.8) + (y_air * 0.6)
                
            except Exception as e:
                print(f"[!] VoiceFixer error: {e}. Fallback to EQ.")
                y_final = y_original
        else:
            # Fallback ללא AI
            print("    -> ","VoiceFixer not available or disabled." if not NO_VOCALS else "Vocals processing disabled by user.", "Using EQ fallback.")
            y_final = y_original

        self.save_wav_safe(out_path, y_final, sr)
        return out_path

    def mix_final_track(self, clean_drums_path, clean_bass_path, vocals_path=None):
        print("[*] Mixing final track...")
        try:
            other_path = self.demucs_out_dir / "other.mp3"
            if not other_path.exists(): other_path = other_path.with_suffix('.wav')

            vocals = AudioSegment.from_file(str(vocals_path)) if vocals_path else AudioSegment.silent(duration=1000)
            other = AudioSegment.from_file(str(other_path))
            
            # וידוא אחידות Sample Rate
            target_fr = vocals.frame_rate

            if clean_drums_path and clean_drums_path.exists():
                new_drums = AudioSegment.from_wav(str(clean_drums_path))
                if new_drums.frame_rate!= target_fr: new_drums = new_drums.set_frame_rate(target_fr)
            else:
                new_drums = AudioSegment.silent(duration=len(vocals), frame_rate=target_fr)

            if clean_bass_path and clean_bass_path.exists():
                new_bass = AudioSegment.from_wav(str(clean_bass_path))
                if new_bass.frame_rate!= target_fr: new_bass = new_bass.set_frame_rate(target_fr)
            else:
                new_bass = AudioSegment.silent(duration=len(vocals), frame_rate=target_fr)

            # פונקציית עזר להמרת ערוצים
            def _normalize_channels(seg, target=2):
                if seg.channels == target: return seg
                if seg.channels!= 1 and target!= 1: seg = seg.set_channels(1)
                return seg.set_channels(target)

            target_channels = 2
            vocals = _normalize_channels(vocals, target_channels)
            other = _normalize_channels(other, target_channels)
            new_drums = _normalize_channels(new_drums, target_channels)
            new_bass = _normalize_channels(new_bass, target_channels)

            # כיוון עוצמות סופי
            drums_gain_db = -1.5
            bass_gain_db = 1.0

            final_mix = (
                vocals
               .overlay(other)
               .overlay(new_drums + drums_gain_db)
               .overlay(new_bass + bass_gain_db)
            )

            out_file = self.output_dir / f"{self.track_name}_CLEANED_PHONK.mp3"
            final_mix.export(str(out_file), format="mp3")
            print(f"\n SUCCESS! Track saved at:\n{out_file.absolute()}")

        except Exception as e:
            print(f"[!] Mixing failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # הקפד לשנות את שם הקובץ לקובץ שלך
    INPUT_FILE = "Song_name.mp3"
    from sys import argv
    if len(argv) > 1:
        INPUT_FILE = argv[1]
        NO_VOCALS = argv[2].lower() in ["no_vocals",'-no_vocals'] if len(argv) > 2 else False
    cleaner = PhonkCleaner(INPUT_FILE)
    cleaner.separate_stems()
    
    clean_drums = cleaner.process_drums()
    clean_bass = cleaner.process_bass()
    restored_vocals = cleaner.process_vocals()
    
    cleaner.mix_final_track(clean_drums, clean_bass, restored_vocals)