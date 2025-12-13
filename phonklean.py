import sys
import os
import shutil
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
from scipy.signal import medfilt, butter, sosfilt
from pydub import AudioSegment
from pathlib import Path
import demucs.separate

class PhonkCleaner:
    def __init__(self, input_file_path):
        self.input_path = Path(input_file_path).resolve()
        
        if not self.input_path.exists():
            print(f"[!] Critical Error: Input file not found: {self.input_path}")
            sys.exit(1)

        self.track_name = self.input_path.stem
        self.output_dir = Path("phonk_cleaned_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # נתיב היעד של Demucs
        self.demucs_out_dir = self.output_dir / "htdemucs_ft" / self.track_name

    def separate_stems(self):
        print(f"[*] Starting Demucs separation for: {self.input_path.name}")
        
        # בניית פקודת ההפרדה
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
        """ שמירה בטוחה בפורמט סטריאו כדי למנוע קריסות ב-Mix """
        # נרמול למניעת דיסטורשן דיגיטלי
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val * 0.90
        # תקן צורות נתונים שונות:
        # - ספק מערכים חד-ממדיים (n,) -> המר לסטריאו על ידי שכפול ערוץ
        # - אם יש צורה (1, n) או (n, 1) -> נשטח קודם
        # - אם יש צורה (2, n) שנוצרה בטעות, נהפוך ל-(n,2)
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # (1, n) -> (n,)
                if data.shape[0] == 1 and data.shape[1] > 1:
                    data = data.flatten()
                # (n, 1) -> (n,)
                elif data.shape[1] == 1:
                    data = data[:, 0]
                # (2, n) -> transpose to (n, 2)
                elif data.shape[0] == 2 and data.shape[1] > 2:
                    data = data.T
            # עכשיו אם זה עדיין חד-ממדי, שכפול לערוץ סטראו
            if data.ndim == 1:
                data = np.column_stack((data, data))
        else:
            data = np.array(data)
            if data.ndim == 1:
                data = np.column_stack((data, data))
        
        # המרה ל-16 ביט
        data_int16 = (data * 32767).astype(np.int16)
        
        try:
            wavfile.write(str(path), sr, data_int16)
            print(f"    -> Saved Stereo WAV: {path.name}")
        except Exception as e:
            print(f"[!] Failed to save {path.name}: {e}")

    def process_drums(self):
        print("[*] Enhancing Drums (de-clip & denoise)...")
        drums_file = self.demucs_out_dir / "drums.mp3"
        if not drums_file.exists(): drums_file = self.demucs_out_dir / "drums.wav"
            
        if not drums_file.exists(): return None

        y, sr = librosa.load(str(drums_file), sr=None, mono=True)
        
        # Soft-clip detection and reduction: saturate at 0.95 to tame clipping
        clip_threshold = 0.95
        y_clipped = np.abs(y) > clip_threshold
        if np.sum(y_clipped) > 0:
            print(f"    -> Detected {np.sum(y_clipped)} clipped samples, applying soft-clip reduction...")
            y = np.tanh(y * 0.9)  # gentle soft-clip
        
        # Light high-pass to remove mud, gentle low-pass to smooth digital artifacts
        sos_hp = butter(2, 40, 'hp', fs=sr, output='sos')
        y = sosfilt(sos_hp, y)
        
        sos_lp = butter(2, 15000, 'lp', fs=sr, output='sos')
        y = sosfilt(sos_lp, y)
        
        # Normalize to safe level
        max_amp = np.max(np.abs(y))
        if max_amp > 0:
            y = y / max_amp * 0.92

        output_path = self.demucs_out_dir / "drums_clean_synth.wav"
        self.save_wav_safe(output_path, y, sr)
        return output_path

    def process_bass(self):
        print("[*] Enhancing Bass (de-clip & restore tone)...")
        bass_file = self.demucs_out_dir / "bass.mp3"
        if not bass_file.exists(): bass_file = self.demucs_out_dir / "bass.wav"
            
        if not bass_file.exists(): return None

        y, sr = librosa.load(str(bass_file), sr=None, mono=True)

        # Step 1: Detect and reduce clipping (soft-clip distortion common in phonk)
        clip_threshold = 0.95
        y_clipped = np.abs(y) > clip_threshold
        if np.sum(y_clipped) > 0:
            print(f"    -> Detected {np.sum(y_clipped)} clipped samples, applying soft-clip reduction...")
            # Apply gentle tanh to recover lost detail
            y = np.tanh(y * 0.85)
        
        # Step 2: Spectral restoration – boost 20-300Hz (sub-bass) where distortion removed detail
        sos_hp = butter(2, 20, 'hp', fs=sr, output='sos')
        y = sosfilt(sos_hp, y)
        
        sos_lp = butter(2, 300, 'lp', fs=sr, output='sos')
        bass_low = sosfilt(sos_lp, y)
        bass_low = bass_low * 1.2  # subtle boost to recovered sub-bass
        
        # Blend: original filtered + boosted sub-bass
        y = y * 0.8 + bass_low * 0.2
        
        # Normalize
        max_amp = np.max(np.abs(y))
        if max_amp > 0:
            y = y / max_amp * 0.92
        
        output_path = self.demucs_out_dir / "bass_clean_synth.wav"
        self.save_wav_safe(output_path, y, sr)
        return output_path

    def mix_final_track(self, clean_drums_path, clean_bass_path):
        print("[*] Mixing final track...")
        try:
            # טעינת ערוצים מקוריים
            vocals_path = self.demucs_out_dir / "vocals.mp3"
            other_path = self.demucs_out_dir / "other.mp3"
            
            if not vocals_path.exists(): vocals_path = vocals_path.with_suffix('.wav')
            if not other_path.exists(): other_path = other_path.with_suffix('.wav')

            # טעינה בטוחה
            vocals = AudioSegment.from_file(str(vocals_path))
            other = AudioSegment.from_file(str(other_path))
            
            # וידוא שכל הערוצים הם באותו קצב דגימה (למניעת גליצ'ים)
            target_fr = vocals.frame_rate
            
            if clean_drums_path and clean_drums_path.exists():
                new_drums = AudioSegment.from_wav(str(clean_drums_path))
                # התאמה כפויה לווקאל
                if new_drums.frame_rate!= target_fr: new_drums = new_drums.set_frame_rate(target_fr)
            else:
                new_drums = AudioSegment.silent(duration=len(vocals), frame_rate=target_fr)

            if clean_bass_path and clean_bass_path.exists():
                new_bass = AudioSegment.from_wav(str(clean_bass_path))
                if new_bass.frame_rate!= target_fr: new_bass = new_bass.set_frame_rate(target_fr)
            else:
                new_bass = AudioSegment.silent(duration=len(vocals), frame_rate=target_fr)

            # Ensure channel counts are compatible for overlay.
            def _normalize_channels(seg, target=2):
                if seg.channels == target:
                    return seg
                # pydub only supports mono<->multi conversions directly.
                # If source has >1 channels and target is >1, first downmix to mono,
                # then expand to the requested stereo target to avoid ValueError.
                if seg.channels != 1 and target != 1:
                    seg = seg.set_channels(1)
                return seg.set_channels(target)

            target_channels = 2
            vocals = _normalize_channels(vocals, target_channels)
            other = _normalize_channels(other, target_channels)
            new_drums = _normalize_channels(new_drums, target_channels)
            new_bass = _normalize_channels(new_bass, target_channels)

            # apply per-stem gain adjustments (dB)
            drums_gain_db = -2
            bass_gain_db = 6

            # המיקס עצמו - כעת הכל אמור להיות בפורמט תואם
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
    INPUT_FILE = "Montagem_Xonada.mp3" # וודא שזה שם הקובץ שלך
    
    cleaner = PhonkCleaner(INPUT_FILE)
    cleaner.separate_stems()
    
    clean_drums = cleaner.process_drums()
    clean_bass = cleaner.process_bass()
    
    cleaner.mix_final_track(clean_drums, clean_bass)