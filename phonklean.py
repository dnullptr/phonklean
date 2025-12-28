import argparse
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

    def attenuate_drum_bleed(self, y, sr, margin_db=2.0, perc_reduce=0.3):
        """
        HPSS-based drum bleed removal: separate harmonic (vocal sustain) from 
        percussive (drum transients) and blend to remove over-gated drum artifacts.
        
        Args:
            y: audio signal
            sr: sample rate
            margin_db: HPSS margin (higher = more separation)
            perc_reduce: percussive blend (0.0 = remove all, 0.3 = keep 30%)
        
        Returns:
            cleaned audio signal
        """
        try:
            D = librosa.stft(y)
            H, P = librosa.decompose.hpss(D, margin=margin_db)
            
            if perc_reduce == 0.0:
                # Pure harmonic (most aggressive)
                D_clean = H
            else:
                # Blend harmonic + reduced percussive
                D_clean = H + P * perc_reduce
            
            y_clean = librosa.istft(D_clean)
            print(f"       ✓ HPSS bleed removal (margin={margin_db}dB, perc_blend={perc_reduce})")
            return y_clean
        except Exception as e:
            print(f"       [!] HPSS failed: {e}, skipping bleed removal.")
            return y

    def process_vocals(self,no_vocals=True):
        print("[*] Restoring Vocals (Parallel Strategy)...")
        vocals_path = self.demucs_out_dir / "vocals.mp3"
        if not vocals_path.exists(): vocals_path = vocals_path.with_suffix(".wav")
        if not vocals_path.exists():
            print("[!] No vocals stem found.")
            return None

        out_path = self.demucs_out_dir / "vocals_restored.wav"
        
        # טעינת המקור (חשוב לטעון ב-44100 לסטנדרטיזציה)
        y_original, sr = librosa.load(str(vocals_path), sr=44100, mono=True)

        if HAS_VOICEFIXER and all([NO_VOCALS,no_vocals]) is False:
            try:
                print(f"    -> Generating Clean Body with VoiceFixer...{' (Mode '+str(VF_MODE)+')'}")
                vf = VoiceFixer()
                # קובץ זמני לפלט של המודל
                temp_vf = self.demucs_out_dir / "temp_vf.wav"
                
                # Mode 0 נותן את הניקוי החזק ביותר (אבל מאבד פרטים)
                vf.restore(input=str(vocals_path), output=str(temp_vf), cuda=torch.cuda.is_available(), mode=VF_MODE)
                
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

        # --- Apply transient-aware spectral attenuation to remove drum bleed (optional) ---
        if BLEED_REMOVAL:
            print(f"    -> Attenuating drum bleed with HPSS (margin={BLEED_MARGIN}dB, blend={BLEED_BLEND})...")
            y_final = self.attenuate_drum_bleed(y_final, sr, margin_db=BLEED_MARGIN, perc_reduce=BLEED_BLEND)
        else:
            print("    -> Drum bleed removal disabled (use --bleed-removal to enable)")

        self.save_wav_safe(out_path, y_final, sr)
        return out_path

    def apply_autotune_to_vocals(self, vocals_path, sr=44100, key='C', scale='chromatic', strength=1.0):
        """
        Simple vocals-only autotune: per-frame F0 estimation, snap to nearest note
        in selected scale, pitch-shift each frame and overlap-add with strength mix.
        """
        print(f"[*] Applying autotune to vocals: key={key}, scale={scale}, strength={strength}")
        vocals_path = Path(vocals_path)
        if not vocals_path.exists():
            print(f"    -> Source not found for autotune: {vocals_path}")
            return None

        y, sr = librosa.load(str(vocals_path), sr=sr, mono=True)
        if len(y) == 0:
            print("    -> Empty vocals file, skipping autotune.")
            return None

        frame_length = 2048
        hop_length = 256

        # If key is auto, try to detect key & scale from vocals
        if isinstance(key, str) and key.lower() in ("auto", "guess"):
            try:
                det_key, det_scale = self.detect_key_scale(y, sr)
                print(f"    -> Detected key: {det_key} {det_scale}")
                key = det_key
                if scale == 'chromatic':
                    # prefer detected scale when user left scale as chromatic
                    scale = det_scale
            except Exception as e:
                print(f"    -> Key detection failed: {e}")

        # Pitch estimation
        try:
            f0 = librosa.pyin(y, fmin=65, fmax=2000, sr=sr, frame_length=frame_length, hop_length=hop_length)
        except Exception:
            # fallback to yin if pyin unavailable
            f0 = librosa.yin(y, fmin=65, fmax=2000, sr=sr, frame_length=frame_length, hop_length=hop_length)

        # map key name to semitone (C=0 ... B=11)
        note_map = {'C':0,'C#':1,'DB':1,'D':2,'D#':3,'EB':3,'E':4,'F':5,'F#':6,'GB':6,'G':7,'G#':8,'AB':8,'A':9,'A#':10,'BB':10,'B':11}
        k = key.upper().replace('MINOR','').strip()
        root_pc = note_map.get(k[:2] if k[:2] in note_map else k[:1], 0)

        if scale == 'chromatic':
            allowed_pc = list(range(12))
        elif scale == 'major':
            allowed_pc = [(root_pc + i) % 12 for i in [0,2,4,5,7,9,11]]
        else: # minor
            allowed_pc = [(root_pc + i) % 12 for i in [0,2,3,5,7,8,10]]

        # Prepare output buffers
        y_out = np.zeros_like(y)
        weight = np.zeros_like(y)
        window = np.hanning(frame_length)

        n_frames = int(np.ceil((len(y) - frame_length) / hop_length)) + 1
        for i in range(n_frames):
            start = i * hop_length
            if start >= len(y):
                break
            L = min(frame_length, len(y) - start)
            frame = y[start:start+L]

            # padded version for pitch-shifting operations
            if L < frame_length:
                frame_p = np.pad(frame, (0, frame_length - L), mode='constant')
            else:
                frame_p = frame

            # estimate f0 for this frame index
            if i < len(f0):
                raw = f0[i]
                if isinstance(raw, np.ndarray) or (hasattr(raw, '__len__') and not isinstance(raw, (str, bytes))):
                    try:
                        f0_i = float(np.array(raw).flatten()[0])
                    except Exception:
                        f0_i = np.nan
                else:
                    try:
                        f0_i = float(raw)
                    except Exception:
                        f0_i = np.nan
            else:
                f0_i = np.nan

            # default shifted is the original padded frame
            shifted_p = frame_p
            if np.isfinite(f0_i) and f0_i > 1e-6:
                try:
                    current_midi = librosa.hz_to_midi(f0_i)
                    if np.isfinite(current_midi):
                        center = int(round(current_midi))
                        candidates = list(range(center-6, center+7))
                        best = min(candidates, key=lambda n: abs(n - current_midi) if (n % 12) in allowed_pc else 1e6)
                        if (best % 12) not in allowed_pc:
                            best = int(round(current_midi))
                        semitone_diff = best - current_midi
                        try:
                            shifted_p = librosa.effects.pitch_shift(frame_p, sr, n_steps=semitone_diff)
                            if len(shifted_p) < frame_length:
                                shifted_p = np.pad(shifted_p, (0, frame_length - len(shifted_p)), mode='constant')
                            elif len(shifted_p) > frame_length:
                                shifted_p = shifted_p[:frame_length]
                        except Exception:
                            shifted_p = frame_p
                except Exception:
                    shifted_p = frame_p

            mixed_p = (1.0 - strength) * frame_p + strength * shifted_p
            w = window[:L]
            y_out[start:start+L] += mixed_p[:L] * w
            weight[start:start+L] += w

        # normalize
        nonzero = weight > 1e-8
        y_final = np.copy(y)
        y_final[nonzero] = y_out[nonzero] / weight[nonzero]

        out_path = self.demucs_out_dir / "vocals_autotuned.wav"
        self.save_wav_safe(out_path, y_final, sr)
        return out_path

    def detect_key_scale(self, y, sr=44100):
        """Estimate key and scale (major/minor) from audio using chroma and Krumhansl templates.

        Returns (key_name, scale_name) where key_name is e.g. 'C'..'B' and scale_name is 'major' or 'minor'.
        """
        # Templates from Krumhansl (typical weights) for major and minor keys
        major_template = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor_template = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

        # compute chroma
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        except Exception:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        chroma_mean = np.mean(chroma, axis=1)
        if np.all(chroma_mean == 0):
            raise RuntimeError("Empty chroma, cannot detect key")

        chroma_norm = chroma_mean / np.linalg.norm(chroma_mean)

        best_key = None
        best_score = -np.inf
        best_scale = 'major'
        note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

        # test rotations
        for root in range(12):
            maj = np.roll(major_template, -root)
            min_t = np.roll(minor_template, -root)
            maj_norm = maj / np.linalg.norm(maj)
            min_norm = min_t / np.linalg.norm(min_t)
            score_maj = np.dot(chroma_norm, maj_norm)
            score_min = np.dot(chroma_norm, min_norm)
            if score_maj > best_score:
                best_score = score_maj
                best_key = note_names[root]
                best_scale = 'major'
            if score_min > best_score:
                best_score = score_min
                best_key = note_names[root]
                best_scale = 'minor'

        return best_key, best_scale
    
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
            bass_gain_db = 2.0

            final_mix = (
                vocals
               .overlay(other)
               .overlay(new_drums + drums_gain_db)
               .overlay(new_bass + bass_gain_db)
            )

            out_file = self.output_dir / f"{self.track_name}_CLEANED_PHONK.{'mp3' if OUTPUT_FORMAT=='mp3' else 'wav'}"
            final_mix.export(str(out_file), format=OUTPUT_FORMAT, bitrate="320k" if OUTPUT_FORMAT=='mp3' else None)
            print(f"\n SUCCESS! Track saved at:\n{out_file.absolute()}")

        except Exception as e:
            print(f"[!] Mixing failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phonklean - Audio Restoration Tool",
                                     epilog="Made with hate for 90s over-saturated instruments by dnullptr.")
    
    # 1. Input File
    parser.add_argument(
        "input_file", 
        nargs="?", 
        default="Song_name.mp3",
        help="Path to the input audio file"
    )
    
    # 2. No Vocals flag (sets arg to True if present)
    parser.add_argument(
        "--no-vocals", "-nv",
        action="store_true",
        help="Skip vocal processing"
    )

    # 3. VoiceFixer mode selection (if available)
    parser.add_argument(
        "--vf-mode", "-vfm",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="VoiceFixer mode: 0 (Default), 1 (Adds pre-processing and high-freq. cut), 2 (train mode)"
    )

    # 4. Autotune options (vocals-only, independent)
    parser.add_argument(
        "--autotune-vocals",
        action="store_true",
        help="Apply simple autotune (pitch-correction) to vocals stem only"
    )

    parser.add_argument(
        "--autotune-key",
        type=str,
        default="auto",
        help="Root key for autotune (e.g. C, D#, F)"
    )

    parser.add_argument(
        "--autotune-scale",
        type=str,
        choices=["chromatic", "major", "minor"],
        default="chromatic",
        help="Scale used by autotune to snap pitches"
    )

    parser.add_argument(
        "--autotune-strength",
        type=float,
        default=1.0,
        help="Autotune strength (0.0 = no effect, 1.0 = full correction)"
    )

    # 5. Drum bleed removal (HPSS-based)
    parser.add_argument(
        "--bleed-removal",
        action="store_true",
        help="Remove over-gated drum bleed from vocals using HPSS (harmonic-percussive separation)"
    )

    parser.add_argument(
        "--bleed-margin",
        type=float,
        default=2.0,
        help="HPSS margin in dB (higher = more separation, default=2.0)"
    )

    parser.add_argument(
        "--bleed-blend",
        type=float,
        default=0.3,
        help="Percussive blend ratio (0.0 = remove all, 0.5 = 50%%, default=0.3)"
    )

    # Add argument for audio format (mp3/wav)
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["mp3", "wav"],
        default="mp3",
        help="Output audio format for the final mixed track (default=mp3)"
    )

    args = parser.parse_args()
    global NO_VOCALS
    INPUT_FILE = args.input_file
    NO_VOCALS = args.no_vocals
    VF_MODE = int(args.vf_mode)
    OUTPUT_FORMAT = args.output_format
    AUTOTUNE_VOCALS = bool(args.autotune_vocals)
    BLEED_REMOVAL = bool(args.bleed_removal)
    BLEED_MARGIN = float(args.bleed_margin)
    BLEED_BLEND = float(args.bleed_blend)
    AUTOTUNE_KEY = args.autotune_key
    AUTOTUNE_SCALE = args.autotune_scale
    AUTOTUNE_STRENGTH = max(0.0, min(1.0, float(args.autotune_strength)))

    cleaner = PhonkCleaner(INPUT_FILE)
    cleaner.separate_stems()
    
    clean_drums = cleaner.process_drums()
    clean_bass = cleaner.process_bass()
    restored_vocals = cleaner.process_vocals()
    autotuned_vocals = None
    if AUTOTUNE_VOCALS:
        # choose source for autotune: prefer processed vocals, fall back to raw stem
        if restored_vocals and Path(restored_vocals).exists():
            source_for_autotune = restored_vocals
        else:
            possible = cleaner.demucs_out_dir / "vocals.mp3"
            if not possible.exists():
                possible = possible.with_suffix('.wav')
            source_for_autotune = possible if possible.exists() else None

        if source_for_autotune is None:
            print("[!] Autotune requested but no vocals stem available.")
        else:
            autotuned_vocals = cleaner.apply_autotune_to_vocals(
                source_for_autotune,
                sr=44100,
                key=AUTOTUNE_KEY,
                scale=AUTOTUNE_SCALE,
                strength=AUTOTUNE_STRENGTH
            )

    final_vocals = autotuned_vocals if autotuned_vocals else restored_vocals
    
    cleaner.mix_final_track(clean_drums, clean_bass, final_vocals)