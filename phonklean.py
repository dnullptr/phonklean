import sys
import os
import shutil
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
from pathlib import Path
import demucs.separate
import torch

# Try importing VoiceFixer
try:
    from voicefixer import VoiceFixer
    HAS_VOICEFIXER = True
    print("[*] VoiceFixer detected! Vocals will be AI restored.")
except ImportError:
    HAS_VOICEFIXER = False
    print("[*] VoiceFixer not found. Vocals will be used raw.")
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
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val * 0.90

        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                if data.shape[0] == 1 and data.shape[1] > 1:
                    data = data.flatten()
                elif data.shape[1] == 1:
                    data = data[:, 0]
                elif data.shape[0] == 2 and data.shape[1] > 2:
                    data = data.T
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

    def process_drums(self):
        print("[*] Enhancing Drums...")
        drums_file = self.demucs_out_dir / "drums.mp3"
        if not drums_file.exists(): drums_file = self.demucs_out_dir / "drums.wav"
        if not drums_file.exists(): return None

        y, sr = librosa.load(str(drums_file), sr=None, mono=True)
        y = np.tanh(y * 0.9)  # soft-clip
        sos_hp = butter(2, 40, 'hp', fs=sr, output='sos')
        y = sosfilt(sos_hp, y)
        sos_lp = butter(2, 15000, 'lp', fs=sr, output='sos')
        y = sosfilt(sos_lp, y)
        y = y / np.max(np.abs(y)) * 0.92

        output_path = self.demucs_out_dir / "drums_clean_synth.wav"
        self.save_wav_safe(output_path, y, sr)
        return output_path

    def process_bass(self):
        print("[*] Enhancing Bass...")
        bass_file = self.demucs_out_dir / "bass.mp3"
        if not bass_file.exists(): bass_file = self.demucs_out_dir / "bass.wav"
        if not bass_file.exists(): return None

        y, sr = librosa.load(str(bass_file), sr=None, mono=True)
        y = np.tanh(y * 0.85)
        sos_hp = butter(2, 20, 'hp', fs=sr, output='sos')
        y = sosfilt(sos_hp, y)
        sos_lp = butter(2, 300, 'lp', fs=sr, output='sos')
        bass_low = sosfilt(sos_lp, y) * 1.2
        y = y * 0.8 + bass_low * 0.2
        y = y / np.max(np.abs(y)) * 0.92

        output_path = self.demucs_out_dir / "bass_clean_synth.wav"
        self.save_wav_safe(output_path, y, sr)
        return output_path

    def process_vocals(self):
        print("[*] Restoring Vocals...")
        vocals_path = self.demucs_out_dir / "vocals.mp3"
        if not vocals_path.exists():
            vocals_path = vocals_path.with_suffix(".wav")
        if not vocals_path.exists():
            print("[!] No vocals stem found.")
            return None

        out_path = self.demucs_out_dir / "vocals_restored.wav"

        if HAS_VOICEFIXER:
            try:
                vf = VoiceFixer()
                vf.restore(
                    input=str(vocals_path),
                    output=str(out_path),
                    cuda=torch.cuda.is_available(),
                    mode=0
                )
                print("    -> VoiceFixer AI restoration complete.")
                return out_path
            except Exception as e:
                print(f"[!] VoiceFixer failed: {e}. Using raw vocals instead.")
                return vocals_path
        else:
            return vocals_path

    def mix_final_track(self, clean_drums_path, clean_bass_path, vocals_path=None):
        print("[*] Mixing final track...")
        try:
            other_path = self.demucs_out_dir / "other.mp3"
            if not other_path.exists():
                other_path = other_path.with_suffix('.wav')

            vocals = AudioSegment.from_file(str(vocals_path)) if vocals_path else AudioSegment.silent(duration=1000)
            other = AudioSegment.from_file(str(other_path))
            target_fr = vocals.frame_rate

            if clean_drums_path and clean_drums_path.exists():
                new_drums = AudioSegment.from_wav(str(clean_drums_path))
                if new_drums.frame_rate != target_fr:
                    new_drums = new_drums.set_frame_rate(target_fr)
            else:
                new_drums = AudioSegment.silent(duration=len(vocals), frame_rate=target_fr)

            if clean_bass_path and clean_bass_path.exists():
                new_bass = AudioSegment.from_wav(str(clean_bass_path))
                if new_bass.frame_rate != target_fr:
                    new_bass = new_bass.set_frame_rate(target_fr)
            else:
                new_bass = AudioSegment.silent(duration=len(vocals), frame_rate=target_fr)

            def _normalize_channels(seg, target=2):
                if seg.channels == target:
                    return seg
                if seg.channels != 1 and target != 1:
                    seg = seg.set_channels(1)
                return seg.set_channels(target)

            target_channels = 2
            vocals = _normalize_channels(vocals, target_channels)
            other = _normalize_channels(other, target_channels)
            new_drums = _normalize_channels(new_drums, target_channels)
            new_bass = _normalize_channels(new_bass, target_channels)

            drums_gain_db = -2
            bass_gain_db = 2

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
    INPUT_FILE = "has2bu.mp3"
    
    cleaner = PhonkCleaner(INPUT_FILE)
    cleaner.separate_stems()
    
    clean_drums = cleaner.process_drums()
    clean_bass = cleaner.process_bass()
    restored_vocals = cleaner.process_vocals()
    
    cleaner.mix_final_track(clean_drums, clean_bass, restored_vocals)
