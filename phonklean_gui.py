import gradio as gr
import os
from pathlib import Path
import numpy as np
from phonklean import PhonkCleaner
import librosa
import torch

class PhonkCleanerGUI:
    def __init__(self):
        self.cleaner = None
        self.current_file = None
        
    def process_audio(self, input_file, no_vocals, vf_mode, autotune_vocals, 
                     autotune_key, autotune_scale, autotune_strength,
                     bleed_removal, bleed_margin, bleed_blend):
        """
        Main processing function that orchestrates the entire phonklean pipeline
        """
        try:
            if input_file is None:
                return "❌ Error: No input file selected", None, None
            
            # Get the file path from the gradio file object
            file_path = input_file.name if hasattr(input_file, 'name') else input_file
            
            if not Path(file_path).exists():
                return f"❌ Error: File not found: {file_path}", None, None
            
            # Initialize cleaner
            self.cleaner = PhonkCleaner(file_path)
            self.current_file = file_path
            
            status_updates = []
            status_updates.append(f"🔄 Processing: {Path(file_path).name}")
            status_updates.append("\n📊 Pipeline Status:")
            
            # Set global variables for the phonklean module
            import phonklean as pk
            pk.NO_VOCALS = no_vocals
            pk.VF_MODE = int(vf_mode)
            pk.AUTOTUNE_VOCALS = autotune_vocals
            pk.BLEED_REMOVAL = bleed_removal
            pk.BLEED_MARGIN = float(bleed_margin)
            pk.BLEED_BLEND = float(bleed_blend)
            pk.AUTOTUNE_KEY = autotune_key
            pk.AUTOTUNE_SCALE = autotune_scale
            pk.AUTOTUNE_STRENGTH = max(0.0, min(1.0, float(autotune_strength)))
            
            # Step 1: Separate stems
            status_updates.append("1️⃣  Separating stems with htdemucs_ft...")
            self.cleaner.separate_stems()
            status_updates.append("   ✅ Stems separated")
            
            # Step 2: Process drums
            status_updates.append("2️⃣  Enhancing drums...")
            clean_drums = self.cleaner.process_drums()
            if clean_drums:
                status_updates.append("   ✅ Drums enhanced")
            else:
                status_updates.append("   ⚠️  No drums to process")
            
            # Step 3: Process bass
            status_updates.append("3️⃣  Enhancing bass...")
            clean_bass = self.cleaner.process_bass()
            if clean_bass:
                status_updates.append("   ✅ Bass enhanced")
            else:
                status_updates.append("   ⚠️  No bass to process")
            
            # Step 4: Process vocals (if enabled)
            restored_vocals = None
            status_updates.append(f"4️⃣  Restoring vocals {'with Voice-Fixer' if not no_vocals else 'without Voice-Fixer'}...")
            restored_vocals = self.cleaner.process_vocals(no_vocals)
            if restored_vocals:
                status_updates.append("   ✅ Vocals restored")
            else:
                status_updates.append("   ⚠️  No vocals to process")
     
            
            # Step 5: Apply autotune (if enabled)
            autotuned_vocals = None
            if autotune_vocals and restored_vocals:
                status_updates.append("5️⃣  Applying autotune...")
                autotuned_vocals = self.cleaner.apply_autotune_to_vocals(
                    restored_vocals,
                    sr=44100,
                    key=autotune_key,
                    scale=autotune_scale,
                    strength=autotune_strength
                )
                if autotuned_vocals:
                    status_updates.append("   ✅ Autotune applied")
                else:
                    status_updates.append("   ⚠️  Autotune failed, using original vocals")
            
            # Step 6: Mix final track
            status_updates.append("6️⃣  Mixing final track...")
            final_vocals = autotuned_vocals if autotuned_vocals else restored_vocals
            self.cleaner.mix_final_track(clean_drums, clean_bass, final_vocals)
            status_updates.append("   ✅ Track mixed and exported")
            
            # Find output file
            output_file = self.cleaner.output_dir / f"{self.cleaner.track_name}_CLEANED_PHONK.mp3"
            
            status_updates.append("\n✨ Processing complete!")
            status_updates.append(f"📁 Output: {output_file.absolute()}")
            
            # Load and return the output audio
            output_audio = None
            if output_file.exists():
                y, sr = librosa.load(str(output_file), sr=None)
                output_audio = (sr, (y * 32767).astype(np.int16))
            
            return "\n".join(status_updates), output_audio, str(output_file)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return error_msg, None, None
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="🎵 Phonklean GUI - Audio Restoration Tool", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🎵 Phonklean GUI
            ### AI-Powered Phonk Audio Restoration & Enhancement
            
            **What it does:**
            - 🎼 Reconstructs distorted audio using htdemucs transformer
            - 🔊 Reduces clipping and saturation artifacts
            - 🥁 Removes drum bleed and gating artifacts
            - 🎤 Applies smart autotune (optional)
            - 🎛️ Enhances drums, bass, and other instruments
            
            **Upload an audio file and adjust the parameters below!**
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Input")
                    input_file = gr.File(
                        label="Upload Audio File",
                        file_types=["audio"],
                        type="filepath"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Core Options")
                    no_vocals = gr.Checkbox(
                        label="Skip Vocal Processing",
                        value=False,
                        info="Check to process only instruments (drums, bass, other)"
                    )
                    vf_mode = gr.Slider(
                        minimum=0,
                        maximum=2,
                        step=1,
                        value=0,
                        label="VoiceFixer Mode",
                        info="0: Default | 1: Extra pre-processing | 2: Train mode"
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 Autotune Options")
                    autotune_vocals = gr.Checkbox(
                        label="Enable Autotune",
                        value=False,
                        info="Apply pitch correction to vocals"
                    )
                    autotune_key = gr.Dropdown(
                        choices=["auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
                        value="auto",
                        label="Root Key",
                        info="auto = detect automatically"
                    )
                    autotune_scale = gr.Radio(
                        choices=["chromatic", "major", "minor"],
                        value="chromatic",
                        label="Scale",
                        info="Which notes to snap to"
                    )
                    autotune_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                        label="Autotune Strength",
                        info="0 = no effect | 1 = full correction"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🥁 Drum Bleed Removal (HPSS)")
                    bleed_removal = gr.Checkbox(
                        label="Enable Bleed Removal",
                        value=False,
                        info="Remove over-gated drum artifacts from vocals"
                    )
                    bleed_margin = gr.Slider(
                        minimum=0.5,
                        maximum=4.0,
                        step=0.1,
                        value=2.0,
                        label="HPSS Margin (dB)",
                        info="Higher = stricter separation"
                    )
                    bleed_blend = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.3,
                        label="Percussive Blend",
                        info="0 = remove all drums | 1 = keep all drums"
                    )
            
            with gr.Row():
                process_btn = gr.Button("🚀 Process Audio", size="lg", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    status_output = gr.Textbox(
                        label="📋 Processing Status",
                        lines=15,
                        interactive=False,
                        value="Ready to process. Upload an audio file and click 'Process Audio'."
                    )
            
            with gr.Row():
                with gr.Column():
                    audio_output = gr.Audio(
                        label="🎵 Processed Output",
                        type="numpy"
                    )
                with gr.Column():
                    file_path_output = gr.Textbox(
                        label="📁 Output File Path",
                        interactive=False
                    )
            
            gr.Markdown("""
            ---
            ### 📚 Parameter Guide
            
            **Core Options:**
            - **Skip Vocal Processing**: Disable all vocal restoration; only enhance drums/bass
            - **VoiceFixer Mode**: 
              - Mode 0 (Default): Standard restoration
              - Mode 1: Extra preprocessing + high-freq cut
              - Mode 2: Training mode
            
            **Autotune:**
            - **Enable Autotune**: Apply pitch correction to vocals only
            - **Root Key**: Musical key to snap pitches to (auto-detect recommended)
            - **Scale**: Chromatic (all notes) | Major | Minor
            - **Strength**: How aggressively to correct pitches
            
            **Drum Bleed Removal:**
            - Uses harmonic-percussive source separation (HPSS)
            - **Margin**: Higher values = more aggressive separation
            - **Blend**: 0 removes all drums, 0.3 keeps 30% of transients (recommended)
            """)
            
            # Connect the button click to processing
            process_btn.click(
                fn=self.process_audio,
                inputs=[
                    input_file, no_vocals, vf_mode, autotune_vocals,
                    autotune_key, autotune_scale, autotune_strength,
                    bleed_removal, bleed_margin, bleed_blend
                ],
                outputs=[status_output, audio_output, file_path_output]
            )
        
        return demo


if __name__ == "__main__":
    gui = PhonkCleanerGUI()
    demo = gui.create_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
