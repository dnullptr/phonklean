import torch
import torchaudio
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# --- הגדרות ---
# נבדוק אם יש כרטיס מסך זמין (חובה לביצועים סבירים)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

if DEVICE == "cpu":
    print("⚠️ Warning: Running on CPU will be extremely slow for music generation.")

# המודל שאנחנו רוצים. 'melody' הוא הטוב ביותר למעקב אחרי מבנה קיים.
# בפעם הראשונה זה יוריד את המודל (כ-3GB).
MODEL_NAME = 'facebook/musicgen-melody'

def load_model():
    """טוען את מודל MusicGen לזיכרון"""
    print(f"⏳ Loading model: {MODEL_NAME}...")
    model = MusicGen.get_pretrained(MODEL_NAME, device=DEVICE)
    return model

def regenerate_drums(model, input_audio_path, output_filename, prompt, duration=None):
    """
    לוקח קובץ תופים מלוכלך ומייצר אותו מחדש נקי בעזרת AI.
    """
    if not os.path.exists(input_audio_path):
        print(f"❌ Error: Input file not found at {input_audio_path}")
        return

    print(f"🎧 Processing input: {input_audio_path}")
    
    # טעינת האודיו המלוכלך
    # אנחנו טוענים אותו כ-Tensor כדי שהמודל יוכל לקרוא אותו
    melody_waveform, sr = torchaudio.load(input_audio_path)
    
    # אם לא הוגדר משך, ניקח את אורך הקובץ המקורי (בשיניות)
    if duration is None:
        duration = melody_waveform.shape[1] / sr
    
    print(f"⏱️ Target Duration: {duration:.2f} seconds")

    # הגדרות יצירה
    # top_k/top_p משפיעים על היצירתיות. הערכים כאן הם סטנדרטיים לאיכות טובה.
    model.set_generation_params(
        duration=duration,
        top_k=250, 
        top_p=0.0, 
        temperature=1.0
    )

    # הכנת האודיו לכניסה למודל (הוספת מימד Batch)
    melody_waveform = melody_waveform.unsqueeze(0).to(DEVICE)

    print(f"🤖 Generating based on prompt: '{prompt}'...")

    # --- הקסם קורה כאן ---
    # generate_with_chroma מכריח את המודל להיצמד למקצב ולמבנה של קובץ הקלט
    output = model.generate_with_chroma(
        descriptions=[prompt],     # הטקסט שמנחה את הסאונד החדש
        melody_wavs=melody_waveform, # האודיו שמנחה את המקצב
        melody_sample_rate=sr,
        progress=True
    )

    # שמירת התוצאה
    # התוצאה היא Tensor [Batch, Channels, Time], צריך להוריד את ה-Batch כדי לשמור.
    output_waveform = output[0].cpu()
    
    # שמירה לקובץ WAV באיכות גבוהה (הספרייה מוסיפה .wav אוטומטית)
    output_path = os.path.join("output_regenerated", output_filename)
    os.makedirs("output_regenerated", exist_ok=True)
    
    audio_write(output_path, output_waveform, model.sample_rate, strategy="loudness", loudness_headroom_db=14)
    print(f"✨ Successfully saved regenerated track to: {output_path}.wav")


# --- אזור הרצה לבדיקה ---
if __name__ == "__main__":
    print(torch.cuda.is_available())
    # # 1. טען את המודל פעם אחת
    # musicgen_model = load_model()

    # # === הגדרות משתמש ===
    
    # # נתיב לקובץ התופים המלוכלך (שהפרדת ב-Demucs)
    # # שנה את זה לנתיב אמיתי במחשב שלך!
    # INPUT_DIRTY_DRUMS = "htdemucs_ft/Montagem_Xonada/drums.mp3" 
    
    # # שם הקובץ החדש שייווצר
    # OUTPUT_NAME = "htym_drums_CLEAN_AI"

    # # הפרומפט: זה הדבר הכי חשוב.
    # # תאר ל-AI בדיוק איך אתה רוצה שהתופים יישמעו.
    # PROMPT = "A high quality, clean, punchy Memphis Phonk drum loop. Crisp snare, deep kick drum, sharp hi-hats. No distortion, high fidelity sound."

    # # === הרצה ===
    # # נסה לייצר 10 שניות ראשונות לבדיקה
    # regenerate_drums(
    #     model=musicgen_model,
    #     input_audio_path=INPUT_DIRTY_DRUMS,
    #     output_filename=OUTPUT_NAME,
    #     prompt=PROMPT,
    #     duration=15 # אם תשים None זה יעשה את כל האורך
    # )