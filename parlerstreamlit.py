import streamlit as st
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import scipy.io.wavfile
import tempfile
import os

# Set page config
st.set_page_config(page_title="TTS Demo", page_icon="🎙️", layout="wide")

@st.cache_resource
def load_models():
    repo_id = "parler-tts/parler-tts-mini-v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, padding_side="left")
    feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
    
    return model, tokenizer, feature_extractor, device

def generate_audio(model, tokenizer, feature_extractor, device, input_text, description):
    inputs = tokenizer([description], return_tensors="pt", padding=True).to(device)
    prompt = tokenizer([input_text], return_tensors="pt", padding=True).to(device)
    
    set_seed(0)
    generation = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_input_ids=prompt.input_ids,
        prompt_attention_mask=prompt.attention_mask,
        do_sample=True,
        return_dict_in_generate=True,
    )
    
    audio = generation.sequences[0, :generation.audios_length[0]]
    return audio, feature_extractor.sampling_rate

def main():
    st.title("🎙️ TTS Demo")
    st.markdown("Generating natural-sounding speech from text")
    
    # Load models
    with st.spinner("Loading models..."):
        model, tokenizer, feature_extractor, device = load_models()
    
    # Input section
    st.subheader("Input Text")
    input_text = st.text_area("Enter the text you want to convert to speech:", 
                              height=100,
                              placeholder="e.g., Hey, how are you doing?")
    
    # Voice description section
    st.subheader("Voice Description")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Speaker Gender", ["male", "female"])
        pitch = st.selectbox("Voice Pitch", ["high-pitched", "low-pitched", "normal"])
    
    with col2:
        speed = st.selectbox("Speech Speed", ["slow", "normal", "fast"])
        environment = st.selectbox("Environment", ["confined", "open space", "studio"])
    
    tone = st.selectbox("Speaking Tone", ["monotone", "expressive", "cheerful", "serious"])
    
    # Construct description
    description = f"A {gender} speaker with a {tone} and {pitch} voice is delivering their speech at a {speed} speed in a {environment}."
    
    st.markdown("### Generated Description")
    st.info(description)
    
    # Generate button
    if st.button("Generate Audio", type="primary"):
        if not input_text.strip():
            st.error("Please enter some text to convert to speech.")
            return
        
        with st.spinner("Generating audio..."):
            try:
                audio, sample_rate = generate_audio(model, tokenizer, feature_extractor, 
                                                   device, input_text, description)
                
                # Save audio to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    scipy.io.wavfile.write(tmp_file.name, rate=sample_rate, 
                                          data=audio.cpu().numpy().squeeze())
                
                # Display audio player
                st.subheader("Generated Audio")
                st.audio(tmp_file.name)
                
                # Provide download button
                with open(tmp_file.name, 'rb') as audio_file:
                    st.download_button(
                        label="Download Audio",
                        data=audio_file,
                        file_name="generated_audio.wav",
                        mime="audio/wav"
                    )
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
            except Exception as e:
                st.error(f"An error occurred during audio generation: {str(e)}")
    
    # Add some information about the model
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This demo uses the Parler TTS model to generate natural-sounding speech from text. 
    You can customize various aspects of the voice using the options above.
    
    - Model: parler-tts-mini-v1
    - Running on: {device}
    """.format(device=device))

if __name__ == "__main__":
    main()