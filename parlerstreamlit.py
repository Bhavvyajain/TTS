import streamlit as st
import torch
import logging
import sys
import os
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import scipy.io.wavfile
import tempfile
from pathlib import Path
import gc
import warnings
from contextlib import contextmanager

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Environment variables for better stability
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Create cache directory if it doesn't exist
CACHE_DIR = Path(tempfile.gettempdir()) / "tts-cache"
CACHE_DIR.mkdir(exist_ok=True)

# Configuration
CONFIG = {
    "model_id": "parler-tts/parler-tts-mini-v1",
    "max_text_length": 500,
    "sampling_rate": 24000,
}

@contextmanager
def torch_gc_context():
    """Context manager for proper PyTorch memory management."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def init_session_state():
    """Initialize session state variables."""
    if 'model_loaded' not in st.session_state:
        st.session_state['model_loaded'] = False
    if 'device' not in st.session_state:
        st.session_state['device'] = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    """Load TTS models with proper error handling and caching."""
    try:
        logger.info(f"Loading models on device: {st.session_state['device']}")
        
        with torch_gc_context():
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                CONFIG["model_id"],
                cache_dir=CACHE_DIR,
                local_files_only=False
            ).to(st.session_state['device'])
            
            tokenizer = AutoTokenizer.from_pretrained(
                CONFIG["model_id"],
                cache_dir=CACHE_DIR,
                padding_side="left",
                local_files_only=False
            )
            
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                CONFIG["model_id"],
                cache_dir=CACHE_DIR,
                local_files_only=False
            )
        
        st.session_state['model_loaded'] = True
        return model, tokenizer, feature_extractor
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        st.session_state['model_loaded'] = False
        raise RuntimeError(f"Failed to load models: {str(e)}")

def generate_audio(model, tokenizer, feature_extractor, input_text, description):
    """Generate audio from text with error handling."""
    try:
        # Input validation
        if len(input_text) > CONFIG["max_text_length"]:
            raise ValueError(f"Input text exceeds maximum length of {CONFIG['max_text_length']} characters")
        
        with torch_gc_context():
            # Prepare inputs
            inputs = tokenizer([description], return_tensors="pt", padding=True)
            prompt = tokenizer([input_text], return_tensors="pt", padding=True)
            
            # Move inputs to device
            inputs = {k: v.to(st.session_state['device']) for k, v in inputs.items()}
            prompt = {k: v.to(st.session_state['device']) for k, v in prompt.items()}
            
            # Generate audio
            set_seed(42)  # For reproducibility
            generation = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                prompt_input_ids=prompt["input_ids"],
                prompt_attention_mask=prompt["attention_mask"],
                do_sample=True,
                return_dict_in_generate=True,
            )
            
            audio = generation.sequences[0, :generation.audios_length[0]]
            return audio.cpu().numpy(), CONFIG["sampling_rate"]
    
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise

def main():
    try:
        # Initialize session state
        init_session_state()
        
        # Page config
        st.set_page_config(
            page_title="TTS Demo",
            page_icon="🎙️",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Title and description
        st.title("🎙️ Text-to-Speech Demo")
        st.markdown("Generate natural-sounding speech from text with customizable voice settings.")
        
        # Display device information
        st.sidebar.info(f"Running on: {st.session_state['device'].upper()}")
        
        # Load models
        with st.spinner("Loading models... This may take a moment."):
            try:
                model, tokenizer, feature_extractor = load_models()
            except Exception as e:
                st.error(f"Failed to load models: {str(e)}")
                st.stop()
        
        # Input section
        st.subheader("Input Text")
        input_text = st.text_area(
            "Enter text to convert to speech:",
            max_chars=CONFIG["max_text_length"],
            height=100,
            placeholder="Enter the text you want to convert to speech..."
        )
        
        # Voice customization
        st.subheader("Voice Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Speaker Gender", ["male", "female"])
            pitch = st.selectbox("Voice Pitch", ["high-pitched", "low-pitched", "normal"])
            
        with col2:
            speed = st.selectbox("Speech Speed", ["slow", "normal", "fast"])
            tone = st.selectbox("Speaking Tone", ["monotone", "expressive", "cheerful", "serious"])
            
        environment = st.selectbox("Environment", ["studio", "confined", "open space"])
        
        # Generate description
        description = f"A {gender} speaker with a {tone} and {pitch} voice is delivering their speech at a {speed} speed in a {environment}."
        
        st.markdown("### Voice Description")
        st.info(description)
        
        # Generate button
        if st.button("Generate Audio", type="primary", disabled=not st.session_state['model_loaded']):
            if not input_text.strip():
                st.error("Please enter some text to convert to speech.")
                return
            
            with st.spinner("Generating audio..."):
                try:
                    # Generate audio
                    audio_data, sample_rate = generate_audio(
                        model, tokenizer, feature_extractor,
                        input_text, description
                    )
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        scipy.io.wavfile.write(
                            tmp_file.name,
                            rate=sample_rate,
                            data=audio_data
                        )
                        
                        # Display audio player
                        st.subheader("Generated Audio")
                        st.audio(tmp_file.name)
                        
                        # Download button
                        with open(tmp_file.name, 'rb') as audio_file:
                            st.download_button(
                                label="Download Audio",
                                data=audio_file,
                                file_name="generated_speech.wav",
                                mime="audio/wav"
                            )
                    
                    # Cleanup
                    os.unlink(tmp_file.name)
                    
                except Exception as e:
                    st.error(f"Error generating audio: {str(e)}")
                    logger.error(f"Generation error: {str(e)}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        ### About
        This demo uses the Parler TTS model to generate natural-sounding speech from text.
        The voice can be customized using various parameters above.
        
        - Model: parler-tts-mini-v1
        - Sampling Rate: 24kHz
        """)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        st.error("Failed to start the application. Please check the logs for details.")
