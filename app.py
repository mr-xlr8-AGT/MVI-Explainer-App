# MVI Explainer App
import streamlit as st
import numpy as np
import shap
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from io import StringIO
import streamlit.components.v1 as components

# --- Configuration & Setup ---

# 1. Set up Streamlit Page Config for a modern look
st.set_page_config(
 page_title="MVI Classifier (Hackathon MVP)",
 layout="wide",
 initial_sidebar_state="collapsed"
)

# Set the title and description
st.title("🛡️ Misinformation Rhetoric Classifier")
st.markdown("---")

# --- Model Loading & Caching ---
# Caching resources ensures the heavy model loads only once,
# which is crucial for fast performance in a deployed Streamlit app.

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource
def load_model_and_pipeline():
  """Loads the pre-trained sentiment model and creates the Hugging Face pipeline."""
  # Use sentiment-analysis pipeline as a proxy for 'rhetoric classification'
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  
  # Create a pipeline for easy prediction
  pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)
  return pipe, model, tokenizer

@st.cache_resource
def load_explainer(pipe):
  """Initializes the SHAP Explainer using the Hugging Face pipeline."""
  # The 'Explainer' automatically selects the right masker for the model type (DistilBERT)
  explainer = shap.Explainer(pipe)
  return explainer

# Load resources
try:
  pipe, model, tokenizer = load_model_and_pipeline()
  explainer = load_explainer(pipe)
except Exception as e:
  st.error(f"Error loading models. Please check your dependencies (requirements.txt). Details: {e}")
  st.stop()


# --- Core Logic Functions ---

def calculate_mvi_score(label, confidence):
  """
  Simplified Misinformation Vulnerability Index (MVI) Score Calculation.
  
  We assume 'NEGATIVE' sentiment correlates with fear-mongering/crisis rhetoric,
  leading to higher vulnerability.
  """
  
  # Scale confidence to 0-100 score
  scaled_confidence = confidence * 100
  
  if label == "NEGATIVE":
      # High confidence in NEGATIVE = High Vulnerability (MVI Score)
      score = scaled_confidence
      caption = "High fear/crisis rhetoric detected."
      delta = f"+{score:.1f} MVI"
  else:
      # High confidence in POSITIVE = Low Vulnerability (MVI Score)
      # We invert the score: high POSITIVE confidence means a low MVI score
      score = 100 - scaled_confidence
      caption = "Low fear/crisis rhetoric detected."
      delta = f"-{100 - score:.1f} MVI"
      
  return int(round(score)), caption, delta


def get_shap_plot_html(text, explainer):
  """Generates the SHAP text plot and returns its HTML source."""
  
  # Generate SHAP values for the input text
  shap_values = explainer([text])
  
  # SHAP text plot function outputs HTML/JS directly
  # We capture this output into an IO stream
  with StringIO() as output:
      # Use target_names to explicitly map the output labels (e.g., NEGATIVE to an index)
      # Since the pipeline returns an array of explanations, we take the first element (index 0)
      shap.plots.text(shap_values[0], show=False, display=False, output=output)
      html_content = output.getvalue()
      
  return html_content


# --- Streamlit UI and Execution ---

st.markdown("""
<style>
.stTextArea label {
  font-size: 1.25rem !important;
  font-weight: 600 !important;
  color: #FF4B4B;
}
.stMetric > div {
  background-color: #f0f2f6;
  padding: 15px;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.shap-force-plot {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 10px;
}
</style>
""", unsafe_allow_html=True)


input_text = st.text_area(
  "Paste the piece of news or social media text to analyze rhetoric:",
  placeholder="Example: The horrifying truth is that without immediate, drastic action on this crisis, our economy will collapse next week. You must panic now.",
  height=200
)

# Use a clear button to trigger analysis
if st.button("Analyze Rhetoric & Calculate MVI", type="primary"):
  if not input_text:
      st.warning("Please enter some text for analysis.")
  else:
      with st.spinner("Analyzing rhetoric and generating XAI explanations..."):
          
          # 1. Prediction
          raw_prediction = pipe(input_text)[0]
          
          # Find the highest confidence prediction
          best_prediction = max(raw_prediction, key=lambda x: x['score'])
          
          label = best_prediction['label']
          confidence = best_prediction['score']
          
          # 2. MVI Calculation
          mvi_score, mvi_caption, mvi_delta = calculate_mvi_score(label, confidence)
          
          # 3. SHAP Explanation
          shap_html = get_shap_plot_html(input_text, explainer)
          
          # --- Results Display ---
          
          st.markdown("## Analysis Results")
          
          col1, col2 = st.columns(2)
          
          with col1:
              st.metric(
                  label="Rhetoric Classification",
                  value=label,
                  delta=f"{confidence*100:.2f}% Confidence"
              )
          
          with col2:
              # Use color based on the MVI score
              if mvi_score > 70:
                  score_color = "#FF4B4B" # Red for High Vulnerability
              elif mvi_score > 40:
                  score_color = "#FF8C00" # Orange for Medium
              else:
                  score_color = "#00BFFF" # Blue for Low
                  
              st.markdown(f"""
              <div style="
                  background-color: {score_color};
                  color: white; 
                  padding: 15px; 
                  border-radius: 10px; 
                  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                  height: 100%;
              ">
                  <p style="font-size:1rem; margin-bottom: 0px;">Misinformation Vulnerability Index (MVI)</p>
                  <h1 style="font-size:2.5rem; margin-top: 5px; margin-bottom: 5px;">{mvi_score}/100</h1>
                  <p style="font-size:0.8rem; margin-top: 0px;">{mvi_caption}</p>
              </div>
              """, unsafe_allow_html=True)
              
          st.markdown("---")
          
          # 4. Explainable AI Visualization (SHAP)
          st.markdown("## 🧠 Explainable AI Breakdown (SHAP)")
          st.markdown("""
          The colors below show which words contributed most to the model's prediction.
          **Red** indicates words that pushed the prediction towards the **Final Result** (High MVI / Negative).
          **Blue** indicates words that pushed the prediction **Away** from the Final Result (Low MVI / Positive).
          """)
          
          # Use Streamlit components to display the raw HTML generated by SHAP
          components.html(
              shap_html,
              height=300,
              scrolling=True
          )

st.sidebar.markdown(f"""
## Hackathon Project Info
**Time Crunch MVP:** This project was scoped for rapid deployment.
**Model:** DistilBERT (Finetuned for Sentiment Analysis).
**XAI Core:** SHAP Text Explainer.
**MVI Score:** A simple, hard-coded formula based on prediction confidence, not a trained index.
""")

