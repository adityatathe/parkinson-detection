import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import os
import time

# Configure page settings
st.set_page_config(
    page_title="Parkinson's Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin: 1rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    .upload-card {
        background: linear-gradient(145deg, #f8f9ff 0%, #e8f2ff 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #4A90E2;
        text-align: center;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        border-color: #357ABD;
        background: linear-gradient(145deg, #f0f7ff 0%, #daeeff 100%);
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    
    .positive-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    
    .negative-result {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
        color: white;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffa726 0%, #ff7043 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Instructions styling */
    .instructions {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .instructions h3 {
        margin-top: 0;
        color: white;
    }
    
    .instructions ul {
        margin-bottom: 0;
    }
    
    /* Progress bar */
    .progress-container {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 0.25rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 0.5rem;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .info-card, .upload-card, .result-card {
            padding: 1rem;
        }
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Function to load the saved model and PCA object
@st.cache_resource
def load_model():
    """Load the trained model and PCA transformer with caching for better performance."""
    model_path = os.path.join('models', 'svm_model.pkl')
    pca_path = os.path.join('models', 'pca.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(pca_path):
        return None, None, "Model files not found. Please run train.py first."
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        return model, pca, "Models loaded successfully!"
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"

# Function to preprocess the uploaded image
def preprocess_image(image_file, target_size=(200, 200)):
    """Preprocess the uploaded image for model prediction."""
    try:
        # Convert the file to an OpenCV image format
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return None, "Could not decode the image. Please try a different file."

        # Resize the image to match training data size
        image = cv2.resize(image, target_size)

        # Apply Gaussian blur for noise reduction
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Canny Edge Detection
        edges = cv2.Canny(blurred_image, 100, 200)

        # Flatten and return
        features = edges.flatten()
        return features, "Image processed successfully!"
    
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def show_loading_animation():
    """Display a loading animation."""
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #667eea;'>Processing your image...</p>", unsafe_allow_html=True)
    return loading_placeholder

def display_result(prediction, confidence=None):
    """Display prediction results with appropriate styling."""
    if prediction == 1:
        st.markdown("""
        <div class="result-card positive-result">
            <h2>⚠️ Potential Signs Detected</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                The system has detected potential signs of Parkinson's disease in the uploaded drawing.
            </p>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>⚠️ Important Disclaimer:</strong><br>
                This is an automated screening tool and should NOT be considered a medical diagnosis. 
                Please consult with a qualified healthcare professional for proper evaluation and diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div class="result-card negative-result">
            <h2>✅ No Signs Detected</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                The system did not detect signs of Parkinson's disease in the uploaded drawing.
            </p>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>ℹ️ Important Note:</strong><br>
                This result does not guarantee absence of the condition. Regular medical check-ups 
                and consultation with healthcare professionals are always recommended.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.snow()

# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Parkinson's Disease Detection System</h1>
        <p>AI-powered analysis of handwritten drawings for early screening</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    svm_model, pca_model, load_status = load_model()
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Instructions card
        st.markdown("""
        <div class="instructions">
            <h3>📋 How to Use This System</h3>
            <ul>
                <li><strong>Step 1:</strong> Upload a clear image of a handwritten spiral or wave drawing</li>
                <li><strong>Step 2:</strong> Ensure the image is in PNG, JPG, or JPEG format</li>
                <li><strong>Step 3:</strong> Click the "Analyze Drawing" button to get results</li>
                <li><strong>Step 4:</strong> Review the results and consult a healthcare professional</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # File upload section
        st.markdown("""
        <div class="upload-card">
            <h3 style="color: #4A90E2; margin-top: 0;">📤 Upload Your Drawing</h3>
            <p style="color: #666; margin-bottom: 1rem;">
                Select a clear, high-quality image of your handwritten spiral or wave pattern
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "", 
            type=["png", "jpg", "jpeg"],
            help="Upload a clear image of a handwritten spiral or wave drawing"
        )

        # Display uploaded image
        if uploaded_file is not None:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            
            # Create columns for image display
            img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
            
            with img_col2:
                st.image(uploaded_file, caption="📸 Uploaded Drawing", use_column_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.success("✅ Image uploaded successfully!")

            # Prediction section
            if st.button("🔍 Analyze Drawing", key="predict_button"):
                if not svm_model or not pca_model:
                    st.markdown("""
                    <div class="warning-card">
                        <h3>⚠️ Model Not Available</h3>
                        <p>The prediction models are not loaded. Please ensure train.py has been run and model files exist.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Show loading animation
                    loading_placeholder = show_loading_animation()
                    
                    # Process image
                    time.sleep(2)  # Simulate processing time
                    features, process_status = preprocess_image(uploaded_file)
                    
                    # Clear loading animation
                    loading_placeholder.empty()
                    
                    if features is None:
                        st.error(f"❌ {process_status}")
                    else:
                        try:
                            # Apply PCA transformation
                            features_pca = pca_model.transform(features.reshape(1, -1))
                            
                            # Make prediction
                            prediction = svm_model.predict(features_pca)
                            
                            # Display results
                            st.markdown("---")
                            display_result(prediction[0])
                            
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
                            st.info("💡 This might be due to image dimension mismatch. Please try a different image.")

    with col2:
        # Information sidebar
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea;">ℹ️ About This System</h3>
            <p style="font-size: 0.9rem; line-height: 1.6;">
                This AI system analyzes handwritten drawings to identify potential signs of Parkinson's disease. 
                It uses advanced machine learning techniques including edge detection and pattern recognition.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea;">🎯 What We Analyze</h3>
            <ul style="font-size: 0.9rem; line-height: 1.6;">
                <li>Drawing smoothness and tremor patterns</li>
                <li>Line consistency and control</li>
                <li>Spatial organization</li>
                <li>Motor control indicators</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea;">⚠️ Important Notes</h3>
            <ul style="font-size: 0.9rem; line-height: 1.6;">
                <li>This is a screening tool, not a diagnostic device</li>
                <li>Results should be discussed with a healthcare provider</li>
                <li>Early detection can lead to better treatment outcomes</li>
                <li>Multiple factors influence drawing patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p style="margin: 0;">
            🏥 This tool is for educational and screening purposes only. Always consult healthcare professionals for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()