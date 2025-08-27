import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import numpy as np
import cv2
from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import time

# Configure page
st.set_page_config(
    page_title="Universal Object Identifier",
    page_icon="🔍",
    layout="wide"
)

# Initialize Gemini API
def configure_genai():
    # Try to get API key from environment variable or Streamlit secrets
    api_key = None
    try:
        # Try to get from Streamlit secrets (if deployed on Streamlit Cloud)
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            api_key = st.secrets['GEMINI_API_KEY']
        # Try to get from environment variable
        elif 'GEMINI_API_KEY' in os.environ:
            api_key = os.environ['GEMINI_API_KEY']
        # Allow user to input API key
        else:
            api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")
    except:
        pass
    
    if not api_key:
        st.warning("Gemini API key not found. Some features may be limited.")
        return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return False

genai_available = configure_genai()

# Load TensorFlow model (using MobileNet for demonstration)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        return model
    except Exception as e:
        st.warning(f"Could not load TensorFlow model: {str(e)}")
        return None

tf_model = load_model()

# Function to get Gemini response
def get_gemini_response(image, prompt):
    if not genai_available:
        return "Gemini API not configured. Please check your API key."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([image, prompt])
        return response.text
    except Exception as e:
        return f"Error with Gemini API: {str(e)}"

# Function to identify objects with Gemini
def identify_object_gemini(image):
    prompt = """
    Analyze this image and identify the primary object(s) in view. 
    This could be animals, plants, birds, insects, vehicles, furniture, or any other recognizable object.
    
    For each primary object, provide a detailed response in the following format:
    
    **Object Type:** [Category - e.g., Animal, Plant, Vehicle, etc.]
    **Specific Identification:** [Specific name if identifiable - e.g., German Shepherd, Rose, Toyota Camry]
    
    **Key Characteristics:**
    - [Characteristic 1]
    - [Characteristic 2]
    - [Characteristic 3]
    
    **Additional Information:**
    [Any other relevant information about this object]
    
    **Safety Assessment (if applicable):**
    - [Danger level: Low/Medium/High if applicable]
    - [Potential risks if any]
    - [Safety precautions if needed]
    
    If the image contains multiple objects, identify the most prominent ones.
    If the image doesn't contain any recognizable objects, please state that clearly.
    """
    return get_gemini_response(image, prompt)

# Function to identify objects with TensorFlow
def identify_object_tensorflow(image):
    if tf_model is None:
        return "TensorFlow model not available."
    
    try:
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make prediction
        predictions = tf_model.predict(img_array)
        results = imagenet_utils.decode_predictions(predictions.numpy())
        
        # Format results
        response = "**TensorFlow Object Detection Results:**\n\n"
        for i, (imagenet_id, label, score) in enumerate(results[0]):
            response += f"{i+1}. {label.replace('_', ' ').title()} ({score*100:.2f}% confidence)\n"
        
        return response
    except Exception as e:
        return f"Error with TensorFlow detection: {str(e)}"

# Function to detect edges with OpenCV
def detect_edges_opencv(image):
    try:
        # Convert PIL image to OpenCV format
        open_cv_image = np.array(image.convert('RGB'))
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
        
        # Convert to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)
        
        # Count edges (simple way to get some info)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_ratio = edge_pixels / total_pixels
        
        response = "**OpenCV Edge Detection Analysis:**\n\n"
        response += f"- Detected {edge_pixels} edge pixels ({edge_ratio*100:.2f}% of image)\n"
        response += "- High edge density suggests detailed or complex objects\n"
        response += "- Low edge density suggests smooth surfaces or uniform areas\n"
        
        return response, edges
    except Exception as e:
        return f"Error with OpenCV detection: {str(e)}", None

# Function to detect dominant colors
def detect_dominant_colors(image, num_colors=5):
    try:
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Reshape the image to be a list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors and their percentages
        counts = Counter(kmeans.labels_)
        total_pixels = len(pixels)
        
        # Format response
        response = "**Dominant Colors Analysis:**\n\n"
        for i, (color_idx, count) in enumerate(counts.most_common(num_colors)):
            percentage = (count / total_pixels) * 100
            color = kmeans.cluster_centers_[color_idx].astype(int)
            response += f"{i+1}. RGB({color[0]}, {color[1]}, {color[2]}) - {percentage:.2f}%\n"
        
        return response
    except Exception as e:
        return f"Error in color detection: {str(e)}"

# Function to get detailed facts
def get_detailed_facts(object_type, specific_name):
    if not genai_available:
        return "Gemini API not available for detailed information."
    
    prompt = f"""
    Provide comprehensive information about {specific_name if specific_name else object_type} in a structured format.
    Include information about:
    
    **Basic Information:**
    - Classification/Category
    - Origin/Natural habitat (if applicable)
    - Typical size/dimensions
    
    **Characteristics:**
    - Key identifying features
    - Varieties/types (if applicable)
    
    **Interesting Facts:**
    - Unique attributes or behaviors
    - Historical or cultural significance (if any)
    - Conservation status (if applicable)
    
    **Practical Information (if applicable):**
    - Care requirements (for living things)
    - Usage/functionality (for objects)
    - Maintenance considerations
    
    Format the response in clear markdown sections.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting detailed information: {str(e)}"

# Function to handle camera input
def capture_image():
    picture = st.camera_input("Take a picture")
    if picture:
        return Image.open(picture)
    return None

# Main app
def main():
    st.title("🔍 Universal Object Identification System")
    st.markdown("Upload an image or use your camera to identify objects, animals, plants, and more.")
    
    # Sidebar
    with st.sidebar:
        st.header("Detection Options")
        
        detection_mode = st.selectbox(
            "Choose detection method:",
            ("Gemini AI (Detailed Analysis)", "TensorFlow (Object Recognition)", "OpenCV (Edge Detection)", "All Methods")
        )
        
        st.header("About")
        st.write("This tool uses multiple AI approaches to identify various objects from images.")
        
        st.header("Instructions")
        st.write("1. Choose detection method")
        st.write("2. Upload image or use camera")
        st.write("3. View analysis results")
        st.write("4. Explore detailed information")
        
        # Show system status
        st.header("System Status")
        if genai_available:
            st.success("✓ Gemini API Connected")
        else:
            st.error("✗ Gemini API Not Available")
            
        if tf_model is not None:
            st.success("✓ TensorFlow Model Loaded")
        else:
            st.warning("⚠ TensorFlow Model Not Available")
            
        st.success("✓ OpenCV Available")
    
    # Input selection
    input_method = st.radio("Choose input method:", ("Upload Image", "Use Camera"))
    
    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        image = capture_image()
    
    if image is not None:
        # Display the image
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
        
        # Process the image based on selected mode
        with st.spinner("Analyzing image..."):
            try:
                results = {}
                
                # Run selected detection methods
                if detection_mode in ["Gemini AI (Detailed Analysis)", "All Methods"] and genai_available:
                    with st.expander("Gemini AI Analysis", expanded=True):
                        gemini_result = identify_object_gemini(image)
                        st.markdown(gemini_result)
                        results["gemini"] = gemini_result
                
                if detection_mode in ["TensorFlow (Object Recognition)", "All Methods"] and tf_model is not None:
                    with st.expander("TensorFlow Object Recognition", expanded=detection_mode != "All Methods"):
                        tf_result = identify_object_tensorflow(image)
                        st.markdown(tf_result)
                        results["tensorflow"] = tf_result
                
                if detection_mode in ["OpenCV (Edge Detection)", "All Methods"]:
                    with st.expander("OpenCV Analysis", expanded=detection_mode != "All Methods"):
                        edge_result, edge_image = detect_edges_opencv(image)
                        st.markdown(edge_result)
                        
                        # Show edge detection visualization
                        if edge_image is not None:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                            ax1.imshow(np.array(image))
                            ax1.set_title('Original Image')
                            ax1.axis('off')
                            
                            ax2.imshow(edge_image, cmap='gray')
                            ax2.set_title('Edge Detection')
                            ax2.axis('off')
                            
                            st.pyplot(fig)
                        
                        # Color analysis
                        color_result = detect_dominant_colors(image)
                        st.markdown(color_result)
                        results["opencv"] = edge_result
                        results["color"] = color_result
                
                # Detailed information tab
                if genai_available and ("gemini" in results or "tensorflow" in results):
                    with st.expander("Detailed Information", expanded=True):
                        # Try to extract object info from Gemini results
                        object_info = ""
                        specific_name = ""
                        
                        if "gemini" in results and "**Object Type:**" in results["gemini"]:
                            object_info = results["gemini"]
                            if "**Specific Identification:**" in results["gemini"]:
                                specific_name = results["gemini"].split("**Specific Identification:**")[1].split("**Key Characteristics:**")[0].strip()
                        
                        # If no Gemini info, try to get from TensorFlow
                        if not object_info and "tensorflow" in results:
                            # Get the top prediction from TensorFlow
                            lines = results["tensorflow"].split('\n')
                            if len(lines) > 2:
                                specific_name = lines[1].split('. ')[1].split(' (')[0]
                                object_info = f"Object detected: {specific_name}"
                        
                        if object_info:
                            st.subheader("Comprehensive Details")
                            facts = get_detailed_facts("object", specific_name)
                            st.markdown(facts)
                        else:
                            st.info("Detailed information not available for this object.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again with a different image.")

    else:
        # Show sample images and capabilities
        st.subheader("Try detecting various objects:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        sample_images = [
            ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg", "Animals"),
            ("https://cdn.pixabay.com/photo/2018/02/09/21/46/rose-3142529_1280.jpg", "Plants & Flowers"),
            ("https://cdn.pixabay.com/photo/2015/01/19/13/51/car-604019_1280.jpg", "Vehicles"),
            ("https://cdn.pixabay.com/photo/2017/07/31/20/53/books-2562355_1280.jpg", "Everyday Objects")
        ]
        
        for col, (url, caption) in zip([col1, col2, col3, col4], sample_images):
            with col:
                st.image(url, caption=caption, use_column_width=True)
        
        # Add information about detection methods
        st.info("""
        **Available Detection Methods:**
        - **Gemini AI**: Detailed analysis and information about objects
        - **TensorFlow**: Fast object recognition with confidence scores
        - **OpenCV**: Edge detection and color analysis
        - **All Methods**: Combine all approaches for comprehensive analysis
        """)

if __name__ == "__main__":
    main()