import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import io
import requests

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
        # Try to get from Streamlit secrets
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
        st.warning("Gemini API key not found. Please enter your API key in the sidebar to use all features.")
        return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return False

genai_available = configure_genai()

# Function to get Gemini response
def get_gemini_response(image, prompt):
    if not genai_available:
        return "Gemini API not configured. Please check your API key in the sidebar."
    
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

# Function to get image from URL
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")
        return None

# Function for simple image analysis (without OpenCV)
def simple_image_analysis(image):
    try:
        # Get basic image information
        width, height = image.size
        format = image.format
        mode = image.mode
        
        response = "**Image Analysis:**\n\n"
        response += f"- Dimensions: {width} x {height} pixels\n"
        response += f"- Format: {format if format else 'Unknown'}\n"
        response += f"- Color mode: {mode}\n"
        
        # Simple analysis based on image characteristics
        if width > height:
            response += "- Landscape orientation\n"
        elif height > width:
            response += "- Portrait orientation\n"
        else:
            response += "- Square format\n"
            
        return response
    except Exception as e:
        return f"Error in image analysis: {str(e)}"

# Main app
def main():
    st.title("🔍 Universal Object Identification System")
    st.markdown("Upload an image or use your camera to identify objects, animals, plants, and more.")
    
    # Sidebar
    with st.sidebar:
        st.header("API Configuration")
        
        # API key input (if not already set)
        if not genai_available:
            api_key = st.text_input("Enter your Gemini API Key:", type="password")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    st.success("API key configured successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error configuring API: {str(e)}")
        
        st.header("About")
        st.write("This tool uses Google's Gemini AI to identify various objects from images.")
        
        st.header("Instructions")
        st.write("1. Enter your Gemini API key (get it from Google AI Studio)")
        st.write("2. Upload image or use camera")
        st.write("3. View analysis results")
        st.write("4. Explore detailed information")
        
        # Show system status
        st.header("System Status")
        if genai_available:
            st.success("✓ Gemini API Connected")
        else:
            st.error("✗ Gemini API Not Configured")
    
    # Input selection
    input_method = st.radio("Choose input method:", ("Upload Image", "Use Camera", "Sample Images"))
    
    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    elif input_method == "Use Camera":
        image = capture_image()
    else:
        # Sample images
        sample_options = {
            "Cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg",
            "Rose": "https://cdn.pixabay.com/photo/2018/02/09/21/46/rose-3142529_1280.jpg",
            "Car": "https://cdn.pixabay.com/photo/2015/01/19/13/51/car-604019_1280.jpg",
            "Books": "https://cdn.pixabay.com/photo/2017/07/31/20/53/books-2562355_1280.jpg"
        }
        
        selected_sample = st.selectbox("Choose a sample image:", list(sample_options.keys()))
        if selected_sample:
            image = load_image_from_url(sample_options[selected_sample])
    
    if image is not None:
        # Display the image
        st.image(image, caption="Input Image", use_column_width=True)
        
        # Basic image analysis
        with st.expander("Basic Image Information", expanded=True):
            basic_info = simple_image_analysis(image)
            st.markdown(basic_info)
        
        # Process the image with Gemini
        if genai_available:
            with st.spinner("Analyzing image with Gemini AI..."):
                try:
                    # Identify object
                    gemini_result = identify_object_gemini(image)
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    # Create tabs for different information sections
                    tab1, tab2 = st.tabs(["Identification Results", "Detailed Information"])
                    
                    with tab1:
                        st.subheader("Gemini AI Analysis")
                        st.markdown(gemini_result)
                    
                    with tab2:
                        # Try to extract object info from Gemini results
                        object_info = ""
                        specific_name = ""
                        
                        if "**Object Type:**" in gemini_result and "**Specific Identification:**" in gemini_result:
                            object_type = gemini_result.split("**Object Type:**")[1].split("**Specific Identification:**")[0].strip()
                            specific_name = gemini_result.split("**Specific Identification:**")[1].split("**Key Characteristics:**")[0].strip()
                            
                            st.subheader(f"Comprehensive Details about {specific_name}")
                            facts = get_detailed_facts(object_type, specific_name)
                            st.markdown(facts)
                        else:
                            st.info("Detailed information not available for this object.")
                
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.info("Please try again with a different image.")
        else:
            st.warning("Please configure your Gemini API key in the sidebar to enable object detection.")
    
    else:
        # Show instructions and capabilities
        st.info("""
        **How to use this app:**
        1. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/)
        2. Enter your API key in the sidebar
        3. Choose an input method (upload, camera, or sample images)
        4. View the analysis results
        
        **Detection Capabilities:**
        - Animals & Pets
        - Plants & Flowers
        - Vehicles & Machinery
        - Household Items
        - Food & Drinks
        - And much more!
        """)
        
        # Show sample images
        st.subheader("Sample Images You Can Try:")
        col1, col2, col3, col4 = st.columns(4)
        
        sample_images = [
            ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg", "Cat"),
            ("https://cdn.pixabay.com/photo/2018/02/09/21/46/rose-3142529_1280.jpg", "Rose"),
            ("https://cdn.pixabay.com/photo/2015/01/19/13/51/car-604019_1280.jpg", "Car"),
            ("https://cdn.pixabay.com/photo/2017/07/31/20/53/books-2562355_1280.jpg", "Books")
        ]
        
        for col, (url, caption) in zip([col1, col2, col3, col4], sample_images):
            with col:
                st.image(url, caption=caption, use_column_width=True)

if __name__ == "__main__":
    main()
