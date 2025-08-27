import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
import io
import requests

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Universal Scanner & Identifier",
    page_icon="🔍",
    layout="wide"
)

# Initialize Gemini API
def configure_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set GEMINI_API_KEY in your .env file")
        st.stop()
    genai.configure(api_key=api_key)

configure_genai()

# Function to get Gemini response
def get_gemini_response(image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([image, prompt])
    return response.text

# Function to identify object
def identify_object(image):
    prompt = """
    Analyze this image and identify what it contains. 
    Provide a detailed response in the following format:
    
    **Object/Subject:** [What is in the image]
    **Type/Category:** [Category it belongs to - animal, plant, product, etc.]
    
    **Key Characteristics:**
    - [Characteristic 1]
    - [Characteristic 2]
    - [Characteristic 3]
    
    **Additional Information:**
    [Any relevant information about what's in the image]
    
    **Interesting Facts:**
    - [Fact 1]
    - [Fact 2]
    
    **Practical Applications/Uses:**
    - [Use 1]
    - [Use 2]
    
    If the image doesn't contain anything recognizable, please state that clearly.
    """
    return get_gemini_response(image, prompt)

# Function to get detailed facts
def get_detailed_facts(subject):
    prompt = f"""
    Provide comprehensive information about {subject} in a structured format.
    Include information about:
    - Origin/history
    - Key features
    - Varieties/types (if applicable)
    - Significance to humans
    - Interesting trivia
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Main app
def main():
    st.title("🔍 Universal Scanner & Identifier")
    st.markdown("Use your camera to scan any object, animal, or plant to identify it and learn more about it.")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("This app uses AI to identify objects, animals, plants, and more through your camera or uploaded images.")
        
        st.header("Instructions")
        st.write("1. Use the camera to scan an object or upload an image")
        st.write("2. The AI will identify what it sees")
        st.write("3. View detailed information about the subject")
        
        st.header("Detection Capabilities")
        st.write("Animals, Plants, Products, Landmarks, Food, and more!")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Camera Scanner", "Upload Image", "Sample Images"])
    
    with tab1:
        st.subheader("Real-time Camera Scanner")
        st.write("Allow camera access and point at an object to identify it.")
        
        # Use Streamlit's built-in camera input
        camera_img = st.camera_input("Take a picture", key="camera_scanner")
        
        if camera_img is not None:
            # Display the captured image
            img = Image.open(camera_img)
            st.image(img, caption="Captured Image", use_column_width=True)
            
            # Process the image
            with st.spinner("Analyzing image..."):
                try:
                    # Identify object
                    response = identify_object(img)
                    
                    # Display results
                    st.success("Analysis Complete!")
                    st.markdown(response)
                    
                    # Extract subject for more facts
                    if "**Object/Subject:**" in response:
                        subject_line = response.split("**Object/Subject:**")[1].split("\n")[0].strip()
                        subject = subject_line.split("[")[1].split("]")[0] if "[" in subject_line else subject_line
                        st.subheader(f"More About {subject}")
                        facts = get_detailed_facts(subject)
                        st.markdown(facts)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please try again.")
    
    with tab2:
        st.subheader("Upload an Image")
        # File upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process the image
            if st.button("Identify This Image", key="identify_btn"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Identify object
                        response = identify_object(image)
                        
                        # Display results
                        st.success("Analysis Complete!")
                        st.markdown(response)
                        
                        # Extract subject for more facts
                        if "**Object/Subject:**" in response:
                            subject_line = response.split("**Object/Subject:**")[1].split("\n")[0].strip()
                            subject = subject_line.split("[")[1].split("]")[0] if "[" in subject_line else subject_line
                            st.subheader(f"More About {subject}")
                            facts = get_detailed_facts(subject)
                            st.markdown(facts)
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.info("Please try again with a different image.")

    with tab3:
        st.subheader("Try with these sample images:")
        col1, col2, col3 = st.columns(3)
        
        sample_images = {
            "Cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg",
            "Dog": "https://imgs.search.brave.com/8UIgd2rGu-w5WNHs1LSAieexcDqKt4liuafSLSDQwHk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWFn/ZXMuZnJlZWltYWdl/cy5jb20vaW1hZ2Vz/L2xhcmdlLXByZXZp/ZXdzL2NlNy9oYXBw/eS1ibGFjay1kb2ct/MDQxMC01NzAxNTc5/LmpwZz9mbXQ",
            "Bird": "https://imgs.search.brave.com/Pgcb9_lcz5h2RJHmkh0swRhKkdKQsfqRGeYICMzK1qg/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvNDgy/NTMwMTE5L3Bob3Rv/L29wZXJhLWJpcmQt/MS5jcGc_cz02MTJ4/NjEyJnc9MCZrPTIw/JmM9Q2E1bi0wOEZO/OW9YZExrM1Vza2lx/ZmpnbXZiXzQ2RHU0/ZlJZQkRGR3UyUT0",
            "Flower": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/800px-Sunflower_sky_backdrop.jpg",
            "Landmark": "https://imgs.search.brave.com/avDkW6NioPcZxp6GjvwHNuA-RiqCg4Fqn2-B40xHsWg/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvMTM3/MzA0ODUwNi9waG90/by90aGUtYnJvb2ts/eW4tYnJpZGdlLWZy/ZWVkb20tdG93ZXIt/YW5kLWxvd2VyLW1h/bmhhdHRhbi5qcGc_/cz02MTJ4NjEyJnc9/MCZrPTIwJmM9N0RZ/NkUzZXhoeDRHUkhT/aEhqUWJVRTc2TkZt/MnQzTFU5aS1PMlRM/RVo5QT0",
            "Food": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Pizza_%281%29.jpg/800px-Pizza_%281%29.jpg"
        }
        
        with col1:
            st.image(sample_images["Cat"], caption="Sample Cat", use_column_width=True)
            if st.button("Identify Cat", key="cat_btn"):
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.get(sample_images["Cat"])
                        img = Image.open(io.BytesIO(response.content))
                        result = identify_object(img)
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Error loading sample image: {str(e)}")
        
        with col2:
            st.image(sample_images["Flower"], caption="Sample Flower", use_column_width=True)
            if st.button("Identify Flower", key="flower_btn"):
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.get(sample_images["Flower"])
                        img = Image.open(io.BytesIO(response.content))
                        result = identify_object(img)
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Error loading sample image: {str(e)}")
        
        with col3:
            st.image(sample_images["Landmark"], caption="Sample Landmark", use_column_width=True)
            if st.button("Identify Landmark", key="landmark_btn"):
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.get(sample_images["Landmark"])
                        img = Image.open(io.BytesIO(response.content))
                        result = identify_object(img)
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Error loading sample image: {str(e)}")

if __name__ == "__main__":
    main()
