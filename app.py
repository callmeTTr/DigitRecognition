import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
import predict
import process

K = 50

def load_data():
    x_train, y_train = process.load_mnist("data/", kind="train")
    x_test, y_test = process.load_mnist("data/", kind="t10k")

    train_flat, train_chunk, train_histogram = process.extract_features(x_train)
    test_flat, test_chunk, test_histogram = process.extract_features(x_test)

    combined_train_flat = process.combine(train_flat, y_train)
    combined_train_chunk = process.combine(train_chunk, y_train)
    combined_train_histogram = process.combine(train_histogram, y_train)
    return combined_train_flat, combined_train_chunk, combined_train_histogram

extract_methods = {
    0: "FLAT",
    1: "CHUNK",
    2: "HISTOGRAM"
}

st.title("Digit Recognition")

st.sidebar.title("Options") 
option = st.sidebar.radio("Choose an option:", ("Upload Image", "Draw"))


if option == "Upload Image":
    left, right = st.columns(2)

    with left:
        
        st.header('Upload a Digit Image')
        
      
        uploaded_file = st.file_uploader("Add image", type=["jpg", "jpeg", "png"])

    with right:
        if uploaded_file is not None:

            raw_image = Image.open(uploaded_file)

            image = raw_image.resize((28, 28), Image.NEAREST)

            st.image(image, caption="Received image")

            if st.button("Submit"):
                st.write('Result = Pé đức thư giãn')

if option == "Draw":
    left, right = st.columns(2)

    with left:
        st.header('Draw a Digit')

        canvas_result = st_canvas(
            fill_color='#000000',
            stroke_width = 25, 
            stroke_color = "#FFFFFF", 
            background_color = "#000000", 
            width = 280, 
            height = 280, 
            drawing_mode = "freedraw", 
            key = "canvas" 
        )

    with right:
        if canvas_result.image_data is not None:
        
            raw_image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            image = raw_image.resize((28, 28), Image.NEAREST)

            st.image(image, caption="Received image")
            if st.button("Submit"):
                #print(image)
                image_arr = np.array(image)
                image_arr = image_arr.mean(axis = 2)
                print(image_arr)
                combined_train_flat, combined_train_chunk, combined_train_histogram = load_data()
                #print(x)
                results = predict.predict_with_methods(image_arr, K, extract_methods, combined_train_flat, combined_train_chunk, combined_train_histogram)
                for method_name, answer in results:
                    st.write(f"{method_name}'s prediction: {answer}")
                #st.write('Result = Pé đức thư giãn')