import streamlit as st
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

# Load the model
loaded_model = pickle.load(open('Wine_model.sav', 'rb'))

def check(input_data):
    array_input = np.array(input_data)
    reshaped_input = array_input.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_input)
    return prediction[0]

def main():
    # Apply custom CSS for background image
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://i.postimg.cc/13wPRjV1/443997.png");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 0.8;  /* Adjust the opacity as needed */
    }
    .css-1cpxqw2 {
        text-align: center;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: 2px solid #ff4b4b;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: white;
        color: #ff4b4b;
    }
    h1 {
        text-align: center;
        text-decoration: underline;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.markdown("<h1>🍷 Red Wine Quality Prediction 🍷</h1>", unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # Input fields in the first column
    with col1:
        fixed_acidity = st.number_input("**Fixed Acidity**")
        volatile_acidity = st.number_input("**Volatile Acidity**")
        citric_acid = st.number_input("**Citric Acid**")
        residual_sugar = st.number_input("**Residual Sugar**")
        chlorides = st.number_input("**Chlorides**")
        free_sulfur_dioxide = st.number_input("**Free Sulfur Dioxide**")

    # Input fields in the second column
    with col2:
        total_sulfur_dioxide = st.number_input("**Total Sulfur Dioxide**")
        density = st.number_input("**Density**")
        pH = st.number_input("**pH**")
        sulphates = st.number_input("**Sulphates**")
        alcohol = st.number_input("**Alcohol**")

    if st.button("**Click Here for Red Wine Prediction**"):
        pred_score = check([
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ])
        
        # Ensure pred_score is within the 3 to 8 range
        pred_score = max(3, min(8, pred_score))

        st.balloons()
        st.success(f"Wine Quality Test Result: {pred_score:.2f}")
        
        # Descriptive text based on the prediction score
        if pred_score <= 4:
            st.warning("**This wine is of poor quality.**")
        elif pred_score <= 6:
            st.info("**This wine is of average quality.**")
        else:
            st.success("**This wine is of good quality.**")

if __name__ == '__main__':
    main()
