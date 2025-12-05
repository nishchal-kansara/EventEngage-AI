# Import Necessary Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Streamlit Page Config
st.set_page_config(page_title="EventEngage AI", layout="wide")

# Load Dataset
@st.cache_resource
def load_dataset():
    df = pd.read_csv("dataset/ybai_cleaned.csv")
    return df

# Feature Columns
regression = [ 'Age','Years_of_Experience','Ticket_Price','Engagement_Score', 'Gender','Location_City','Location_Country','Event_Type', 'Attendance_Status','Session_Attended','Interests','Professional_Industry', 'Preferred_Event_Type','Payment_Method','Registration_Source', 'Attendance_History','Event_Frequency','Follow_Up_Status' ]
classification = [ 'Age', 'Years_of_Experience', 'Ticket_Price', 'Engagement_Score', 'Gender', 'Location_City', 'Location_Country', 'Event_Type', 'Session_Attended', 'Interests', 'Professional_Industry', 'Preferred_Event_Type', 'Payment_Method', 'Registration_Source', 'Attendance_History', 'Event_Frequency', 'Follow_Up_Status', 'Feedback_Rating' ]

num_regression = ['Age','Years_of_Experience','Ticket_Price','Engagement_Score']
num_classification = ['Age','Years_of_Experience','Feedback_Rating','Ticket_Price','Engagement_Score']

# Load Models and Artifacts
@st.cache_resource
def load_artifacts():
    rf_reg = joblib.load("models/rf_reg.pkl")
    rf_clf = joblib.load("models/rf_clf.pkl")
    scaler_reg = joblib.load("models/scaler_reg.pkl")
    scaler_clf = joblib.load("models/scaler_clf.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    return rf_reg, rf_clf, scaler_reg, scaler_clf, encoders

# Load Data and Models
with st.spinner("Application is Loading..."):
    df_full = load_dataset.__wrapped__()
    rf_reg, rf_clf, scaler_reg, scaler_clf, encoders = load_artifacts.__wrapped__()

st.success("Loaded Successfully!")

# Encoding Function
def encode_value(encoder, value):
    try:
        return encoder.transform([value])[0]
    except:
        return encoder.transform([encoder.classes_[0]])[0]

# Preprocessing Function
def preprocess(df, feature_order, numeric_cols, scaler):
    df = df.copy()

    # Encode Categorical
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: encode_value(encoder, x))

    # Scale Numeric
    num_present = [c for c in numeric_cols if c in df.columns]
    df[num_present] = scaler.transform(df[num_present])

    # Correct Column Order
    df = df.reindex(columns=feature_order).fillna(0)

    return df

# Sorting Attendance History
def sort_events(values):
    return sorted(values, key=lambda x: int(x.split(" ")[1]))

if "Attendance_History" in encoders:
    encoders["Attendance_History"].classes_ = np.array(sort_events(list(encoders["Attendance_History"].classes_)))

# Streamlit User Interface
st.title("EventEngage AI")
st.write("Predict Attendee Behavior & Experience")

tab_reg, tab_clf, tab_charts = st.tabs(
    ["Feedback Rating Prediction", "Attendance Status Prediction", "Interactive Charts"]
)

# Regression Tab
with tab_reg:
    st.subheader("Predict Feedback Rating")

    with st.form("regression_form"):
        c1, c2, c3 = st.columns(3)
        inputs = {}

        for idx, col in enumerate(regression):
            col_block = [c1, c2, c3][idx % 3]

            # Numeric Inputs
            if col in num_regression:
                if col == "Age":
                    inputs[col] = col_block.number_input(col, min_value=1, max_value=100, value=25)
                elif col == "Years_of_Experience":
                    inputs[col] = col_block.number_input(col, min_value=0, max_value=100, value=1)
                else:
                    inputs[col] = col_block.number_input(col, value=0.0)

            # Categorical Inputs
            elif col in encoders:
                vals = list(encoders[col].classes_)
                inputs[col] = col_block.selectbox(col, vals)

            else:
                inputs[col] = col_block.text_input(col)

        predict_btn = st.form_submit_button("Predict Feedback")

    if predict_btn:
        df = pd.DataFrame([inputs])
        df_p = preprocess(df, regression, num_regression, scaler_reg)
        pred = rf_reg.predict(df_p)[0]
        pred = max(0, min(5, pred))
        st.success(f"Predicted Feedback Rating: {round(pred, 2)}")

# Classification Tab
with tab_clf:
    st.subheader("Predict Attendance Status")

    with st.form("classification_form"):
        c1, c2, c3 = st.columns(3)
        inputs = {}

        for idx, col in enumerate(classification):
            col_block = [c1, c2, c3][idx % 3]

            if col in num_classification:
                if col == "Age":
                    inputs[col] = col_block.number_input(col, min_value=1, max_value=100, value=25)
                elif col == "Years_of_Experience":
                    inputs[col] = col_block.number_input(col, min_value=0, max_value=100, value=1)
                else:
                    inputs[col] = col_block.number_input(col, value=0.0)

            elif col in encoders:
                vals = list(encoders[col].classes_)
                inputs[col] = col_block.selectbox(col, vals)

            else:
                inputs[col] = col_block.text_input(col)

        predict_btn2 = st.form_submit_button("Predict Attendance")

    if predict_btn2:
        df = pd.DataFrame([inputs])
        df_p = preprocess(df, classification, num_classification, scaler_clf)
        
        # Predict encoded class
        pred_encoded = rf_clf.predict(df_p)[0]

        # Decode back to original label
        if "Attendance_Status" in encoders:
            pred_decoded = encoders["Attendance_Status"].inverse_transform([pred_encoded])[0]
        else:
            pred_decoded = pred_encoded  # fallback

        st.success(f"Predicted Attendance Status: {pred_decoded}")

# Data Visualization Tab
with tab_charts:
    st.subheader("Visual Insights")

    chart_type = st.selectbox(
        "Select Type:",
        ["Registration", "Event-Session Attendance", "Interested Domains", "Event Attendance"]
    )

    if chart_type == "Registration":
        col = st.selectbox("Select Column:", df_full.columns)
        fig = px.bar(
            df_full[col].value_counts().reset_index().rename(columns={"index": col, col: "Count"}),
            x=col,
            y="Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    if chart_type == "Event-Session Attendance":
        if "Session_Attended" in df_full.columns:
            df_counts = df_full['Session_Attended'].value_counts().reset_index(name='count')
            df_counts = df_counts.rename(columns={'index': 'Session_Attended'})

            fig = px.bar(df_counts, x="Session_Attended", y="count")
            st.plotly_chart(fig, use_container_width=True)

    if chart_type == "Interested Domains":
        if "Interests" in df_full.columns:
            df_counts = df_full['Interests'].value_counts().reset_index(name='count')
            df_counts = df_counts.rename(columns={'index': 'Interests'})

            fig = px.bar(df_counts, x="Interests", y="count")
            st.plotly_chart(fig, use_container_width=True)

    if chart_type == "Event Attendance":
        if "Attendance_History" in df_full.columns:
            df_counts = df_full['Attendance_History'].value_counts().reset_index(name='count')
            df_counts = df_counts.rename(columns={'index': 'Attendance_History'})

            fig = px.bar(df_counts, x="Attendance_History", y="count")
            st.plotly_chart(fig, use_container_width=True)

st.markdown(""" <hr style='margin-top:50px;'> <div style='text-align:center; font-size: 14px; color: gray;'> EventEngage AI Built with ❤️ using Streamlit </div> """, unsafe_allow_html=True)