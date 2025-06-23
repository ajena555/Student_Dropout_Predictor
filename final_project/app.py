import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from edubot.engine import ask_edubot

from fpdf import FPDF
from datetime import datetime

def generate_pdf(input_data, prediction, probability=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.cell(200, 10, txt="Student Dropout Prediction Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    # Input data
    pdf.set_font("Arial", size=11)
    for key, value in input_data.items():
        safe_value = str(value).encode("latin-1", "ignore").decode("latin-1")
        pdf.cell(0, 10, f"{key}: {safe_value}", ln=True)

    pdf.ln(5)

    # Prediction result
    result = "LIKELY TO DROP OUT" if prediction == 1 else "NOT LIKELY TO DROP OUT"
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(0, 10, f"Prediction Result: {result}", ln=True)

    if probability is not None:
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 10, f"Model Confidence: {probability:.2%}", ln=True)

    # Save file
    file_path = "report.pdf"
    pdf.output(file_path)
    return file_path





# Load data


@st.cache_data
def load_data():
    df = pd.read_csv("reduced_student_dropout.csv")
    return df


df = load_data()

# Encode categorical variables for modeling
df_encoded = df.copy()
label_encoders = {}
for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# Target
target_col = 'Dropped_Out'  # Make sure this column exists

X = df_encoded.drop(columns=[target_col])
imp=X.columns

# Train model
import joblib

# Load trained model
model = joblib.load("random.pkl")


# --- Streamlit App ---
st.title("Student Dropout Analyzer")

st.sidebar.title("Navigation")
tabs = ["Home", "Visualization", "Prediction", "EduBot 🤖"]
choice = st.sidebar.radio("Go to", tabs)

# --- Home Tab ---
if choice == "Home":
    st.title("🎓 Student Dropout Prediction & Analysis")
    st.markdown("""
    This web application helps educational institutions identify students at risk of academic failure or dropout.

    **Features:**
    - Analyze factors influencing dropout.
    - Visualize dropout trends with respect to different variables.
    - Predict dropout risk for individual students using machine learning.

    📊 Empower timely intervention for better student outcomes.
    """)

# --- Visualization Tab ---
elif choice == "Visualization":
    st.title("📊 Dropout Data Visualization")

    st.subheader("Dropout Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x=target_col, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Dropout by Feature")
    feature = st.selectbox("Choose feature to analyze:", [col for col in df.columns if col != target_col])
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x=feature, hue=target_col, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# --- Prediction Tab ---


elif choice == "Prediction":
    st.title("🔍 Predict Student Dropout Risk")

    input_data = {}
    for col in X.columns:
        if df[col].dtype == 'object':
            input_data[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))



    if st.button("Submit"):
        # Encode input
        input_df = pd.DataFrame([input_data])
        for col, le in label_encoders.items():
            if col in input_df:
                input_df[col] = le.transform(input_df[col].astype(str))


        input_df = input_df[imp]

        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
        else:
            probability = None

        # Generate PDF
        file_path = generate_pdf(input_data, prediction, probability)

        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name="dropout_report.pdf",
                mime="application/pdf"
            )

        #probability = model.predict_proba(input_df)[0][1]  # class 1 = dropout

        st.markdown("---")
        st.write("### Prediction Result:")

        if prediction == 1:
            st.error("🚨 Student is Likely to Dropout")
        else:
            st.success("✅ Student is Not Likely to Dropout")

# --- EduBot Tab ---


elif choice == "EduBot 🤖":
    st.title("💬 Ask EduBot")
    query = st.text_input("Ask something about the app, model, or dropouts:")
    if st.button("Ask"):
        if query.strip():
            answer = ask_edubot(query)
            st.success(f"🤖 {answer}")
