import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fpdf import FPDF
import tempfile

# Load dataset
try:
    df = pd.read_csv("house_1_daily.csv", parse_dates=['timestamp'])
except FileNotFoundError:
    st.error("Dataset file not found. Please check the file path.")
    st.stop()

# Drop missing values
df.dropna(inplace=True)

# Time features
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['day'] = pd.to_datetime(df['timestamp']).dt.date

# Create binary usage label
df['used'] = df['power_kwh'] > 0.05

# Streamlit title
st.title("🔌 Personalized Appliance Usage Predictor")

# Sidebar appliance selection
if 'appliance' not in df.columns:
    st.error("The dataset does not contain an 'appliance' column.")
    st.stop()

appliances = df['appliance'].unique()
selected_appliance = st.sidebar.selectbox("Select Appliance", appliances)

# Filter appliance data
appliance_df = df[df['appliance'] == selected_appliance].copy()

if appliance_df.empty:
    st.warning("No data available for the selected appliance.")
else:
    features = ['hour', 'day_of_week', 'temperature', 'is_weekend', 'is_holiday']
    if not all(feature in appliance_df.columns for feature in features):
        st.error("The dataset is missing required columns for training.")
        st.stop()

    X = appliance_df[features]
    y = appliance_df['used']

    if len(y.unique()) < 2:
        st.warning("Not enough usage variation to train the model.")
    else:
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

        # Train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluation
        report = classification_report(y_test, model.predict(X_test), output_dict=True)

        # User Input Section
        st.subheader("🔮 Predict Future Usage")

        input_hour = st.slider("Select Hour", 0, 23, 18)
        input_day = st.selectbox("Select Day of Week (0=Mon, 6=Sun)", list(range(7)))
        input_temp = st.slider("Temperature (°C)", 10, 40, 22)
        input_weekend = st.checkbox("Is Weekend?", value=False)
        input_holiday = st.checkbox("Is Holiday?", value=False)

        # Predict
if st.button("Predict Appliance Usage"):
    X_new = pd.DataFrame([[input_hour, input_day, input_temp, int(input_weekend), int(input_holiday)]], columns=features)
    prediction_proba = model.predict_proba(X_new)[0][1]  # Probability of "used" being True
    st.session_state.prediction_result = prediction_proba

# Show prediction result
if 'prediction_result' in st.session_state:
    result = st.session_state.prediction_result
    st.success(f"✅ Prediction Probability: {result:.4f}")  # Display up to 4 decimal places
    st.write("Prediction Value :", result)
# Model Performance
    st.markdown("### ⚙ Model Performance")
    st.write(f"Accuracy: {round(report['accuracy']*100, 2)}%")
    # st.json(report['weighted avg'])

# PDF Suggestion Generator
def generate_pdf(appliance, suggestions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Energy Optimization Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Appliance: {appliance}", ln=True)

    for suggestion in suggestions:
        pdf.multi_cell(0, 10, f"- {suggestion}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        return tmp_file.name

# Suggestions Section
st.markdown("## 💡 Optimization Suggestions")
sample_suggestions = [
    "Use this appliance during off-peak hours to save on electricity bills.",
    "Consider replacing with an energy-efficient model.",
    "Avoid usage during high ambient temperatures unless necessary.",
    "Evaluate potential for solar energy usage."
]

for s in sample_suggestions:
    st.markdown(f"- {s}")

# Download PDF
if st.button("📄 Download Suggestions as PDF"):
    pdf_path = generate_pdf(selected_appliance, sample_suggestions)
    with open(pdf_path, "rb") as f:
        st.download_button(label="Download PDF", data=f, file_name="optimization_suggestions.pdf")




# streamlit run "C:\Users\Akash\Desktop\electricity3\behaviour_learn\behaviour_learn.py"