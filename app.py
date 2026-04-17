import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Fairness Governance System", layout="wide")

# ---------------------------
# Utility Functions
# ---------------------------

def load_data(uploaded_file):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        # sample dataset
        np.random.seed(42)
        df = pd.DataFrame({
            "age": np.random.randint(18, 60, 200),
            "income": np.random.randint(20000, 100000, 200),
            "gender": np.random.choice([0, 1], 200),
            "approved": np.random.choice([0, 1], 200)
        })
        return df

def demographic_parity(y_pred, sensitive):
    group0 = y_pred[sensitive == 0]
    group1 = y_pred[sensitive == 1]
    return abs(group0.mean() - group1.mean())

def equal_opportunity(y_true, y_pred, sensitive):
    mask0 = (sensitive == 0) & (y_true == 1)
    mask1 = (sensitive == 1) & (y_true == 1)
    if mask0.sum() == 0 or mask1.sum() == 0:
        return 0
    tpr0 = y_pred[mask0].mean()
    tpr1 = y_pred[mask1].mean()
    return abs(tpr0 - tpr1)

# ---------------------------
# UI
# ---------------------------

st.title("⚖️ AI Fairness Governance System")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(uploaded)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select columns
target = st.selectbox("Select Target Column", df.columns)
sensitive = st.selectbox("Select Sensitive Attribute", [c for c in df.columns if c != target])

# ---------------------------
# Run Model
# ---------------------------

if st.button("Run Analysis"):
    try:
        X = df.drop(columns=[target])
        y = df[target]

        # Convert categorical
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # fairness metrics
        sensitive_test = df.loc[y_test.index, sensitive]

        dp = demographic_parity(y_pred, sensitive_test)
        eo = equal_opportunity(y_test, y_pred, sensitive_test)

        # ---------------------------
        # Display Results
        # ---------------------------

        st.header("📊 Model Performance")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("Demographic Parity Gap", f"{dp:.3f}")
        col3.metric("Equal Opportunity Gap", f"{eo:.3f}")

        # Bias flag
        if dp > 0.1 or eo > 0.1:
            st.error("⚠️ Bias Detected in Model")
        else:
            st.success("✅ Model is Fair")

        # ---------------------------
        # Prediction UI
        # ---------------------------

        st.header("🔮 Prediction")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=float(X[col].mean()))

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            st.success(f"Prediction: {pred}")

    except Exception as e:
        st.error(f"Error: {e}")