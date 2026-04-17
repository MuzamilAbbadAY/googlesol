import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="AI Fairness Governance System", layout="wide")

# ---------------------------
# Data Loader
# ---------------------------
def load_data(file):
    if file:
        return pd.read_csv(file)
    else:
        np.random.seed(42)
        return pd.DataFrame({
            "age": np.random.randint(18, 60, 300),
            "income": np.random.randint(20000, 100000, 300),
            "gender": np.random.choice([0, 1], 300),
            "approved": np.random.choice([0, 1], 300)
        })

# ---------------------------
# Fairness Metrics
# ---------------------------
def demographic_parity(y_pred, sensitive):
    return abs(y_pred[sensitive == 0].mean() - y_pred[sensitive == 1].mean())

def equal_opportunity(y_true, y_pred, sensitive):
    mask0 = (sensitive == 0) & (y_true == 1)
    mask1 = (sensitive == 1) & (y_true == 1)
    if mask0.sum() == 0 or mask1.sum() == 0:
        return 0
    return abs(y_pred[mask0].mean() - y_pred[mask1].mean())

# ---------------------------
# Model Training
# ---------------------------
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# ---------------------------
# Reweighting (Simple Bias Fix)
# ---------------------------
def reweight_data(X, y, sensitive):
    weights = np.where(sensitive == 1, 1.2, 1.0)
    return weights

# ---------------------------
# UI
# ---------------------------

st.title("⚖️ AI Fairness Governance System (Advanced)")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(uploaded)

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

target = st.selectbox("🎯 Target Column", df.columns)
sensitive = st.selectbox("⚠️ Sensitive Attribute", [c for c in df.columns if c != target])

if st.button("🚀 Run Full Analysis"):

    try:
        X = df.drop(columns=[target])
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # ---------------------------
        # Baseline Model
        # ---------------------------
        base_model = train_model(X_train, y_train)
        y_pred_base = base_model.predict(X_test)

        acc_base = accuracy_score(y_test, y_pred_base)
        sens_test = df.loc[y_test.index, sensitive]

        dp_base = demographic_parity(y_pred_base, sens_test)
        eo_base = equal_opportunity(y_test, y_pred_base, sens_test)

        # ---------------------------
        # Fair Model (Reweighted)
        # ---------------------------
        weights = reweight_data(X_train, y_train, df.loc[y_train.index, sensitive])
        fair_model = LogisticRegression(max_iter=1000)
        fair_model.fit(X_train, y_train, sample_weight=weights)

        y_pred_fair = fair_model.predict(X_test)

        acc_fair = accuracy_score(y_test, y_pred_fair)
        dp_fair = demographic_parity(y_pred_fair, sens_test)
        eo_fair = equal_opportunity(y_test, y_pred_fair, sens_test)

        # ---------------------------
        # Metrics Display
        # ---------------------------
        st.header("📊 Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline Model")
            st.metric("Accuracy", f"{acc_base:.3f}")
            st.metric("DP Gap", f"{dp_base:.3f}")
            st.metric("EO Gap", f"{eo_base:.3f}")

        with col2:
            st.subheader("Fair Model")
            st.metric("Accuracy", f"{acc_fair:.3f}")
            st.metric("DP Gap", f"{dp_fair:.3f}")
            st.metric("EO Gap", f"{eo_fair:.3f}")

        # ---------------------------
        # Visualization
        # ---------------------------
        st.header("📈 Fairness Visualization")

        chart_df = pd.DataFrame({
            "Metric": ["Accuracy", "DP Gap", "EO Gap"],
            "Baseline": [acc_base, dp_base, eo_base],
            "Fair Model": [acc_fair, dp_fair, eo_fair]
        })

        fig = px.bar(chart_df, x="Metric", y=["Baseline", "Fair Model"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------
        # Bias Detection
        # ---------------------------
        st.header("🚨 Bias Detection")

        if dp_base > 0.1 or eo_base > 0.1:
            st.error("Baseline model is biased")
        else:
            st.success("Baseline model is fair")

        if dp_fair < dp_base and eo_fair < eo_base:
            st.success("Fair model improved fairness ✅")

        # ---------------------------
        # Prediction UI
        # ---------------------------
        st.header("🔮 Make Prediction")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=float(X[col].mean()))

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            pred = fair_model.predict(input_df)[0]
            st.success(f"Prediction: {pred}")

        # ---------------------------
        # Download Report
        # ---------------------------
        st.header("📄 Download Report")

        report = chart_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV Report",
            data=report,
            file_name="fairness_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
