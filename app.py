import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="Fair AI Governance System", layout="wide")

# ---------------------------
# Load Data
# ---------------------------
def load_data(file):
    return pd.read_csv(file)

# ---------------------------
# Auto Merit Score
# ---------------------------
def compute_merit_auto(df, numeric_cols):
    weights = np.linspace(1, 2, len(numeric_cols))
    score = np.zeros(len(df))

    for i, col in enumerate(numeric_cols):
        norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-5)
        score += weights[i] * norm

    return score / weights.sum()

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

def merit_gap(df, y_pred, sensitive, merit):
    df_temp = df.copy()
    df_temp["pred"] = y_pred
    df_temp["merit"] = merit

    high_merit = df_temp["merit"] > df_temp["merit"].median()

    g0 = df_temp[(df_temp[sensitive] == 0) & high_merit]
    g1 = df_temp[(df_temp[sensitive] == 1) & high_merit]

    if len(g0) == 0 or len(g1) == 0:
        return 0

    return abs(g0["pred"].mean() - g1["pred"].mean())

# ---------------------------
# UI
# ---------------------------
st.title("⚖️ Fair AI Governance System (Universal)")

uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if not uploaded:
    st.warning("Please upload a dataset to proceed")
    st.stop()

df = load_data(uploaded)

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# Column Selection
target = st.selectbox("🎯 Select Target Column", df.columns)
sensitive = st.selectbox("⚠️ Select Gender Column", df.columns)

# Encode gender if needed
if df[sensitive].dtype == "object":
    df[sensitive] = df[sensitive].astype("category").cat.codes

# Select numeric features
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != target]

# ---------------------------
# Run Analysis
# ---------------------------
if st.button("🚀 Run Governance Analysis"):

    try:
        df["merit_score"] = compute_merit_auto(df, numeric_cols)

        X = df.drop(columns=[target])
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)

        # Remove gender influence
        if sensitive in X.columns:
            X = X.drop(columns=[sensitive])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Baseline Model
        base_model = LogisticRegression(max_iter=1000)
        base_model.fit(X_train, y_train)

        y_pred_base = base_model.predict(X_test)

        sens_test = df.loc[y_test.index, sensitive]
        merit_test = df.loc[y_test.index, "merit_score"]

        acc_base = accuracy_score(y_test, y_pred_base)
        dp_base = demographic_parity(y_pred_base, sens_test)
        eo_base = equal_opportunity(y_test, y_pred_base, sens_test)
        mg_base = merit_gap(df.loc[y_test.index], y_pred_base, sensitive, merit_test)

        # Fair Model
        weights = np.where(df.loc[y_train.index, sensitive] == 1, 1.2, 1.0)

        fair_model = LogisticRegression(max_iter=1000)
        fair_model.fit(X_train, y_train, sample_weight=weights)

        y_pred_fair = fair_model.predict(X_test)

        acc_fair = accuracy_score(y_test, y_pred_fair)
        dp_fair = demographic_parity(y_pred_fair, sens_test)
        eo_fair = equal_opportunity(y_test, y_pred_fair, sens_test)
        mg_fair = merit_gap(df.loc[y_test.index], y_pred_fair, sensitive, merit_test)

        # ---------------------------
        # Results
        # ---------------------------
        st.header("📊 Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline")
            st.metric("Accuracy", f"{acc_base:.3f}")
            st.metric("DP Gap", f"{dp_base:.3f}")
            st.metric("EO Gap", f"{eo_base:.3f}")
            st.metric("Merit Gap", f"{mg_base:.3f}")

        with col2:
            st.subheader("Fair Model")
            st.metric("Accuracy", f"{acc_fair:.3f}")
            st.metric("DP Gap", f"{dp_fair:.3f}")
            st.metric("EO Gap", f"{eo_fair:.3f}")
            st.metric("Merit Gap", f"{mg_fair:.3f}")

        # Visualization
        chart_df = pd.DataFrame({
            "Metric": ["Accuracy", "DP Gap", "EO Gap", "Merit Gap"],
            "Baseline": [acc_base, dp_base, eo_base, mg_base],
            "Fair Model": [acc_fair, dp_fair, eo_fair, mg_fair]
        })

        fig = px.bar(chart_df, x="Metric", y=["Baseline", "Fair Model"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        # Download
        st.download_button(
            "📄 Download Report",
            data=chart_df.to_csv(index=False),
            file_name="fairness_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
