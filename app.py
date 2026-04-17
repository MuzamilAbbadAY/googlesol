import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fair AI Governance System", layout="wide")

# ---------------------------
# Load Data
# ---------------------------
def load_data(file):
    return pd.read_csv(file)

# ---------------------------
# Merit Score (Auto)
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
# NEW MODULE 1: Fair Threshold
# ---------------------------
def find_fair_threshold(y_probs, sensitive, y_true):
    thresholds = np.linspace(0.1, 0.9, 50)
    best_t = 0.5
    best_gap = float("inf")

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        gap = abs(preds[sensitive==0].mean() - preds[sensitive==1].mean())

        if gap < best_gap:
            best_gap = gap
            best_t = t

    return best_t, best_gap

# ---------------------------
# NEW MODULE 2: Tradeoff Curve
# ---------------------------
def tradeoff_curve(y_probs, sensitive, y_true):
    thresholds = np.linspace(0.1, 0.9, 50)

    acc_list = []
    dp_list = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        dp = abs(preds[sensitive==0].mean() - preds[sensitive==1].mean())

        acc_list.append(acc)
        dp_list.append(dp)

    return thresholds, acc_list, dp_list

# ---------------------------
# NEW MODULE 3: Selection Simulation
# ---------------------------
def fairness_selection(df, merit, sensitive, k=50):
    df_temp = df.copy()
    df_temp["merit"] = merit

    df_sorted = df_temp.sort_values(by="merit", ascending=False)
    selected = df_sorted.head(k)

    male_ratio = (selected[sensitive]==0).mean()
    female_ratio = (selected[sensitive]==1).mean()

    return selected, male_ratio, female_ratio

# ---------------------------
# UI
# ---------------------------
st.title("⚖️ Fair AI Governance System (Advanced)")

uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if not uploaded:
    st.warning("Upload a dataset to proceed")
    st.stop()

df = load_data(uploaded)

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

target = st.selectbox("🎯 Target Column", df.columns)
sensitive = st.selectbox("⚠️ Gender Column", df.columns)

# Encode gender if needed
if df[sensitive].dtype == "object":
    df[sensitive] = df[sensitive].astype("category").cat.codes

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != target]

# ---------------------------
# RUN ANALYSIS
# ---------------------------
if st.button("🚀 Run Governance Analysis"):

    df["merit_score"] = compute_merit_auto(df, numeric_cols)

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)

    if sensitive in X.columns:
        X = X.drop(columns=[sensitive])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    sens_test = df.loc[y_test.index, sensitive]
    merit_test = df.loc[y_test.index, "merit_score"]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    dp = demographic_parity(y_pred, sens_test)
    eo = equal_opportunity(y_test, y_pred, sens_test)
    mg = merit_gap(df.loc[y_test.index], y_pred, sensitive, merit_test)

    st.header("📊 Core Metrics")
    st.metric("Accuracy", f"{acc:.3f}")
    st.metric("DP Gap", f"{dp:.3f}")
    st.metric("EO Gap", f"{eo:.3f}")
    st.metric("Merit Gap", f"{mg:.3f}")

    # ---------------------------
    # MODULE 1: Threshold Optimization
    # ---------------------------
    st.header("⚙️ Fairness-Constrained Threshold")

    best_t, best_gap = find_fair_threshold(y_probs, sens_test, y_test)

    st.write("Optimal Threshold:", round(best_t, 3))
    st.write("Reduced Bias (DP Gap):", round(best_gap, 3))

    # ---------------------------
    # MODULE 2: Tradeoff Curve
    # ---------------------------
    st.header("📈 Fairness vs Accuracy Trade-off")

    thresholds, acc_list, dp_list = tradeoff_curve(y_probs, sens_test, y_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dp_list, y=acc_list, mode='lines+markers'))

    fig.update_layout(
        title="Trade-off Curve",
        xaxis_title="Bias (DP Gap)",
        yaxis_title="Accuracy"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # MODULE 3: Hiring Simulation
    # ---------------------------
    st.header("🎯 Fair Hiring Simulation")

    selected, male_ratio, female_ratio = fairness_selection(
        df.loc[y_test.index], merit_test, sensitive
    )

    st.write("Male Selection Rate:", male_ratio)
    st.write("Female Selection Rate:", female_ratio)

    st.dataframe(selected.head(20))
