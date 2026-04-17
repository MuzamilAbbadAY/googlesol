import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

st.set_page_config(page_title="MeritAI", layout="wide")

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
# ⚖️ MeritAI
### Multi-Dimensional Fair Decision Intelligence Platform

Ensuring decisions are based on **merit, not bias across gender, religion, ethnicity, or location**.
""")

st.caption("Sensitive attributes are removed from decision-making and used only for fairness auditing.")

# ---------------------------
# LOAD DATA
# ---------------------------
def load_data(file):
    return pd.read_csv(file)

# ---------------------------
# DETECT MULTIPLE SENSITIVE COLUMNS
# ---------------------------
def detect_sensitive_columns(df):
    keywords = {
        "gender": ["gender", "sex"],
        "religion": ["religion"],
        "ethnicity": ["ethnicity", "race"],
        "location": ["location", "city", "region"]
    }

    detected = {}

    for key, vals in keywords.items():
        for col in df.columns:
            if any(v in col.lower() for v in vals):
                detected[key] = col

    return detected

# ---------------------------
# MERIT SCORE
# ---------------------------
def compute_merit_auto(df, numeric_cols):
    weights = np.linspace(1, 2, len(numeric_cols))
    score = np.zeros(len(df))

    for i, col in enumerate(numeric_cols):
        norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-5)
        score += weights[i] * norm

    return score / weights.sum()

# ---------------------------
# FAIRNESS METRICS
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
# MULTI-BIAS METRICS
# ---------------------------
def multi_bias_metrics(df, y_pred, y_true, sensitive_cols):
    results = {}

    for name, col in sensitive_cols.items():
        sens = df.loc[y_true.index, col]

        dp = abs(y_pred[sens==0].mean() - y_pred[sens==1].mean())

        mask0 = (sens == 0) & (y_true == 1)
        mask1 = (sens == 1) & (y_true == 1)

        eo = 0
        if mask0.sum() > 0 and mask1.sum() > 0:
            eo = abs(y_pred[mask0].mean() - y_pred[mask1].mean())

        results[name] = {"DP": dp, "EO": eo}

    return results

# ---------------------------
# FAIR THRESHOLD
# ---------------------------
def find_fair_threshold(y_probs, sensitive):
    thresholds = np.linspace(0.1, 0.9, 50)
    best_t, best_gap = 0.5, float("inf")

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        gap = abs(preds[sensitive==0].mean() - preds[sensitive==1].mean())

        if gap < best_gap:
            best_gap = gap
            best_t = t

    return best_t, best_gap

# ---------------------------
# TRADEOFF CURVE
# ---------------------------
def tradeoff_curve(y_probs, sensitive, y_true):
    thresholds = np.linspace(0.1, 0.9, 50)
    acc_list, dp_list = [], []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        acc_list.append(accuracy_score(y_true, preds))
        dp_list.append(abs(preds[sensitive==0].mean() - preds[sensitive==1].mean()))

    return dp_list, acc_list

# ---------------------------
# HIRING SIMULATION
# ---------------------------
def fairness_selection(df, merit, sensitive, k=50):
    df_temp = df.copy()
    df_temp["merit"] = merit

    selected = df_temp.sort_values(by="merit", ascending=False).head(k)

    ratio = (selected[sensitive]==1).mean()

    return selected, ratio

# ---------------------------
# UI
# ---------------------------
st.header("📌 1. Data Input")

uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if not uploaded:
    st.warning("Upload dataset to continue")
    st.stop()

df = load_data(uploaded)
st.dataframe(df.head())

# Detect sensitive columns
sensitive_cols = detect_sensitive_columns(df)

# Encode sensitive columns
for col in sensitive_cols.values():
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

target = st.selectbox("Select Target Column", df.columns)

# ---------------------------
# RUN ANALYSIS
# ---------------------------
st.header("⚙️ 2. Governance Analysis")

if st.button("Run Analysis"):

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target]

    df["merit_score"] = compute_merit_auto(df, numeric_cols)

    X = df.drop(columns=[target])

    # REMOVE ALL SENSITIVE FEATURES
    for col in sensitive_cols.values():
        if col in X.columns:
            X = X.drop(columns=[col])

    X = pd.get_dummies(X, drop_first=True)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    st.header("📊 3. Core Metrics")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")

    # ---------------------------
    # MULTI-BIAS AUDIT
    # ---------------------------
    st.header("📊 4. Multi-Dimensional Fairness Audit")

    bias_results = multi_bias_metrics(df, y_pred, y_test, sensitive_cols)

    for attr, vals in bias_results.items():
        st.subheader(f"{attr.upper()} Bias")

        col1, col2 = st.columns(2)
        col1.metric("DP Gap", f"{vals['DP']:.3f}")
        col2.metric("EO Gap", f"{vals['EO']:.3f}")

    # ---------------------------
    # THRESHOLD OPTIMIZATION
    # ---------------------------
    st.header("⚙️ 5. Fairness Optimization")

    if sensitive_cols:
        first_sensitive = list(sensitive_cols.values())[0]
        sens_test = df.loc[y_test.index, first_sensitive]

        best_t, best_gap = find_fair_threshold(y_probs, sens_test)

        st.write("Optimal Threshold:", round(best_t, 3))
        st.write("Reduced Bias:", round(best_gap, 3))

    # ---------------------------
    # TRADEOFF
    # ---------------------------
    st.header("📈 6. Trade-off Analysis")

    if sensitive_cols:
        dp_list, acc_list = tradeoff_curve(y_probs, sens_test, y_test)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dp_list, y=acc_list, mode='lines+markers'))

        fig.update_layout(
            xaxis_title="Bias",
            yaxis_title="Accuracy"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # SIMULATION
    # ---------------------------
    st.header("🎯 7. Hiring Simulation")

    if sensitive_cols:
        selected, ratio = fairness_selection(
            df.loc[y_test.index], df.loc[y_test.index]["merit_score"], first_sensitive
        )

        st.write("Selection Ratio (group=1):", ratio)
        st.dataframe(selected.head(20))

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Built for Google Solution Challenge | Multi-Bias Fair AI System")
