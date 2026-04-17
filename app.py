import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

st.set_page_config(page_title="MeritAI", layout="wide")

# ---------------------------
# HEADER (POLISHED UI)
# ---------------------------
st.markdown("""
# ⚖️ MeritAI
### Fair Decision Intelligence Platform

Ensuring hiring and workplace decisions are driven by **merit, not bias**.
""")

st.caption("Gender is removed from decision-making and used only for internal fairness auditing.")

# ---------------------------
# LOAD DATA
# ---------------------------
def load_data(file):
    return pd.read_csv(file)

# ---------------------------
# AUTO DETECT GENDER
# ---------------------------
def detect_sensitive_column(df):
    keywords = ["gender", "sex"]
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None

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
# FAIR THRESHOLD
# ---------------------------
def find_fair_threshold(y_probs, sensitive):
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

    male_ratio = (selected[sensitive]==0).mean()
    female_ratio = (selected[sensitive]==1).mean()

    return selected, male_ratio, female_ratio

# ---------------------------
# UI FLOW
# ---------------------------
st.header("📌 1. Data Input")

uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if not uploaded:
    st.warning("Upload a dataset to begin")
    st.stop()

df = load_data(uploaded)
st.dataframe(df.head())

# Auto detect sensitive column
sensitive = detect_sensitive_column(df)

if sensitive is None:
    st.warning("No gender column detected. Fairness audit limited.")
else:
    if df[sensitive].dtype == "object":
        df[sensitive] = df[sensitive].astype("category").cat.codes

# Target selection (only this is allowed)
target = st.selectbox("Select Target Column", df.columns)

# ---------------------------
# RUN SYSTEM
# ---------------------------
st.header("⚙️ 2. Governance Analysis")

if st.button("Run Analysis"):

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target]

    df["merit_score"] = compute_merit_auto(df, numeric_cols)

    X = df.drop(columns=[target])

    # REMOVE GENDER FROM MODEL
    if sensitive and sensitive in X.columns:
        X = X.drop(columns=[sensitive])

    X = pd.get_dummies(X, drop_first=True)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    sens_test = df.loc[y_test.index, sensitive] if sensitive else None
    merit_test = df.loc[y_test.index, "merit_score"]

    # ---------------------------
    # METRICS
    # ---------------------------
    st.header("📊 3. Fairness Metrics")

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.3f}")

    if sensitive is not None:
        dp = demographic_parity(y_pred, sens_test)
        eo = equal_opportunity(y_test, y_pred, sens_test)
        mg = merit_gap(df.loc[y_test.index], y_pred, sensitive, merit_test)

        st.metric("DP Gap", f"{dp:.3f}")
        st.metric("EO Gap", f"{eo:.3f}")
        st.metric("Merit Gap", f"{mg:.3f}")

        # Insight box
        if mg > 0.1:
            st.warning("⚠️ High-merit candidates are not being treated equally.")
        else:
            st.success("✅ Fair treatment across high-merit candidates.")

    # ---------------------------
    # THRESHOLD
    # ---------------------------
    st.header("⚙️ 4. Fairness Optimization")

    if sensitive is not None:
        best_t, best_gap = find_fair_threshold(y_probs, sens_test)
        st.write("Optimal Threshold:", round(best_t, 3))
        st.write("Reduced Bias:", round(best_gap, 3))

    # ---------------------------
    # TRADEOFF
    # ---------------------------
    st.header("📈 5. Trade-off Analysis")

    if sensitive is not None:
        dp_list, acc_list = tradeoff_curve(y_probs, sens_test, y_test)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dp_list, y=acc_list, mode='lines+markers'))

        fig.update_layout(
            xaxis_title="Bias (DP Gap)",
            yaxis_title="Accuracy"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # SIMULATION
    # ---------------------------
    st.header("🎯 6. Hiring Simulation")

    if sensitive is not None:
        selected, male_ratio, female_ratio = fairness_selection(
            df.loc[y_test.index], merit_test, sensitive
        )

        st.write("Male Selection Rate:", male_ratio)
        st.write("Female Selection Rate:", female_ratio)

        st.dataframe(selected.head(20))

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Built for Google Solution Challenge | Ethical AI for Real-World Impact")
