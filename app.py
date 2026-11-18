import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, precision_score, recall_score, f1_score
)

st.set_page_config(
    page_title="AURA | Student Performance AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Premium UI ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root { color-scheme: dark; }
.stApp {
    background:
      radial-gradient(ellipse at 12% 0%, rgba(99,102,241,.20), transparent 34%),
      radial-gradient(ellipse at 92% 12%, rgba(14,165,233,.13), transparent 30%),
      linear-gradient(145deg, #070b16 0%, #0b1020 52%, #080d19 100%);
    color: #edf2ff;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1427 0%, #0a1020 100%);
    border-right: 1px solid rgba(148,163,184,.16);
}
.block-container { padding-top: 2rem; padding-bottom: 3.5rem; max-width: 1440px; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f8fafc !important; }
p, label, .stMarkdown { color: #cbd5e1; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(25,36,64,.92), rgba(15,23,42,.92));
    border: 1px solid rgba(129,140,248,.24);
    border-radius: 18px; padding: 17px 18px;
    box-shadow: 0 12px 32px rgba(0,0,0,.22);
}
[data-testid="stMetricLabel"] { color: #a5b4fc !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-family: 'Space Grotesk'; }
.hero {
    position: relative; overflow: hidden;
    padding: 30px 32px; border-radius: 24px;
    border: 1px solid rgba(165,180,252,.30);
    background: linear-gradient(115deg, rgba(30,41,89,.94), rgba(17,24,39,.96) 58%, rgba(14,116,144,.38));
    box-shadow: 0 18px 55px rgba(0,0,0,.25);
    margin-bottom: 24px;
}
.hero:after {
    content: "✦"; position: absolute; right: 38px; top: -38px;
    font-size: 190px; color: rgba(199,210,254,.07);
}
.eyebrow {
    display: inline-block; letter-spacing: .14em; text-transform: uppercase;
    font-size: 11px; font-weight: 700; color: #c4b5fd;
    background: rgba(99,102,241,.16); border: 1px solid rgba(165,180,252,.25);
    padding: 7px 11px; border-radius: 999px; margin-bottom: 12px;
}
.hero h1 { font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.08; margin: 0 0 10px 0; }
.hero p { max-width: 720px; color: #cbd5e1; font-size: 1rem; margin: 0; }
.section-head { margin: 24px 0 12px; }
.section-sub { color: #94a3b8; font-size: .9rem; margin-top: -7px; }
.panel {
    background: linear-gradient(145deg, rgba(20,30,52,.88), rgba(13,21,38,.94));
    border: 1px solid rgba(100,116,139,.25); border-radius: 18px;
    padding: 18px 20px; box-shadow: 0 12px 32px rgba(0,0,0,.15);
}
.prediction-pass {
    border: 1px solid rgba(52,211,153,.45);
    background: linear-gradient(120deg, rgba(6,78,59,.38), rgba(15,23,42,.92));
    border-radius: 18px; padding: 20px;
}
.prediction-fail {
    border: 1px solid rgba(251,113,133,.45);
    background: linear-gradient(120deg, rgba(136,19,55,.30), rgba(15,23,42,.92));
    border-radius: 18px; padding: 20px;
}
div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(100deg, #6366f1, #8b5cf6 55%, #0ea5e9);
    color: white; border: 0; border-radius: 12px; min-height: 44px;
    font-weight: 700; box-shadow: 0 8px 22px rgba(99,102,241,.20);
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    border: 1px solid #c4b5fd; color: white; transform: translateY(-1px);
}
div[data-testid="stDataFrame"] { border: 1px solid rgba(100,116,139,.28); border-radius: 12px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(30,41,59,.62); border-radius: 10px 10px 0 0;
    padding: 10px 16px; color: #cbd5e1;
}
.small-note { color: #94a3b8; font-size: 12px; }
hr { border-color: rgba(148,163,184,.16); }
</style>
""", unsafe_allow_html=True)

# ---------- Reproducible synthetic training data ----------
@st.cache_data
def create_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0, 10, n)
    attendance = rng.uniform(40, 100, n)
    previous_marks = rng.uniform(20, 100, n)
    score = (
        study_hours * 4.0
        + attendance * 0.30
        + previous_marks * 0.25
        + rng.normal(0, 5, n)
    )
    result = (score >= 55).astype(int)
    return pd.DataFrame({
        "Study_Hours": np.round(study_hours, 2),
        "Attendance": np.round(attendance, 2),
        "Previous_Marks": np.round(previous_marks, 2),
        "Result": result
    })

@st.cache_resource
def train_model(test_size=0.2, max_depth=4, seed=42):
    data = create_dataset()
    X = data[["Study_Hours", "Attendance", "Previous_Marks"]]
    y = data["Result"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return data, model, X_train, X_test, y_train, y_test, predictions

# ---------- Sidebar controls ----------
st.sidebar.markdown("## ✦ AURA AI")
st.sidebar.caption("STUDENT PERFORMANCE INTELLIGENCE")
st.sidebar.markdown("---")
max_depth = st.sidebar.slider("Decision Tree depth", 1, 10, 4)
test_size = st.sidebar.slider("Test data proportion", 0.10, 0.40, 0.20, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model configuration**")
st.sidebar.write("Algorithm: Decision Tree Classifier")
st.sidebar.write("Dataset: 300 synthetic records")
st.sidebar.write("Random state: 42")
st.sidebar.markdown("---")
st.sidebar.caption("Learning demo • Not a real-world student assessment tool")

data, model, X_train, X_test, y_train, y_test, y_pred = train_model(
    test_size=test_size, max_depth=max_depth
)

accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ Machine learning workspace</div>
  <h1>Student Performance<br>Intelligence</h1>
  <p>Explore a supervised learning pipeline, evaluate a Decision Tree model, and predict a sample student's result — all in one interactive workspace.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Top metrics ----------
st.markdown("### Model at a glance")
st.markdown('<div class="section-sub">Live evaluation metrics update when you adjust the model settings in the sidebar.</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy * 100:.1f}%", help="Share of test predictions classified correctly.")
m2.metric("Precision", f"{precision * 100:.1f}%", help="Of predicted Pass students, how many actually passed.")
m3.metric("Recall", f"{recall * 100:.1f}%", help="Of actual Pass students, how many were found.")
m4.metric("F1 score", f"{f1 * 100:.1f}%", help="Harmonic mean of precision and recall.")

# ---------- Tabs ----------
tab_predict, tab_model, tab_data, tab_about = st.tabs([
    "✦ Predict a student", "◈ Model evaluation", "▤ Dataset explorer", "ⓘ About"
])

with tab_predict:
    left, right = st.columns([1.02, 0.98], gap="large")
    with left:
        st.markdown("### Student profile")
        st.markdown('<div class="section-sub">Adjust the inputs, then ask the model for a prediction.</div>', unsafe_allow_html=True)
        study = st.slider("Daily study hours", 0.0, 10.0, 6.0, 0.5)
        attendance_input = st.slider("Attendance", 40, 100, 85, 1)
        marks = st.slider("Previous exam marks", 20, 100, 75, 1)
        predict_clicked = st.button("✦ Predict performance", use_container_width=True)
    with right:
        st.markdown("### Prediction result")
        if predict_clicked or "last_prediction" in st.session_state:
            if predict_clicked:
                sample = pd.DataFrame([{
                    "Study_Hours": study,
                    "Attendance": attendance_input,
                    "Previous_Marks": marks
                }])
                pred = int(model.predict(sample)[0])
                proba = model.predict_proba(sample)[0]
                st.session_state.last_prediction = {
                    "pred": pred,
                    "probability": float(proba[pred]),
                    "study": study,
                    "attendance": attendance_input,
                    "marks": marks
                }
            result = st.session_state.last_prediction
            if result["pred"] == 1:
                st.markdown(f"""
                <div class="prediction-pass">
                  <div class="eyebrow">Model prediction</div>
                  <h2 style="color:#6ee7b7 !important; margin:0;">PASS ✦</h2>
                  <p style="margin:8px 0 0;">The model classifies this sample as likely to pass.</p>
                  <h3 style="margin:14px 0 0;">{result['probability']*100:.1f}%</h3>
                  <p class="small-note">Model probability for the predicted class — not a guarantee.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-fail">
                  <div class="eyebrow">Model prediction</div>
                  <h2 style="color:#fda4af !important; margin:0;">NEEDS SUPPORT</h2>
                  <p style="margin:8px 0 0;">The model classifies this sample as likely to fail.</p>
                  <h3 style="margin:14px 0 0;">{result['probability']*100:.1f}%</h3>
                  <p class="small-note">Model probability for the predicted class — not a guarantee.</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("**Inputs used**")
            st.write(f"Study: {result['study']} hrs/day  ·  Attendance: {result['attendance']}%  ·  Previous marks: {result['marks']}")
        else:
            st.markdown('<div class="panel"><h3>Ready when you are ✦</h3><p>Set the student profile and select <b>Predict performance</b> to see the model output here.</p></div>', unsafe_allow_html=True)
    st.info("This is a demonstration trained on synthetic data. Do not use it to make decisions about real students.")

with tab_model:
    st.markdown("### Evaluation on held-out test data")
    st.markdown('<div class="section-sub">The test split is kept separate from the data used to fit the model.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Purples",
            x=["Predicted Fail", "Predicted Pass"],
            y=["Actual Fail", "Actual Pass"], aspect="auto"
        )
        fig_cm.update_layout(
            title="Confusion matrix", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Additional metrics**")
        st.metric("Mean Squared Error", f"{mse:.3f}")
        st.metric("Test records", len(y_test))
        st.metric("Training records", len(y_train))
        st.markdown("</div>", unsafe_allow_html=True)
        report = classification_report(
            y_test, y_pred, labels=[0, 1],
            target_names=["Fail", "Pass"], output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)
    st.caption("For classification, precision, recall and F1-score are usually more informative than MSE. Here MSE is the average squared error on the 0/1 labels.")

    st.markdown("### Feature importance")
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig_imp = px.bar(
        importance, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Teal"
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Relative importance"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with tab_data:
    st.markdown("### Dataset explorer")
    st.markdown('<div class="section-sub">A preview of the generated sample dataset used for this learning exercise.</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Total records", len(data))
    b.metric("Pass records", int((data.Result == 1).sum()))
    c.metric("Fail records", int((data.Result == 0).sum()))
    chart_data = data.copy()
    chart_data["Result_Label"] = chart_data["Result"].map({0: "Fail", 1: "Pass"})
    fig_scatter = px.scatter(
        chart_data, x="Study_Hours", y="Previous_Marks",
        color="Result_Label", size="Attendance",
        hover_data=["Attendance"], color_discrete_map={"Fail": "#fb7185", "Pass": "#34d399"},
        title="Study hours vs previous marks"
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", legend_title_text="Outcome"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.dataframe(data.head(30), use_container_width=True)
    st.download_button(
        "↓ Download full dataset (CSV)",
        data.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_dataset.csv",
        mime="text/csv"
    )

with tab_about:
    st.markdown("### How this project works")
    st.markdown("""
    <div class="panel">
      <b>1 · Generate data</b><br>Creates 300 fictional student records with study hours, attendance and previous marks.<br><br>
      <b>2 · Split data</b><br>Uses a stratified train/test split so the test set is held out for evaluation.<br><br>
      <b>3 · Train model</b><br>Fits a Scikit-Learn DecisionTreeClassifier. Change its maximum depth in the sidebar.<br><br>
      <b>4 · Evaluate</b><br>Reports accuracy, precision, recall, F1, confusion matrix and MSE.<br><br>
      <b>5 · Predict</b><br>Accepts a new profile and returns the predicted class.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Important limitations")
    st.warning(
        "The dataset and labels are synthetic and based on a simplified formula. "
        "The model is for learning the supervised ML workflow only; it is not validated "
        "for real educational decisions. A high score on synthetic data does not imply "
        "real-world accuracy."
    )

st.markdown("---")
st.markdown('<div class="small-note">AURA Student Performance AI · Built with Streamlit + Scikit-Learn · Learning project</div>', unsafe_allow_html=True)
