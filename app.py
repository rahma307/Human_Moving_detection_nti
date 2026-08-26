"""
Human Activity Recognition — Streamlit Dashboard

Run with:  streamlit run app.py

Loads pre-computed artifacts produced by HAR_Project.ipynb (in ./artifacts/)
so the dashboard starts instantly without retraining any model.
"""

import ast

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Human Activity Recognition", layout="wide")

ARTIFACTS = "artifacts"


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(f"{ARTIFACTS}/scaler.joblib")
    final_model = joblib.load(f"{ARTIFACTS}/final_model.joblib")
    final_model_name = joblib.load(f"{ARTIFACTS}/final_model_name.joblib")
    comparison_df = pd.read_csv(f"{ARTIFACTS}/model_comparison.csv", index_col=0)
    activity_dist = pd.read_csv(f"{ARTIFACTS}/activity_distribution.csv", index_col=0)
    confusion = pd.read_csv(f"{ARTIFACTS}/final_confusion_matrix.csv", index_col=0)
    pca_2d = pd.read_csv(f"{ARTIFACTS}/pca_2d.csv")
    sample_X = pd.read_csv(f"{ARTIFACTS}/sample_test_X.csv")
    sample_y = pd.read_csv(f"{ARTIFACTS}/sample_test_y.csv")
    return (scaler, final_model, final_model_name, comparison_df,
            activity_dist, confusion, pca_2d, sample_X, sample_y)


(scaler, final_model, final_model_name, comparison_df, activity_dist,
 confusion, pca_2d, sample_X, sample_y) = load_artifacts()

st.title("🏃 Human Activity Recognition Using Smartphones")

# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
section = st.sidebar.radio(
    "Go to",
    ["Project Description", "Dataset Information", "Activity Distribution",
     "Model Comparison", "Confusion Matrix", "PCA Visualization",
     "Live Prediction Demo"],
)

# ---------------------------------------------------------------
# Project description
# ---------------------------------------------------------------
if section == "Project Description":
    st.header("Project Description")
    st.markdown("""
This dashboard summarizes a supervised **multiclass classification** project
that predicts one of six human activities from smartphone accelerometer and
gyroscope features:

- 🚶 Walking
- ⬆️ Walking Upstairs
- ⬇️ Walking Downstairs
- 🪑 Sitting
- 🧍 Standing
- 🛌 Laying

**Dataset:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

The full modeling pipeline (EDA, preprocessing, model training, PCA, feature
importance, hyperparameter tuning, and final evaluation) lives in
`HAR_Project.ipynb`. This dashboard loads the artifacts produced by that
notebook — it does not retrain anything.
""")

# ---------------------------------------------------------------
# Dataset information
# ---------------------------------------------------------------
elif section == "Dataset Information":
    st.header("Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Features", "561")
    col2.metric("Activities", "6")
    col3.metric("Subjects", "30")

    st.markdown("""
The data was collected from 30 volunteers wearing a waist-mounted smartphone
(Samsung Galaxy S II) recording 3-axial linear acceleration and 3-axial
angular velocity at 50 Hz. Signals were split into fixed 2.56-second windows
(128 readings, 50% overlap), and 561 statistical/time/frequency-domain
features were engineered from each window.

**Official split:** 21 subjects (70%) → training, 9 subjects (30%) → testing,
split by subject to avoid leakage between train and test.
""")
    st.write(f"Sample of the test-set features used in this dashboard: {sample_X.shape[0]} rows x {sample_X.shape[1]} columns")
    st.dataframe(sample_X.head())

# ---------------------------------------------------------------
# Activity distribution
# ---------------------------------------------------------------
elif section == "Activity Distribution":
    st.header("Activity Distribution")
    counts = activity_dist["activity"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, palette="viridis", ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel("Activity")
    ax.set_title("Activity Distribution (Train + Test)")
    st.pyplot(fig)
    st.dataframe(counts.rename("Count"))

# ---------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------
elif section == "Model Comparison":
    st.header("Model Comparison")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color="lightgreen"))

    fig, ax = plt.subplots(figsize=(9, 5))
    comparison_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

    best_model = comparison_df["Accuracy"].idxmax()
    st.success(f"Best model by test accuracy: **{best_model}** "
               f"({comparison_df.loc[best_model, 'Accuracy']:.2%})")
    st.info(f"Final tuned model selected for deployment: **{final_model_name}**")

# ---------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------
elif section == "Confusion Matrix":
    st.header(f"Confusion Matrix — {final_model_name} (Test Set)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("""
**Reading the matrix:** the dominant confusion is typically between
`SITTING` and `STANDING` — both static, upright postures that produce very
similar accelerometer/gyroscope signals over a short window. Dynamic
activities (walking variants) are usually well separated from static ones.
""")

# ---------------------------------------------------------------
# PCA visualization
# ---------------------------------------------------------------
elif section == "PCA Visualization":
    st.header("2D PCA Visualization (Training Data)")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=pca_2d, x="PC1", y="PC2", hue="Activity", alpha=0.6, s=15, ax=ax)
    ax.set_title("2D PCA Projection")
    st.pyplot(fig)
    st.markdown("""
Even in two dimensions, the six activities largely split into two visual
clusters: **static** activities (Sitting, Standing, Laying) vs. **dynamic**
activities (the three Walking variants). This confirms that overall
motion intensity is the strongest signal captured by the top principal
components.
""")

# ---------------------------------------------------------------
# Live prediction demo
# ---------------------------------------------------------------
elif section == "Live Prediction Demo":
    st.header("Final Model — Prediction Demonstration")
    st.markdown("""
Manually entering values for 561 features isn't practical, so this demo lets
you pick a **real row sampled from the test set** and see the final tuned
model's prediction compared to the true activity label.
""")

    idx = st.selectbox("Choose a sample row index", sample_X.index)
    true_label = sample_y.iloc[idx, 0]

    row = sample_X.loc[[idx]]
    row_scaled = scaler.transform(row)
    pred_label = final_model.predict(row_scaled)[0]

    col1, col2 = st.columns(2)
    col1.metric("True Activity", true_label)
    col2.metric("Predicted Activity", pred_label,
                delta="✅ Correct" if pred_label == true_label else "❌ Incorrect")

    if hasattr(final_model, "predict_proba"):
        proba = final_model.predict_proba(row_scaled)[0]
        proba_df = pd.DataFrame({"Activity": final_model.classes_, "Probability": proba})
        proba_df = proba_df.sort_values("Probability", ascending=False)
        st.bar_chart(proba_df.set_index("Activity"))

    with st.expander("View raw feature values for this row"):
        st.dataframe(row.T.rename(columns={idx: "value"}))
