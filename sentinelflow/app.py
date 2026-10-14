from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_generator import (
    save_sample_data
)

from src.etl import (
    clean_events,
    save_processed_data
)

from src.models import (
    train_risk_model,
    score_events
)

from src.storage import (
    save_to_sqlite
)


# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

SAMPLE_PATH = (
    DATA_DIR / "sample_events.csv"
)


# =====================================================
# STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SentinelFlow",
    page_icon="🛡️",
    layout="wide"
)


# =====================================================
# PAGE TITLE
# =====================================================

st.title(
    "🛡️ SentinelFlow"
)

st.subheader(
    "Real-Time Predictive Intelligence Platform"
)

st.caption(
    "ETL + Anomaly Detection + Risk Forecasting "
    "+ Automated Decision Support"
)


# =====================================================
# LOAD OR CREATE SAMPLE DATA
# =====================================================

@st.cache_data
def load_or_create_sample():

    DATA_DIR.mkdir(
        exist_ok=True
    )

    if not SAMPLE_PATH.exists():

        save_sample_data(
            str(SAMPLE_PATH)
        )

    return pd.read_csv(
        SAMPLE_PATH
    )


# =====================================================
# TRAIN MODEL
# =====================================================

@st.cache_resource
def get_risk_model():

    training_raw = (
        load_or_create_sample()
    )

    training_clean = (
        clean_events(
            training_raw
        )
    )

    return train_risk_model(
        training_clean
    )


# =====================================================
# PROCESS DATA
# =====================================================

def process_dataframe(raw_df):

    cleaned = (
        clean_events(
            raw_df
        )
    )

    model, _ = get_risk_model()

    scored = (
        score_events(
            cleaned,
            model
        )
    )

    save_processed_data(
        scored,
        str(
            DATA_DIR
            / "processed_events.csv"
        )
    )

    save_to_sqlite(
        scored,
        str(
            DATA_DIR
            / "sentinelflow.db"
        )
    )

    return scored


# =====================================================
# SIDEBAR
# =====================================================

sample_df = (
    load_or_create_sample()
)

with st.sidebar:

    st.header(
        "Data Input"
    )

    source = st.radio(
        "Choose data source",
        [
            "Use sample data",
            "Upload CSV"
        ]
    )

    uploaded = None

    if source == "Upload CSV":

        uploaded = st.file_uploader(
            "Upload a CSV",
            type=["csv"]
        )

    st.markdown(
        "### Required CSV columns"
    )

    st.code(
        "timestamp, entity_id, scenario, "
        "event_type, location, value_1, "
        "value_2, value_3"
    )

    st.markdown(
        "### Prototype scenarios"
    )

    st.write(
        "Retail • Manufacturing • Cybersecurity"
    )


# =====================================================
# SELECT DATA
# =====================================================

if (
    source == "Upload CSV"
    and uploaded is not None
):

    raw_df = pd.read_csv(
        uploaded
    )

    st.success(
        "CSV uploaded successfully."
    )

elif (
    source == "Upload CSV"
    and uploaded is None
):

    st.info(
        "Upload a CSV to continue. "
        "Sample data is being shown for now."
    )

    raw_df = sample_df

else:

    raw_df = sample_df


# =====================================================
# PROCESS DATA
# =====================================================

try:

    results = process_dataframe(
        raw_df
    )

except Exception as error:

    st.error(
        f"Could not process the data: {error}"
    )

    st.stop()


model, metrics = (
    get_risk_model()
)


# =====================================================
# TOP METRICS
# =====================================================

total_events = len(
    results
)

anomalies = int(
    results["anomaly_flag"].sum()
)

high_risk = int(
    (
        results["risk_level"]
        == "HIGH"
    ).sum()
)

average_priority = float(
    results["priority_score"].mean()
)


col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(
    "Total Events",
    f"{total_events:,}"
)

col2.metric(
    "Anomalies",
    f"{anomalies:,}"
)

col3.metric(
    "High-Risk Events",
    f"{high_risk:,}"
)

col4.metric(
    "Average Priority",
    f"{average_priority:.2f}"
)


st.divider()


# =====================================================
# EVENT TREND
# =====================================================

st.subheader(
    "📈 Event Activity Over Time"
)

trend = (
    results
    .set_index("timestamp")
    .resample("D")
    .size()
    .rename("events")
    .to_frame()
)

st.line_chart(
    trend
)


# =====================================================
# SCENARIO + RISK
# =====================================================

left, right = (
    st.columns(2)
)


with left:

    st.subheader(
        "Events by Scenario"
    )

    scenario_counts = (
        results["scenario"]
        .value_counts()
    )

    st.bar_chart(
        scenario_counts
    )


with right:

    st.subheader(
        "Risk-Level Distribution"
    )

    risk_counts = (
        results["risk_level"]
        .value_counts()
    )

    st.bar_chart(
        risk_counts
    )


# =====================================================
# HIGH PRIORITY EVENTS
# =====================================================

st.subheader(
    "🚨 Highest-Priority Events"
)

display_columns = [
    "timestamp",
    "entity_id",
    "scenario",
    "event_type",
    "anomaly_score",
    "risk_probability",
    "risk_level",
    "priority_score",
    "recommended_action"
]

top_events = (
    results
    .sort_values(
        "priority_score",
        ascending=False
    )
    [display_columns]
    .head(15)
)

st.dataframe(
    top_events,
    use_container_width=True
)


# =====================================================
# ETL STATUS
# =====================================================

st.subheader(
    "🔄 ETL Pipeline Status"
)

etl1, etl2, etl3 = (
    st.columns(3)
)

etl1.success(
    "Extract ✓"
)

etl2.success(
    "Transform ✓"
)

etl3.success(
    "Load ✓"
)

st.write(
    "The system reads the data, cleans it, "
    "creates features, detects anomalies, "
    "predicts risk and stores the processed "
    "results in CSV and SQLite."
)


# =====================================================
# MODEL EVALUATION
# =====================================================

st.subheader(
    "🤖 Risk Model Evaluation"
)

m1, m2, m3, m4 = (
    st.columns(4)
)

m1.metric(
    "Accuracy",
    f"{metrics['accuracy']:.2%}"
)

m2.metric(
    "Precision",
    f"{metrics['precision']:.2%}"
)

m3.metric(
    "Recall",
    f"{metrics['recall']:.2%}"
)

m4.metric(
    "F1 Score",
    f"{metrics['f1']:.2%}"
)

st.caption(
    "These metrics are calculated using a held-out "
    "test portion of the built-in demonstration dataset. "
    "For the final project, evaluate using a separate "
    "real-world labelled dataset."
)


# =====================================================
# DOWNLOAD RESULTS
# =====================================================

st.subheader(
    "⬇️ Download Results"
)

csv_bytes = (
    results
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Download processed results as CSV",
    data=csv_bytes,
    file_name="sentinelflow_results.csv",
    mime="text/csv"
)


# =====================================================
# PROJECT NOTE
# =====================================================

st.info(
    "Prototype note: this first version processes "
    "timestamped event data locally to simulate "
    "real-time intelligence. Kafka/Kinesis and "
    "Spark Streaming can be added later."
)