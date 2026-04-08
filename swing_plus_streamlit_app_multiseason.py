
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Swing+ Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR

FEATURE_COLS = [
    "avg_swing_speed",
    "avg_swing_length",
    "attack_angle",
    "attack_direction",
    "vertical_swing_path",
]

DISPLAY_NAMES = {
    "pred_xwobacon": "xwOBAcon",
    "pred_ev": "EV",
    "pred_la": "LA",
    "pred_barrel_percent": "Barrel%",
    "pred_gb_percent": "GB%",
    "pred_zone_contact_percent": "Zone Contact%",
    "pred_whiff_percent": "Whiff%",
}

METRIC_ORDER = [
    "pred_xwobacon",
    "pred_ev",
    "pred_la",
    "pred_barrel_percent",
    "pred_gb_percent",
    "pred_zone_contact_percent",
    "pred_whiff_percent",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = BASE_DIR / "swing_plus_scored_stats51.csv"
    df = pd.read_csv(csv_path)
    if "last_name, first_name" in df.columns and "player_name" not in df.columns:
        df = df.rename(columns={"last_name, first_name": "player_name"})
    return df


@st.cache_resource
def load_models():
    model_whiff = joblib.load(BASE_DIR / "swing_plus_z_whiff_ridge.joblib")
    model_contact = joblib.load(BASE_DIR / "swing_plus_iz_contact_ridge.joblib")
    poly_exec = joblib.load(BASE_DIR / "swing_plus_interaction_transformer.joblib")

    model_ev = joblib.load(BASE_DIR / "ev_elastic.joblib")
    model_la = joblib.load(BASE_DIR / "la_elastic.joblib")
    model_barrel = joblib.load(BASE_DIR / "barrel_elastic.joblib")

    poly_ev = joblib.load(MODEL_DIR / "poly_ev.joblib")
    poly_la = joblib.load(MODEL_DIR / "poly_la.joblib")
    poly_barrel = joblib.load(MODEL_DIR / "poly_barrel.joblib")

    model_xw = joblib.load(BASE_DIR / "xwobacon_ridge.joblib")
    model_gb = joblib.load(BASE_DIR / "gb_elastic.joblib")

    poly_xw = joblib.load(MODEL_DIR / "poly_xw.joblib")
    poly_gb = joblib.load(MODEL_DIR / "poly_gb.joblib")

    return {
        "model_whiff": model_whiff,
        "model_contact": model_contact,
        "poly_exec": poly_exec,
        "model_ev": model_ev,
        "model_la": model_la,
        "model_barrel": model_barrel,
        "poly_ev": poly_ev,
        "poly_la": poly_la,
        "poly_barrel": poly_barrel,
        "model_xw": model_xw,
        "model_gb": model_gb,
        "poly_xw": poly_xw,
        "poly_gb": poly_gb,
    }


def inv_logit(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))) * 100.0


def score_from_inputs(models: dict, inputs: dict) -> dict:
    x_base = pd.DataFrame([inputs], columns=FEATURE_COLS)

    x_exec = models["poly_exec"].transform(x_base)
    pred_whiff = inv_logit(models["model_whiff"].predict(x_exec))[0]
    pred_contact = inv_logit(models["model_contact"].predict(x_exec))[0]

    pred_ev = float(models["model_ev"].predict(models["poly_ev"].transform(x_base))[0])
    pred_la = float(models["model_la"].predict(models["poly_la"].transform(x_base))[0])

    pred_barrel = inv_logit(models["model_barrel"].predict(models["poly_barrel"].transform(x_base)))[0]
    pred_gb = inv_logit(models["model_gb"].predict(models["poly_gb"].transform(x_base)))[0]

    pred_xw = float(models["model_xw"].predict(models["poly_xw"].transform(x_base))[0])

    return {
        "pred_xwobacon": pred_xw,
        "pred_ev": pred_ev,
        "pred_la": pred_la,
        "pred_barrel_percent": pred_barrel,
        "pred_gb_percent": pred_gb,
        "pred_zone_contact_percent": pred_contact,
        "pred_whiff_percent": pred_whiff,
    }


def format_metric(metric_key: str, value: float) -> str:
    if metric_key == "pred_xwobacon":
        return f"{value:.3f}"
    if metric_key in {"pred_ev", "pred_la"}:
        return f"{value:.2f}"
    return f"{value:.2f}%"


def metric_percentile(df: pd.DataFrame, metric_key: str, value: float, higher_is_better: bool = True) -> float:
    series = df[metric_key].dropna()
    if series.empty:
        return math.nan
    pct = float((series <= value).mean() * 100.0)
    return pct if higher_is_better else (100.0 - pct)


def render_metric_cards(df: pd.DataFrame, predictions: dict):
    row1 = st.columns(4)
    row2 = st.columns(3)

    higher_is_better = {
        "pred_xwobacon": True,
        "pred_ev": True,
        "pred_la": True,
        "pred_barrel_percent": True,
        "pred_gb_percent": False,
        "pred_zone_contact_percent": True,
        "pred_whiff_percent": False,
    }

    for col, metric_key in zip(row1 + row2, METRIC_ORDER):
        value = predictions[metric_key]
        pct = metric_percentile(df, metric_key, value, higher_is_better[metric_key])
        delta = None if math.isnan(pct) else f"{pct:.0f}th pct"
        col.metric(label=DISPLAY_NAMES[metric_key], value=format_metric(metric_key, value), delta=delta)


def player_page(df: pd.DataFrame):
    st.title("Swing+ Player Profile")

    if "player_name" not in df.columns:
        st.error("The scored CSV needs a 'player_name' column for the player profile page.")
        return

    working_df = df.copy()

    if "year" in working_df.columns:
        all_years = sorted(working_df["year"].dropna().unique(), reverse=True)
        year_options = ["All"] + [str(y) for y in all_years]
        selected_year = st.selectbox("Season", year_options, index=0)

        if selected_year != "All":
            working_df = working_df[working_df["year"] == int(selected_year)].copy()

    player_options = sorted(working_df["player_name"].astype(str).unique().tolist())
    selected_player = st.selectbox("Select a player", player_options)

    player_df = working_df[working_df["player_name"] == selected_player].copy()

    if player_df.empty:
        st.warning("No player data found for this selection.")
        return

    if "year" in player_df.columns and player_df["year"].nunique() > 1:
        selected_player_year = st.selectbox(
            "Select player season",
            sorted(player_df["year"].unique(), reverse=True),
        )
        player_row = player_df[player_df["year"] == selected_player_year].iloc[0]
    else:
        player_row = player_df.iloc[0]

    left, right = st.columns([1.3, 1.0])

    with left:
        st.subheader(f"{player_row['player_name']}")
        info = []
        if "year" in player_row.index:
            info.append(f"Year: {int(player_row['year'])}")
        if "pa" in player_row.index and pd.notna(player_row["pa"]):
            info.append(f"PA: {int(player_row['pa'])}")
        if info:
            st.caption(" | ".join(info))

        predictions = {k: float(player_row[k]) for k in METRIC_ORDER if k in player_row.index}
        render_metric_cards(df, predictions)

    with right:
        st.subheader("Swing Inputs")
        inputs_table = pd.DataFrame({
            "Input": [
                "Avg Swing Speed",
                "Avg Swing Length",
                "Attack Angle",
                "Attack Direction",
                "Vertical Swing Path",
            ],
            "Value": [
                round(float(player_row["avg_swing_speed"]), 2),
                round(float(player_row["avg_swing_length"]), 2),
                round(float(player_row["attack_angle"]), 2),
                round(float(player_row["attack_direction"]), 2),
                round(float(player_row["vertical_swing_path"]), 2),
            ],
        })
        st.dataframe(inputs_table, use_container_width=True, hide_index=True)

    st.subheader("Predicted Metrics")
    display_df = pd.DataFrame({
        "Metric": [DISPLAY_NAMES[k] for k in METRIC_ORDER],
        "Prediction": [format_metric(k, float(player_row[k])) for k in METRIC_ORDER],
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def leaderboard_page(df: pd.DataFrame):
    st.title("Swing+ Leaderboards")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "year" in df.columns:
            year_options = sorted(df["year"].dropna().unique(), reverse=True)
            selected_year = st.selectbox("Season", year_options)
        else:
            selected_year = None

    with col2:
        metric_key = st.selectbox(
            "Leaderboard metric",
            METRIC_ORDER,
            format_func=lambda x: DISPLAY_NAMES[x],
            index=0,
        )

    with col3:
        top_n = st.selectbox("Show top", [10, 25], index=1)

    leaderboard_source = df.copy()
    if selected_year is not None:
        leaderboard_source = leaderboard_source[leaderboard_source["year"] == selected_year].copy()

    leaderboard_cols = ["player_name"]
    if "year" in leaderboard_source.columns:
        leaderboard_cols.append("year")
    if "pa" in leaderboard_source.columns:
        leaderboard_cols.append("pa")
    leaderboard_cols += METRIC_ORDER

    leaderboard_df = (
        leaderboard_source[leaderboard_cols]
        .sort_values(metric_key, ascending=False)
        .head(top_n)
        .copy()
    )

    leaderboard_df = leaderboard_df.rename(columns=DISPLAY_NAMES)

    for col in ["xwOBAcon", "EV", "LA", "Barrel%", "GB%", "Zone Contact%", "Whiff%"]:
        if col in leaderboard_df.columns:
            leaderboard_df[col] = leaderboard_df[col].round(3 if col == "xwOBAcon" else 2)

    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)


def slider_page(df: pd.DataFrame, models: dict):
    st.title("Swing+ Sandbox")
    st.caption("Move each slider anywhere from the lowest to highest value in the selected season, then view the predicted offensive markers.")

    working_df = df.copy()

    if "year" in working_df.columns:
        year_options = ["All"] + [str(y) for y in sorted(working_df["year"].dropna().unique(), reverse=True)]
        selected_year = st.selectbox("Slider season range", year_options, index=0)

        if selected_year != "All":
            working_df = working_df[working_df["year"] == int(selected_year)].copy()

    cols = st.columns(2)
    slider_values = {}

    for idx, feature in enumerate(FEATURE_COLS):
        col = cols[idx % 2]
        series = working_df[feature].dropna()
        min_val = float(series.min())
        max_val = float(series.max())
        default_val = float(series.median())
        step = max((max_val - min_val) / 200.0, 0.01)

        slider_values[feature] = col.slider(
            label=feature.replace("_", " ").title(),
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step,
        )

    predictions = score_from_inputs(models, slider_values)

    st.subheader("Predicted Metrics")
    render_metric_cards(df, predictions)

    st.subheader("Input Snapshot")
    snapshot_df = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in FEATURE_COLS],
        "Value": [round(float(slider_values[f]), 2) for f in FEATURE_COLS],
    })
    st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

    st.subheader("Predicted Metrics Table")
    pred_table = pd.DataFrame({
        "Metric": [DISPLAY_NAMES[k] for k in METRIC_ORDER],
        "Prediction": [format_metric(k, predictions[k]) for k in METRIC_ORDER],
    })
    st.dataframe(pred_table, use_container_width=True, hide_index=True)


def main():
    df = load_data()
    models = load_models()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Player Profile", "Leaderboards", "Swing Sandbox"])

    if page == "Player Profile":
        player_page(df)
    elif page == "Leaderboards":
        leaderboard_page(df)
    else:
        slider_page(df, models)


if __name__ == "__main__":
    main()
