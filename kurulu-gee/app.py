# Kurulu Gee - Ensemble Learning Dashboard (Streamlit)
# Run with:  streamlit run app.py
# GenAI disclosure: this dashboard code was generated with AI assistance (Claude / GitHub Copilot)
# and reviewed and edited by the student.
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Kurulu Gee | Sri Lanka Bird Diversity", page_icon="🐦", layout="wide")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FAMILY_COLORS = {"Baseline": "#9e9e9e", "Bagging": "#2e7d32", "Boosting": "#ef6c00",
                 "Neural net": "#6a1b9a", "Stacking": "#c62828", "Hybrid": "#1565c0"}


@st.cache_resource
def load_models():
    return joblib.load("artifacts/models.joblib")


@st.cache_data
def load_tables():
    return (pd.read_csv("artifacts/district_month.csv"),
            pd.read_csv("artifacts/results.csv"),
            pd.read_csv("artifacts/feature_importance.csv"),
            pd.read_csv("artifacts/map_sample.csv"),
            json.load(open("artifacts/features.json")))


models = load_models()
agg, results, importance, map_sample, feats = load_tables()
ENV = feats["env"]

st.title("🐦 Kurulu Gee: Sri Lanka Bird Diversity Predictor")
st.caption("Ensemble learning (Bagging · Boosting · Stacking · Hybrid RF+ANN) on 2014–2024 bird "
           "sightings with satellite and climate features.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏆 Model comparison", "🔮 Predict", "💡 Insights"])

# ---------------- Overview ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sightings used", f"{int(agg['n_records'].sum()):,}")
    c2.metric("Districts", agg["district"].nunique())
    c3.metric("District-months (rows)", f"{len(agg):,}")
    c4.metric("Best model", results.iloc[0]["model"])

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Where the sightings are")
        st.map(map_sample, latitude="lat", longitude="lon", size=20)
    with right:
        st.subheader("Species richness by month")
        by_month = agg.groupby("month")["species_richness"].mean().reindex(range(1, 13)).reset_index()
        by_month["month_name"] = [MONTHS[m - 1] for m in by_month["month"]]
        st.plotly_chart(px.bar(by_month, x="month_name", y="species_richness",
                               labels={"species_richness": "avg species", "month_name": ""},
                               color_discrete_sequence=["#2e7d32"]), width="stretch")
        st.caption("Sep–Apr is the northern-hemisphere migrant season in Sri Lanka.")

    st.subheader("Species richness by district")
    by_d = agg.groupby("district")["species_richness"].mean().sort_values().reset_index()
    st.plotly_chart(px.bar(by_d, x="species_richness", y="district", orientation="h", height=600,
                           labels={"species_richness": "avg species per month", "district": ""},
                           color_discrete_sequence=["#1565c0"]), width="stretch")

# ---------------- Model comparison ----------------
with tab2:
    st.subheader("Test-set performance (same 20% hold-out for every model)")
    metric = st.radio("Metric", ["RMSE", "MAE", "R2"], horizontal=True)
    order = results.sort_values(metric, ascending=(metric != "R2"))
    fig = px.bar(order, x=metric, y="model", color="family", orientation="h",
                 color_discrete_map=FAMILY_COLORS, height=480)
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": order["model"][::-1].tolist()})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(results.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
                 width="stretch", hide_index=True)
    st.info("**Lower RMSE/MAE = better. Higher R² = better.** RMSE punishes big mistakes more than MAE.")

# ---------------- Predict ----------------
with tab3:
    st.subheader("Predict how many bird species you'd record")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        district = st.selectbox("District", sorted(agg["district"].unique()))
        month = st.select_slider("Month", options=list(range(1, 13)), value=1,
                                 format_func=lambda m: MONTHS[m - 1])
        year = st.slider("Year", int(agg["year"].min()), int(agg["year"].max()) + 1, int(agg["year"].max()))
        n_records = st.slider("Survey effort (number of sightings logged)", 20, 5000, 300, step=10)

        hist = agg[(agg["district"] == district) & (agg["month"] == month)]
        if hist.empty:
            hist = agg[agg["district"] == district]
        defaults = hist[ENV].median()
        with st.expander("Environment (defaults = typical values for this district & month)"):
            env_vals = {}
            for c in ENV:
                lo, hi = float(agg[c].min()), float(agg[c].max())
                env_vals[c] = st.slider(c, lo, hi, float(np.clip(defaults[c], lo, hi)))

    row = {"district": district, "year": year,
           "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
           "migrant_season": int(month in [9, 10, 11, 12, 1, 2, 3, 4]),
           "log_records": np.log1p(n_records),
           "lat": agg.loc[agg["district"] == district, "lat"].mean(),
           "lon": agg.loc[agg["district"] == district, "lon"].mean(), **env_vals}
    X = pd.DataFrame([row])[feats["cat"] + feats["num"]]

    rf = models["Random Forest"].predict(X)[0]
    ann = models["ANN (MLP)"].predict(X)[0]
    w = models["blend_weight"]
    preds = {"Random Forest": rf,
             "LightGBM": models["LightGBM"].predict(X)[0],
             "ANN (MLP)": ann,
             "Stacking (RF+LGBM+ANN)": models["Stacking (RF+LGBM+ANN)"].predict(X)[0],
             "Hybrid RF + ANN (residual)": rf + models["ANN residual"].predict(X)[0],
             "Hybrid RF + ANN (blend)": w * rf + (1 - w) * ann}
    pred_df = pd.DataFrame({"model": list(preds), "predicted species": [max(0, v) for v in preds.values()]})

    with col_b:
        # headline = the best-scoring model that the dashboard has loaded
        best = next(m for m in results["model"] if m in preds)
        st.metric(f"Predicted species richness ({best})", f"{max(0, preds[best]):.0f} species")
        st.plotly_chart(px.bar(pred_df, x="predicted species", y="model", orientation="h",
                               color_discrete_sequence=["#2e7d32"]), width="stretch")
        st.caption("When the models agree, the prediction is more trustworthy. "
                   "Try changing the month to see the migrant-season effect.")

# ---------------- Insights ----------------
with tab4:
    st.subheader("What drives bird diversity? (permutation importance, Random Forest)")
    st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature",
                           orientation="h", height=500, color_discrete_sequence=["#2e7d32"]),
                    width="stretch")
    st.markdown("""
**How to read this:** each bar shows how much the test error grows when that feature is shuffled.
A bigger bar means the model relies on that feature more.

**Key takeaways:**
- **Survey effort dominates.** `log_records` is by far the strongest feature, because more birdwatching finds
  more species. Even linear regression reaches R² 0.87 mostly through effort, so what we "know" about
  bird diversity is shaped by where citizen scientists go.
- **Location comes next.** Longitude and latitude separate the dry-zone and coastal districts from the
  wet-zone and hill country. Elevation adds the hill-country habitat effect.
- **Human footprint:** night-time lights (`avg_rad`), a proxy for urbanisation, is the third most
  important feature.
- **Season matters:** the month and the Sep–Apr migrant season add species as winter migrants arrive.
- **NDVI, rainfall and aerosol add little** once location and season are known.
- **Best model: LightGBM** (RMSE 12.67, R² 0.916). Boosting beat bagging, stacking tied with it, and
  mixing RF with an ANN (hybrid) improved RF from RMSE 14.67 to 13.80.
""")
