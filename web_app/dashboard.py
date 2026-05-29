"""
web_app/dashboard.py
────────────────────
Career Guidance AI — Dark Analytics Dashboard (Streamlit)

Run from project root:
    streamlit run web_app/dashboard.py
"""

import ast
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import torch
import torch.nn.functional as F
import pickle
from collections import Counter
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Absolute paths ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_DIR  = os.path.join(BASE_DIR, "models", "custom_distilbert")
DATA_DIR   = os.path.join(MODEL_DIR, "data")
OUTPUT_DIR = os.path.join(MODEL_DIR, "outputs")

DATA_PATH            = os.path.join(DATA_DIR,   "cleaned_jobs_dataset.csv")
CLUSTERED_DATA_PATH  = os.path.join(OUTPUT_DIR, "FINAL_CLUSTERED_DATA.csv")
CAREER_MAPPING_PATH  = os.path.join(OUTPUT_DIR, "career_mapping.json")
CLUSTER_KEYWORDS_PATH= os.path.join(OUTPUT_DIR, "cluster_keywords.json")
CUSTOM_PATH          = os.path.join(MODEL_DIR,  "saved_models", "custom_distilbert", "custom_model.pt")
LE_PATH              = os.path.join(MODEL_DIR,  "saved_models", "label_encoder.pkl")

# Add custom_distilbert to path for model imports
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from modeling_distilbert import DistilBertForCareerPath
from configuration_distilbert import OptimDistilBertConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Theme initialisation ───────────────────────────────────────────────────────
# st.session_state is accessible before st.set_page_config (it is not a render
# call). We init here so every design token below is already theme-conditional.
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
IS_DARK = st.session_state.get("theme", "dark") == "dark"

# ── Design tokens (all theme-conditional) ─────────────────────────────────────
# Accent colours shift slightly between themes for contrast on light backgrounds.
PINK   = "#FF2D78" if IS_DARK else "#E0005E"
CYAN   = "#00F0FF" if IS_DARK else "#007ACC"
PURPLE = "#7B2FFF" if IS_DARK else "#5A00CC"
LIME   = "#39FF14"
ORANGE = "#FF6B00"
YELLOW = "#FFD700"

BG         = "#0D0D2B" if IS_DARK else "#F0F2FF"
SIDEBAR_BG = "#13133A" if IS_DARK else "#FFFFFF"
CARD_BG    = "#1A1A40" if IS_DARK else "#FFFFFF"
TEXT_PRI   = "#FFFFFF"  if IS_DARK else "#1A1A3E"
TEXT_SEC   = "#9B9BC4"  if IS_DARK else "#4A4A7A"
BORDER     = "#2A2A5A"  if IS_DARK else "#D8DAF0"

# High-contrast color palette
CLUSTER_PALETTE = [
    "#FF2D78","#00F0FF","#7B2FFF","#39FF14","#FF6B00",
    "#FFD700","#FF69B4","#00BFFF","#FF4500","#ADFF2F"
]

# Force st.radio label color inheritance
st.markdown(f"<style>.stRadio label {{ color: {TEXT_SEC} !important; }}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — compute_stats() — no static numbers, reads live from CSV
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_stats(data_path: str) -> dict:
    """
    Compute live stats from the canonical CSV.
    Returns a dict with total_jobs, total_careers, total_clusters, model_name.
    Gracefully handles missing columns (count = 0, logs warning).
    """
    try:
        df_raw = pd.read_csv(data_path)
    except Exception as exc:
        st.warning(f"[compute_stats] Could not read CSV: {exc}")
        return {"total_jobs": 0, "total_careers": 0, "total_clusters": 0, "model_name": "Custom DistilBERT"}

    result = {"model_name": "Custom DistilBERT"}

    # Count from raw — no filters applied here
    result["total_jobs"] = len(df_raw)

    for key, col in [("total_careers", "career_path_name"), ("total_clusters", "cluster")]:
        if col in df_raw.columns:
            result[key] = int(df_raw[col].nunique())
        else:
            st.warning(f"[compute_stats] Column '{col}' not found — defaulting {key} to 0")
            result[key] = 0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (all cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_main_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_clustered_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1 — Model metrics loader (reads files; returns N/A if absent)
# ─────────────────────────────────────────────────────────────────────────────
SAVED_MODELS_DIR    = os.path.join(MODEL_DIR, "saved_models")
OUTPUT_METRICS_DIR  = os.path.join(MODEL_DIR, "outputs")  # where CSVs live

MODEL_REGISTRY = [
    {
        "name":       "Custom DistilBERT",
        "dir":        os.path.join(SAVED_MODELS_DIR, "custom_distilbert"),
        "report_csv": os.path.join(OUTPUT_METRICS_DIR, "custom_distilbert_classification_report.csv"),
        "model_files": ["custom_model.pt"],
    },
    {
        "name":       "BERT Base",
        "dir":        os.path.join(SAVED_MODELS_DIR, "bert_base"),
        "report_csv": os.path.join(OUTPUT_METRICS_DIR, "bert base_classification_report.csv"),
        "model_files": ["bert_base.pt"],
    },
    {
        "name":       "ML Baseline",
        "dir":        os.path.join(SAVED_MODELS_DIR, "ml_baseline"),
        "report_csv": os.path.join(OUTPUT_METRICS_DIR, "ml baseline_classification_report.csv"),
        "model_files": ["random_forest.pkl", "vectorizer.pkl"],
    },
]


def _fmt_pct(v: float) -> str:
    """Format a 0-1 float as 'XX.XX%'."""
    return f"{v * 100:.2f}%"


def _model_size_mb(entry: dict) -> str:
    """Sum sizes of all listed model files and return as 'X.X MB'."""
    total = 0
    for fname in entry.get("model_files", []):
        fpath = os.path.join(entry["dir"], fname)
        if os.path.isfile(fpath):
            total += os.path.getsize(fpath)
    return f"{total / 1_048_576:.1f} MB" if total > 0 else "N/A"

METRIC_KEYS = [
    ("accuracy",       ["accuracy", "eval_accuracy", "test_accuracy"]),
    ("f1",             ["f1", "f1_score", "macro_f1", "weighted_f1", "eval_f1"]),
    ("precision",      ["precision", "macro_precision", "eval_precision"]),
    ("recall",         ["recall", "macro_recall", "eval_recall"]),
    ("inference_time", ["inference_time", "inference_time_ms", "avg_inference_ms"]),
]


@st.cache_data(show_spinner=False)
def load_model_metrics() -> list:
    """
    Load real metrics from classification_report CSVs in outputs/.
    Falls back to JSON metric files if a CSV is absent.
    Never fabricates numbers; missing values -> 'N/A'.
    Returns list of dicts: Model, accuracy, f1, precision, recall,
    inference_time, model_size, per_class.
    """
    rows = []
    for entry in MODEL_REGISTRY:
        row = {
            "Model":          entry["name"],
            "accuracy":       "N/A",
            "f1":             "N/A",
            "precision":      "N/A",
            "recall":         "N/A",
            "inference_time": "N/A",
            "model_size":     _model_size_mb(entry),
            "per_class":      {},
        }

        # ── 1. Read from classification_report CSV (primary) ───────────────
        csv_path = entry.get("report_csv", "")
        if csv_path and os.path.isfile(csv_path):
            try:
                df_rep = pd.read_csv(csv_path, index_col=0)
                if "accuracy" in df_rep.index:
                    row["accuracy"] = _fmt_pct(float(df_rep.loc["accuracy", "f1-score"]))
                for avg_row, src_col, dst in [
                    ("weighted avg", "f1-score",  "f1"),
                    ("weighted avg", "precision", "precision"),
                    ("weighted avg", "recall",    "recall"),
                ]:
                    if avg_row in df_rep.index:
                        row[dst] = _fmt_pct(float(df_rep.loc[avg_row, src_col]))
                skip = {"accuracy", "macro avg", "weighted avg"}
                for idx in df_rep.index:
                    if idx not in skip and pd.notna(idx):
                        try:
                            row["per_class"][str(idx)] = {
                                "precision": float(df_rep.loc[idx, "precision"]),
                                "recall":    float(df_rep.loc[idx, "recall"]),
                                "f1":        float(df_rep.loc[idx, "f1-score"]),
                                "support":   int(float(df_rep.loc[idx, "support"])),
                            }
                        except Exception:
                            pass
            except Exception:
                pass

        # ── 2. Fallback: JSON metric files ─────────────────────────────────
        if row["accuracy"] == "N/A":
            JSON_FILES = ["metrics.json", "eval_results.json", "results.json", "eval_result.json"]
            data = {}
            for fname in JSON_FILES:
                fpath = os.path.join(entry["dir"], fname)
                if os.path.isfile(fpath):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        break
                    except Exception:
                        pass
            for out_key, candidates in METRIC_KEYS:
                if row.get(out_key, "N/A") != "N/A":
                    continue
                for c in candidates:
                    if c in data:
                        found = data[c]
                        if out_key == "inference_time":
                            row[out_key] = f"{float(found):.1f} ms"
                        else:
                            try:
                                fv = float(found)
                                row[out_key] = _fmt_pct(fv) if fv <= 1.0 else f"{fv:.2f}%"
                            except (ValueError, TypeError):
                                row[out_key] = str(found)
                        break

        rows.append(row)
    return rows


@st.cache_resource(show_spinner=False)
def load_model_resources():
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    CAREERS = le.classes_.tolist()
    NUM_CLASSES = len(CAREERS)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    config = OptimDistilBertConfig()
    config.num_labels = NUM_CLASSES
    model = DistilBertForCareerPath(config)
    model.load_state_dict(torch.load(CUSTOM_PATH, map_location=DEVICE), strict=False)
    model.to(DEVICE).eval()
    return le, CAREERS, tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Cluster Scatter with UMAP/t-SNE + Convex Hulls
# ─────────────────────────────────────────────────────────────────────────────
def _reduce_dimensions(features: np.ndarray, clusters: np.ndarray) -> pd.DataFrame:
    n_clusters = len(np.unique(clusters))
    n_samples = features.shape[0]
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.8,
            spread=3.0,
            random_state=42,
            metric="euclidean",
        )
        coords = reducer.fit_transform(features)
    except ImportError:
        from sklearn.manifold import TSNE
        perp = max(5, min(50, n_samples // (n_clusters * 2)))
        coords = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=2000,
            learning_rate="auto",
            init="pca",
            random_state=42,
        ).fit_transform(features)
    return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": clusters})

def _spread_clusters(coords_df: pd.DataFrame, spread_factor: float = 5.0) -> pd.DataFrame:
    coords_df = coords_df.copy()
    global_cx = coords_df["x"].mean()
    global_cy = coords_df["y"].mean()
    clusters = sorted(coords_df["cluster"].unique())
    for i, cluster_id in enumerate(clusters):
        mask = coords_df["cluster"] == cluster_id
        cx = coords_df.loc[mask, "x"].mean() - global_cx
        cy = coords_df.loc[mask, "y"].mean() - global_cy
        coords_df.loc[mask, "x"] += cx * (spread_factor - 1) * (1 + i * 0.1)
        coords_df.loc[mask, "y"] += cy * (spread_factor - 1) * (1 + i * 0.1)
    return coords_df

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"

@st.cache_data(show_spinner=False)
def build_cluster_scatter(clustered_path: str, theme: str = "dark") -> go.Figure:
    """
    Builds an enhanced cluster scatter:
    - t-SNE 2D projection
    - Per-cluster neon color from CLUSTER_PALETTE
    - Convex hull boundary per cluster
    - Cluster ID label centred inside each hull
    - Marker size 7, opacity 0.78, white border
    - Legend per cluster
    theme: 'dark' | 'light' — each value is cached separately so theme
    switches produce a correctly-coloured new figure without re-running t-SNE.
    """
    # Local theme tokens — must be local because @st.cache_data returns the
    # same Figure object for repeated (clustered_path, theme) calls; without
    # locals the figure would keep the colours from the very first call.
    _is_dark   = theme == "dark"
    _bg_card   = "#1A1A40" if _is_dark else "#FFFFFF"
    _text_pri  = "#FFFFFF"  if _is_dark else "#1A1A3E"
    _text_sec  = "#9B9BC4"  if _is_dark else "#6B6B9B"
    _border    = "#2A2A5A"  if _is_dark else "#D8DAF0"
    _legend_bg = "rgba(26,26,64,0.85)" if _is_dark else "rgba(255,255,255,0.92)"

    df = pd.read_csv(clustered_path)

    # Parse skills to flat text for t-SNE
    def parse_skills(val):
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return " ".join(str(s).lower() for s in parsed)
        except Exception:
            pass
        return str(val).lower()

    skills_text = df["skills"].fillna("").apply(parse_skills)

    # TF-IDF → t-SNE 2D
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=200, sublinear_tf=True)
    X = vec.fit_transform(skills_text).toarray()

    coords_df = _reduce_dimensions(X, df["cluster"].values)
    coords_df = _spread_clusters(coords_df, spread_factor=5.0)

    df["tsne_x"] = coords_df["x"].values
    df["tsne_y"] = coords_df["y"].values

    # Fix 2: Downsample points per cluster
    MAX_PER_CLUSTER = 300
    df = (
        df.groupby("cluster", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), MAX_PER_CLUSTER), random_state=42))
        .reset_index(drop=True)
    )

    clusters = sorted(df["cluster"].unique())
    career_names = df.groupby("cluster")["career_path_name"].agg(lambda x: x.value_counts().idxmax()).to_dict()

    fig = go.Figure()

    from scipy.spatial import ConvexHull
    import numpy as np

    # 1. Hulls
    for i, cluster_id in enumerate(clusters):
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        mask = df["cluster"] == cluster_id
        pts = df.loc[mask, ["tsne_x", "tsne_y"]].values

        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                fig.add_trace(go.Scatter(
                    x=pts[hull_pts, 0], y=pts[hull_pts, 1],
                    fill="toself",
                    fillcolor=_hex_to_rgba(color, 0.18),
                    line=dict(color=color, width=2.5),
                    mode="lines",
                    hoverinfo="skip",
                    showlegend=False,
                ))
            except Exception:
                pass

        if len(pts) > 0:
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="text",
                text=[f"<b>Cluster {cluster_id}</b>"],
                textfont=dict(color=color, size=13, family="sans-serif", weight="bold"),
                showlegend=False,
                hoverinfo="skip",
            ))

    # 2. Scatter dots
    for i, cid in enumerate(clusters):
        color  = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        mask   = df["cluster"] == cid
        sub    = df[mask]
        label  = career_names.get(cid, f"Cluster {cid}")

        fig.add_trace(go.Scattergl(
            x=sub["tsne_x"], y=sub["tsne_y"],
            mode="markers",
            name=f"C{cid}: {label}",
            marker=dict(
                color=color,
                size=9,
                opacity=0.85,
                line=dict(color="white", width=0.8),
            ),
            hovertemplate=(
                f"<b>Cluster {cid}</b><br>"
                "Job: %{customdata[0]}<br>"
                "Career: %{customdata[1]}<extra></extra>"
            ),
            customdata=sub[["title", "career_path_name"]].values,
        ))

    x_pad = (df["tsne_x"].max() - df["tsne_x"].min()) * 0.1
    y_pad = (df["tsne_y"].max() - df["tsne_y"].min()) * 0.1

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor =_bg_card,           # uses local theme var, not global
        font=dict(color=_text_pri, family="sans-serif"),
        legend=dict(
            bgcolor=_legend_bg,
            bordercolor=_border,
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title="t-SNE dim 1", title_font=dict(color=_text_sec),
                   range=[df["tsne_x"].min()-x_pad, df["tsne_x"].max()+x_pad]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title="t-SNE dim 2", title_font=dict(color=_text_sec),
                   range=[df["tsne_y"].min()-y_pad, df["tsne_y"].max()+y_pad]),
        margin=dict(l=10, r=10, t=10, b=30),
        height=420,
        transition={"duration": 0},
        uirevision="cluster-scatter",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
# ── CHANGE 1: chart layout tokens are derived fresh every render (theme-safe) ─
_CHART_FONT_COLOR   = TEXT_PRI   # dark: #FFFFFF | light: #1A1A3E
_CHART_AXIS_COLOR   = TEXT_SEC   # dark: #9B9BC4 | light: #4A4A7A
_CHART_GRID_COLOR   = BORDER     # dark: #2A2A5A | light: #D8DAF0
_CHART_PLOT_BG      = CARD_BG    # dark: #1A1A40 | light: #FFFFFF

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor =_CHART_PLOT_BG,
    font=dict(color=_CHART_FONT_COLOR, family="sans-serif"),
    margin=dict(l=16, r=16, t=28, b=16),
    height=300,
)


def chart_career_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["career_path_name"].value_counts().reset_index()
    counts.columns = ["career", "count"]
    colors = CLUSTER_PALETTE[:len(counts)]
    fig = go.Figure(go.Bar(
        x=counts["count"], y=counts["career"],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
        ),
        hovertemplate="%{y}: <b>%{x}</b> jobs<extra></extra>",
    ))
    # CHANGE 1: explicit axis tick/label colors so light mode stays readable
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(
            gridcolor=_CHART_GRID_COLOR,
            title="Job postings",
            title_font=dict(color=_CHART_AXIS_COLOR),
            tickfont=dict(color=_CHART_FONT_COLOR),
        ),
        yaxis=dict(
            gridcolor=_CHART_GRID_COLOR,
            tickfont=dict(color=_CHART_FONT_COLOR),
        ),
        bargap=0.35,
    )
    return fig


def chart_top_skills(df: pd.DataFrame) -> go.Figure:
    all_skills = []
    for val in df["skills"].dropna():
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                all_skills.extend([str(s).strip() for s in parsed])
                continue
        except Exception:
            pass
        all_skills.extend([s.strip() for s in str(val).split(",") if s.strip()])

    top = pd.DataFrame(Counter(all_skills).most_common(15), columns=["skill", "count"])
    fig = go.Figure(go.Bar(
        x=top["skill"], y=top["count"],
        marker=dict(
            color=top["count"],
            colorscale=[[0, PURPLE], [0.5, CYAN], [1, PINK]],
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        hovertemplate="<b>%{x}</b>: %{y}<extra></extra>",
    ))
    # CHANGE 1: explicit tick font colors
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(
            tickangle=-35,
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=_CHART_FONT_COLOR),
        ),
        yaxis=dict(
            gridcolor=_CHART_GRID_COLOR,
            title="Occurrences",
            title_font=dict(color=_CHART_AXIS_COLOR),
            tickfont=dict(color=_CHART_FONT_COLOR),
        ),
    )
    return fig


def chart_career_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["career_path_name"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.58,
        marker=dict(
            colors=CLUSTER_PALETTE[:len(counts)],
            line=dict(color=_CHART_PLOT_BG, width=2),
        ),
        textinfo="percent",
        textfont=dict(color=_CHART_FONT_COLOR),  # CHANGE 1: percent labels readable
        hovertemplate="<b>%{label}</b><br>%{value} jobs (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **{**_LAYOUT_BASE, "height": 320},
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=_CHART_FONT_COLOR),  # CHANGE 1: legend text color
            orientation="v",
            x=1.02, y=0.5,
        ),
        annotations=[dict(
            text="<b>Careers</b>",
            font=dict(color=_CHART_FONT_COLOR, size=13),
            x=0.5, y=0.5, showarrow=False,
        )],
    )
    return fig


def chart_cluster_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.groupby(["career_path_name", "cluster"]).size().unstack(fill_value=0)
    # CHANGE 1: heatmap zero-end uses a light-mode-safe neutral rather than CARD_BG
    _heatmap_zero = "#1A1A40" if IS_DARK else "#E8EAF6"
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"C{c}" for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, _heatmap_zero], [0.3, PURPLE], [0.7, CYAN], [1, PINK]],
        hovertemplate="Career: %{y}<br>Cluster: %{x}<br>Count: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=_CHART_FONT_COLOR),  # CHANGE 1: colorbar scale text
            title=dict(font=dict(color=_CHART_AXIS_COLOR)),
        ),
    ))
    fig.update_layout(
        **{**_LAYOUT_BASE, "height": 320},
        xaxis=dict(
            title="Cluster ID",
            title_font=dict(color=_CHART_AXIS_COLOR),
            tickfont=dict(color=_CHART_FONT_COLOR),  # CHANGE 1: C0-C5 labels
        ),
        yaxis=dict(
            title="",
            automargin=True,
            tickfont=dict(color=_CHART_FONT_COLOR),  # CHANGE 1: career name labels
        ),
    )
    return fig


def chart_skills_by_career(df: pd.DataFrame) -> go.Figure:
    """Grouped bar — avg skill_count per career."""
    avg = df.groupby("career_path_name")["skill_count"].mean().sort_values(ascending=False)
    colors = CLUSTER_PALETTE[:len(avg)]
    fig = go.Figure(go.Bar(
        x=avg.index.tolist(),
        y=avg.values.tolist(),
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        hovertemplate="<b>%{x}</b><br>Avg skills: %{y:.1f}<extra></extra>",
    ))
    # CHANGE 1: explicit tick colors
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(
            tickangle=-30,
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=_CHART_FONT_COLOR),
        ),
        yaxis=dict(
            gridcolor=_CHART_GRID_COLOR,
            title="Avg skill count",
            title_font=dict(color=_CHART_AXIS_COLOR),
            tickfont=dict(color=_CHART_FONT_COLOR),
        ),
        bargap=0.3,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
import re

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9+#.,!?-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def predict_career(text: str, model, tokenizer, le, careers):
    cleaned = clean_text(text)
    inputs  = tokenizer(cleaned, return_tensors="pt", truncation=True,
                        padding=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out   = model(inputs["input_ids"], inputs["attention_mask"])
    probs     = F.softmax(out["logits"], dim=-1)[0]
    pred_idx  = probs.argmax().item()
    return careers[pred_idx], float(probs[pred_idx]) * 100, probs.cpu().numpy()


def update_model_comparison(input_text: str):
    from web_app.model_runner import run_all_models
    results = run_all_models(input_text)

    # Table data
    table_data = [
        {
            "Model": r["model"],
            "Prediction": r["prediction"],
            "Confidence": f"{r['confidence']:.1%}" if r["error"] is None else "—",
            "Inference (ms)": r["inference_ms"],
            "Status": "✅" if r["error"] is None else f"❌ {r['error'][:40]}"
        }
        for r in results
    ]

    # Bar chart — inference time comparison
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=[r["model"] for r in results],
        y=[r["inference_ms"] for r in results],
        marker_color=["#e91e8c", "#6366f1", "#f59e0b"],
        text=[f"{r['inference_ms']}ms" for r in results],
        textposition="outside",
        textfont=dict(color=_CHART_FONT_COLOR),
    ))
    _layout = {
        **_LAYOUT_BASE,
        "height": 300,
        "font": dict(color=_CHART_FONT_COLOR),
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
    }
    fig.update_layout(
        **_layout,
        title="Inference Speed Comparison (ms)",
        yaxis_title="Time (ms)",
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=_CHART_FONT_COLOR)),
        yaxis=dict(gridcolor=_CHART_GRID_COLOR, tickfont=dict(color=_CHART_FONT_COLOR)),
    )
    return table_data, fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Career Guidance AI — Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* ── Global reset ────────────────────────────── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {BG} !important;
      color: {TEXT_PRI};
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}
  [data-testid="stHeader"] {{ background: transparent !important; }}
  section[data-testid="stSidebar"] > div:first-child {{
      background-color: {SIDEBAR_BG} !important;
      border-right: 1px solid {BORDER};
  }}

  /* ── Sidebar nav ─────────────────────────────── */
  .sidebar-title {{
      font-size: 1.1rem; font-weight: 700; color: {PINK};
      letter-spacing: 1px; padding: 18px 0 8px 0; text-align: center;
  }}
  .nav-item {{
      display:flex; align-items:center; gap:10px;
      padding: 10px 16px; border-radius: 8px; cursor:pointer;
      color: {TEXT_SEC}; font-size: 0.92rem; margin: 3px 0;
      transition: background 0.18s;
  }}
  .nav-item:hover, .nav-item.active {{
      background: rgba(255,45,120,0.12);
      color: {PINK}; border-left: 3px solid {PINK};
  }}

  /* ── Stat cards ──────────────────────────────── */
  .stat-card {{
      background: {CARD_BG};
      border-radius: 12px;
      padding: 20px 18px 14px 18px;
      border: 1px solid {BORDER};
      position: relative; overflow: hidden;
  }}
  .stat-card::before {{
      content:''; position:absolute; top:0; left:0;
      width:100%; height:3px;
  }}
  .stat-card.pink::before   {{ background: {PINK}; }}
  .stat-card.cyan::before   {{ background: {CYAN}; }}
  .stat-card.purple::before {{ background: {PURPLE}; }}
  .stat-card.orange::before {{ background: {ORANGE}; }}
  .stat-label {{
      font-size: 0.72rem; color: {TEXT_SEC}; letter-spacing: 1px;
      text-transform: uppercase; margin-bottom: 6px;
  }}
  .stat-value {{
      font-size: 2rem; font-weight: 800; line-height: 1;
  }}
  .stat-value.pink   {{ color: {PINK}; }}
  .stat-value.cyan   {{ color: {CYAN}; }}
  .stat-value.purple {{ color: {PURPLE}; }}
  .stat-value.orange {{ color: {ORANGE}; }}
  .stat-icon {{
      font-size: 1.4rem; position: absolute; right: 18px; top: 18px;
      opacity: 0.5;
  }}

  /* ── Section headers ─────────────────────────── */
  .section-title {{
      font-size: 0.72rem; color: {TEXT_SEC}; letter-spacing: 2px;
      text-transform: uppercase; margin: 28px 0 10px 0;
      border-left: 3px solid {CYAN}; padding-left: 10px;
  }}

  /* ── Chart card wrapper ──────────────────────── */
  .chart-card {{
      background: {CARD_BG};
      border-radius: 12px;
      border: 1px solid {BORDER};
      padding: 14px 10px 6px 10px;
  }}
  .chart-title {{
      font-size: 0.82rem; color: {TEXT_SEC}; letter-spacing: 1px;
      text-transform: uppercase; margin-bottom: 8px; padding-left: 6px;
  }}

  /* ── Prediction result ───────────────────────── */
  .pred-badge {{
      background: linear-gradient(135deg, {PINK}22, {PURPLE}22);
      border: 1px solid {PINK};
      border-radius: 10px; padding: 18px 22px;
      text-align: center;
  }}
  .pred-career {{
      font-size: 1.5rem; font-weight: 800; color: {PINK}; margin-bottom: 4px;
  }}
  .pred-conf {{
      font-size: 0.85rem; color: {TEXT_SEC};
  }}

  /* ── Skill tags ──────────────────────────────── */
  .skill-tag {{
      display: inline-block;
      padding: 3px 10px; border-radius: 20px; font-size: 0.78rem;
      margin: 3px 3px;
  }}
  .skill-tag.have  {{ background: {CYAN}22;   border: 1px solid {CYAN};   color: {CYAN}; }}
  .skill-tag.miss  {{ background: {PINK}22;   border: 1px solid {PINK};   color: {PINK}; }}

  /* ── Divider ─────────────────────────────────── */
  hr {{ border-color: {BORDER} !important; }}

  /* ── Plotly chart transparent bg ────────────── */
  .js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {{
      background: transparent !important;
  }}

  /* ── Streamlit overrides ─────────────────────── */
  .stButton > button {{
      background: linear-gradient(135deg, {PINK}, {PURPLE});
      color: white; border: none; border-radius: 8px;
      font-weight: 600; padding: 10px 24px; transition: opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity: 0.85; }}
  .stTextArea textarea {{
      background: {CARD_BG} !important; color: {TEXT_PRI} !important;
      border: 1px solid {BORDER} !important; border-radius: 8px;
  }}
  .stProgress > div > div > div > div {{
      background: linear-gradient(90deg, {PINK}, {CYAN});
  }}
  [data-testid="stMetric"] {{ color: {TEXT_PRI}; }}
  [data-testid="stMetricLabel"] > div {{ color: {TEXT_SEC} !important; }}
  [data-testid="stMetricValue"] > div {{ color: {TEXT_PRI} !important; }}

  /* ── Pill nav buttons (st.radio full override) ────────── */
  div[data-testid="stRadio"] > label {{ display: none !important; }}
  div[data-testid="stRadio"] > div {{
      display: flex !important;
      flex-direction: column !important;
      gap: 4px !important;
  }}
  div[data-testid="stRadio"] label {{
      display: flex !important;
      align-items: center !important;
      gap: 10px !important;
      width: 100% !important;
      padding: 12px 18px !important;
      border-radius: 10px !important;
      border: none !important;
      background: transparent !important;
      color: {TEXT_SEC} !important;
      font-size: 14px !important;
      font-weight: 500 !important;
      cursor: pointer !important;
      transition: background 0.2s, color 0.2s !important;
      margin: 0 !important;
      white-space: nowrap !important;
  }}
  div[data-testid="stRadio"] label:hover {{
      background: rgba(255,45,120,0.12) !important;
      color: {TEXT_PRI} !important;
  }}
  div[data-testid="stRadio"] label > div:first-child {{
      display: none !important;
  }}
  div[data-testid="stRadio"] label:has(input:checked) {{
      background: {PINK} !important;
      color: #ffffff !important;
      font-weight: 600 !important;
  }}
  div[data-testid="stRadio"] label p,
  div[data-testid="stRadio"] label span {{
      color: inherit !important;
  }}

  /* ── CHANGE 2: Theme toggle FAB REMOVED from fixed position (now lives in sidebar) ─── */
  /* ── CHANGE 3: Sidebar hamburger / collapse button — more visible ────────────────── */
  [data-testid="collapsedControl"],
  button[data-testid="stSidebarCollapseButton"],
  section[data-testid="stSidebar"] button[aria-label="Close sidebar"],
  button[aria-label="Open sidebar"],
  button[aria-label="Close sidebar"] {{
      min-width: 44px  !important;
      min-height: 44px !important;
      width: 44px      !important;
      height: 44px     !important;
      border-radius: 10px !important;
      background: rgba(99, 102, 241, 0.15) !important;
      border: 1.5px solid {BORDER} !important;
      box-shadow: 0 2px 8px rgba(0,0,0,0.18) !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      transition: background 0.2s, border-color 0.2s !important;
  }}
  [data-testid="collapsedControl"]:hover,
  button[data-testid="stSidebarCollapseButton"]:hover,
  button[aria-label="Open sidebar"]:hover,
  button[aria-label="Close sidebar"]:hover {{
      background: rgba(99, 102, 241, 0.30) !important;
      border-color: {CYAN} !important;
  }}
  [data-testid="collapsedControl"] svg,
  button[data-testid="stSidebarCollapseButton"] svg,
  button[aria-label="Open sidebar"] svg,
  button[aria-label="Close sidebar"] svg {{
      width: 22px !important;
      height: 22px !important;
      color: {TEXT_PRI} !important;
      fill: {TEXT_PRI} !important;
  }}

  /* ── CHANGE 3: Light-mode text fixes (WCAG AA 4.5:1) ──
     Only injected when IS_DARK is False. Dark mode untouched.
  ─────────────────────────────────────────────────────── */
  {"" if IS_DARK else f"""
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main .block-container {{
      color: {TEXT_PRI} !important;
  }}
  [data-testid="stMarkdownContainer"],
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] span,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3,
  [data-testid="stMarkdownContainer"] h4 {{
      color: {TEXT_PRI} !important;
  }}
  h1, h2, h3, h4, h5, h6 {{ color: {TEXT_PRI} !important; }}
  section[data-testid="stSidebar"] *,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] div {{ color: {TEXT_PRI} !important; }}
  div[data-testid="stRadio"] label {{ color: {TEXT_SEC} !important; }}
  div[data-testid="stRadio"] label:has(input:checked) {{ color: #ffffff !important; }}
  .stat-label {{ color: {TEXT_SEC} !important; }}
  .stat-card {{ box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
  .chart-title {{ color: {TEXT_SEC} !important; }}
  .section-title {{ color: {TEXT_SEC} !important; }}
  .chart-card {{ box-shadow: 0 2px 10px rgba(0,0,0,0.07); }}
  [data-testid="stMetricLabel"] > div {{ color: {TEXT_SEC} !important; }}
  [data-testid="stMetricValue"] > div {{ color: {TEXT_PRI} !important; }}
  [data-testid="stMetricDelta"] > div {{ color: {TEXT_PRI} !important; }}
  .stTextArea label, .stTextArea .stMarkdown {{ color: {TEXT_PRI} !important; }}
  .stTextArea textarea {{ color: {TEXT_PRI} !important; }}
  .stSelectbox label, .stTextInput label {{ color: {TEXT_PRI} !important; }}
  .stRadio label, .stCheckbox label {{ color: {TEXT_PRI} !important; }}
  .pred-conf {{ color: {TEXT_SEC} !important; }}
  .stAlert {{ color: {TEXT_PRI} !important; }}
  .model-cmp-table th, .model-cmp-table td {{ color: {TEXT_PRI} !important; }}
  """}

  /* ── CHANGE 1: Model comparison table styles ─────────────────────────── */
  .model-cmp-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
  }}
  .model-cmp-table th {{
      background: {"#1E1E4A" if IS_DARK else "#EEF0FF"};
      color: {TEXT_PRI};
      padding: 10px 14px;
      text-align: left;
      font-size: 0.75rem;
      letter-spacing: 1px;
      text-transform: uppercase;
      border-bottom: 2px solid {BORDER};
  }}
  .model-cmp-table td {{
      padding: 10px 14px;
      border-bottom: 1px solid {BORDER};
      color: {TEXT_PRI} !important;
  }}
  .model-cmp-table tr:hover td {{
      background: {"rgba(255,45,120,0.06)" if IS_DARK else "rgba(224,0,94,0.05)"};
  }}
  /* CHANGE 1: na-cell tooltip support */
  .na-cell[title] {{ cursor: help; }}
  .best-badge {{
      display: inline-block;
      background: {"rgba(57,255,20,0.18)" if IS_DARK else "rgba(0,160,0,0.12)"};
      color: {"#39FF14" if IS_DARK else "#006600"};
      border: 1px solid {"#39FF14" if IS_DARK else "#00AA00"};
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 700;
      padding: 1px 8px;
      margin-left: 6px;
      vertical-align: middle;
  }}
  .na-cell {{
      color: {TEXT_SEC};
      font-style: italic;
  }}

  /* ── CHANGE 2: Sidebar drawer theme-toggle button ─────────────────────── */
  #sidebar-theme-btn {{
      display: flex;
      align-items: center;
      gap: 10px;
      width: calc(100% - 24px);
      margin: 8px 12px 4px 12px;
      padding: 10px 16px;
      border-radius: 10px;
      border: 1.5px solid {BORDER};
      background: linear-gradient(135deg, {PINK}22, {PURPLE}22);
      color: {TEXT_PRI};
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, border-color 0.2s;
      font-family: inherit;
  }}
  #sidebar-theme-btn:hover {{
      background: linear-gradient(135deg, {PINK}44, {PURPLE}44);
      border-color: {PINK};
  }}
  #sidebar-theme-btn .btn-icon {{
      font-size: 1.1rem;
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load data once at startup
# ─────────────────────────────────────────────────────────────────────────────
stats        = compute_stats(DATA_PATH)          # TASK 1 — live, no static mock
df_jobs      = load_main_df(DATA_PATH)
career_map   = load_json(CAREER_MAPPING_PATH)
cluster_kws  = load_json(CLUSTER_KEYWORDS_PATH)

try:
    df_clustered = load_clustered_df(CLUSTERED_DATA_PATH)
except Exception:
    df_clustered = df_jobs.copy()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Logo ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sidebar-title">🚀 CareerAI</div>
    <hr style="border-color:{BORDER}; margin:4px 0 8px 0;">
    """, unsafe_allow_html=True)

    # ── CHANGE 2: Theme toggle button inside sidebar ─────────────────────────
    _fab_icon_sidebar    = "☀️ Light Mode" if IS_DARK else "🌙 Dark Mode"
    _fab_tooltip_sidebar = "Switch to Light Mode" if IS_DARK else "Switch to Dark Mode"

    if st.button(
        f"{'☀️' if IS_DARK else '🌙'}  {'Light Mode' if IS_DARK else 'Dark Mode'}",
        key="sidebar_theme_btn",
        use_container_width=True,
        help=_fab_tooltip_sidebar,
    ):
        st.session_state["theme"] = "light" if IS_DARK else "dark"
        st.rerun()

    st.markdown(f'<hr style="border-color:{BORDER}; margin:8px 0 8px 0;">', unsafe_allow_html=True)

    page = st.radio(
        label="",
        options=["🏠  Overview", "🔮  Predict & Roadmap", "📊  Analytics", "🗺️  Roadmap"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <hr style="border-color:{BORDER}; margin:12px 0 12px 0;">
    <div style="font-size:0.7rem; color:{TEXT_SEC}; text-align:center; line-height:1.8;">
        Model: <span style="color:{CYAN};">Custom DistilBERT</span><br>
        Data: <span style="color:{PINK};">{stats['total_jobs']:,} jobs</span><br>
        &copy; 2026 Career Guidance AI
    </div>
    """, unsafe_allow_html=True)

# ── CHANGE 2: Old fixed FAB removed — toggle is now in the sidebar above ──────




# ─────────────────────────────────────────────────────────────────────────────
# Helper — stat card HTML
# ─────────────────────────────────────────────────────────────────────────────
def stat_card(label: str, value, color: str, icon: str) -> str:
    val_str = f"{value:,}" if isinstance(value, int) else str(value)
    return f"""
    <div class="stat-card {color}">
      <div class="stat-icon">{icon}</div>
      <div class="stat-label">{label}</div>
      <div class="stat-value {color}">{val_str}</div>
    </div>
    """


def chart_card(title: str, fig):
    st.markdown(f'<div class="chart-card"><div class="chart-title">{title}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    # ── Hero banner ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {SIDEBAR_BG} 0%, #1a1040 60%, #0d0d2b 100%);
        border-radius: 14px; padding: 36px 40px 28px 40px;
        border: 1px solid {BORDER}; margin-bottom: 28px;
        position: relative; overflow: hidden;">
      <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
          border-radius:50%;background:radial-gradient({PINK}33,transparent 70%);"></div>
      <div style="position:absolute;bottom:-60px;left:30%;width:160px;height:160px;
          border-radius:50%;background:radial-gradient({CYAN}22,transparent 70%);"></div>
      <h1 style="color:{TEXT_PRI};font-size:2rem;margin:0 0 8px 0;position:relative;">
          🤖 Career Guidance AI
      </h1>
      <p style="color:{TEXT_SEC};font-size:1rem;max-width:600px;line-height:1.7;margin:0;position:relative;">
          Custom DistilBERT deep-learning platform — predict IT careers from skills,
          analyse your skill gaps, and explore workforce cluster trends in real time.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat cards row (Task 1 — live from compute_stats) ───────────────────
    st.markdown('<div class="section-title">Live Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Total Job Postings",  stats["total_jobs"],     "pink",   "💼"), unsafe_allow_html=True)
    c2.markdown(stat_card("Career Categories",   stats["total_careers"],  "cyan",   "🎯"), unsafe_allow_html=True)
    c3.markdown(stat_card("Skill Clusters",       stats["total_clusters"], "purple", "🔵"), unsafe_allow_html=True)
    c4.markdown(stat_card("AI Model",             stats["model_name"],     "orange", "🤖"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick charts row ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Quick Insights</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2], gap="medium")

    with col_l:
        chart_card("Job Postings by Career", chart_career_distribution(df_jobs))

    with col_r:
        chart_card("Career Share", chart_career_donut(df_jobs))

    # ── Feature cards row ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Platform Features</div>', unsafe_allow_html=True)
    fa, fb, fc = st.columns(3, gap="medium")

    feat_style = f"background:{CARD_BG};border-radius:12px;padding:24px 20px;border:1px solid {BORDER};height:100%;"

    with fa:
        st.markdown(f"""<div style="{feat_style}">
            <div style="font-size:1.6rem;margin-bottom:10px;">🔮</div>
            <h4 style="color:{CYAN};margin:0 0 8px 0;">Predict Career</h4>
            <p style="color:{TEXT_SEC};font-size:0.88rem;line-height:1.6;margin:0;">
                Enter your skill set — Custom DistilBERT returns your best-fit IT career
                with confidence score in under 1 second.
            </p></div>""", unsafe_allow_html=True)

    with fb:
        st.markdown(f"""<div style="{feat_style}">
            <div style="font-size:1.6rem;margin-bottom:10px;">🔍</div>
            <h4 style="color:{PINK};margin:0 0 8px 0;">Skill Gap Analysis</h4>
            <p style="color:{TEXT_SEC};font-size:0.88rem;line-height:1.6;margin:0;">
                Compare your skills against cluster-derived benchmarks and see
                exactly which skills you're missing.
            </p></div>""", unsafe_allow_html=True)

    with fc:
        st.markdown(f"""<div style="{feat_style}">
            <div style="font-size:1.6rem;margin-bottom:10px;">📊</div>
            <h4 style="color:{PURPLE};margin:0 0 8px 0;">Cluster Analytics</h4>
            <p style="color:{TEXT_SEC};font-size:0.88rem;line-height:1.6;margin:0;">
                Deep-dive into job market cluster distributions, skill frequency charts,
                and interactive t-SNE scatter with convex-hull boundaries.
            </p></div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT & ROADMAP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict & Roadmap":
    st.markdown(f"<h2 style='color:{TEXT_PRI};margin-bottom:4px;'>🔮 Career Prediction</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_SEC};margin-bottom:24px;'>Enter your skills — the AI will predict your ideal IT career and analyse your skill gaps.</p>", unsafe_allow_html=True)

    try:
        le, CAREERS, tokenizer, model = load_model_resources()
        model_loaded = True
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        model_loaded = False

    if model_loaded:
        user_input = st.text_area(
            "Your skills (comma-separated or free text):",
            placeholder="e.g. Python, React, Docker, AWS, SQL, Kubernetes …",
            height=110,
            key="predict_user_input",
        )

        if st.button("🚀 Analyse & Predict", use_container_width=False):
            if len(user_input.strip()) < 3:
                st.warning("Please enter at least a few skills.")
            else:
                with st.spinner("AI is thinking …"):
                    career, conf, probs_arr = predict_career(user_input, model, tokenizer, le, CAREERS)
                    live_table_data, live_fig = update_model_comparison(user_input)

                # Persist all result state in session_state
                cluster_id_tmp = str(career_map.get(career, -1))
                req_skills_tmp = set(cluster_kws.get(cluster_id_tmp, []))
                user_skills_tmp = set(s.strip().lower() for s in re.split(r"[,\s]+", user_input) if s.strip())
                matched_tmp = user_skills_tmp & req_skills_tmp
                missing_tmp = req_skills_tmp - user_skills_tmp

                st.session_state["predicted_career"]  = career
                st.session_state["predict_conf"]      = conf
                st.session_state["predict_probs_arr"] = probs_arr
                st.session_state["predict_req_skills"]= req_skills_tmp
                st.session_state["predict_matched"]   = matched_tmp
                st.session_state["predict_missing"]   = missing_tmp
                st.session_state["missing_skills"]    = sorted(missing_tmp)
                st.session_state["live_table_data"]   = live_table_data
                st.session_state["live_fig"]          = live_fig

        # ── Display results from session_state ─────────────────────────
        if st.session_state.get("predicted_career") and st.session_state.get("predict_probs_arr") is not None:
            career     = st.session_state["predicted_career"]
            conf       = st.session_state["predict_conf"]
            probs_arr  = st.session_state["predict_probs_arr"]
            req_skills = st.session_state.get("predict_req_skills", set())
            matched    = st.session_state.get("predict_matched", set())
            missing    = st.session_state.get("predict_missing", set())

            # Result badge
            st.markdown(f"""
            <div class="pred-badge">
              <div class="pred-career">🎯 {career}</div>
              <div class="pred-conf">Confidence: <b style="color:{PINK};">{conf:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(conf), text=f"{conf:.1f}% confidence")

            # Top-5 probability bar
            st.markdown(f'<div class="section-title" style="margin-top:20px;">Top Career Probabilities</div>', unsafe_allow_html=True)
            top5_idx  = np.argsort(probs_arr)[::-1][:5]
            top5_df   = pd.DataFrame({
                "Career": [CAREERS[i] for i in top5_idx],
                "Probability": [float(probs_arr[i]) * 100 for i in top5_idx],
            })
            prob_fig = go.Figure(go.Bar(
                x=top5_df["Probability"], y=top5_df["Career"],
                orientation="h",
                marker=dict(color=[PINK, CYAN, PURPLE, LIME, ORANGE]),
                hovertemplate="%{y}: <b>%{x:.2f}%</b><extra></extra>",
            ))
            # CHANGE 1: explicit tick colors so Y-axis career names are visible in light mode
            prob_fig.update_layout(
                **{**_LAYOUT_BASE, "height": 220},
                xaxis=dict(
                    title="%",
                    gridcolor=_CHART_GRID_COLOR,
                    tickfont=dict(color=_CHART_FONT_COLOR),
                    title_font=dict(color=_CHART_AXIS_COLOR),
                ),
                yaxis=dict(
                    gridcolor="rgba(0,0,0,0)",
                    tickfont=dict(color=_CHART_FONT_COLOR),  # CHANGE 1: career name labels
                ),
            )
            st.plotly_chart(prob_fig, use_container_width=True, config={"displayModeBar": False})

            # Skill gap
            st.markdown(f'<div class="section-title">Skill Gap Analysis</div>', unsafe_allow_html=True)

            if req_skills:
                score = len(matched) / len(req_skills) * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Required Skills", len(req_skills))
                m2.metric("You Have", len(matched))
                m3.metric("Match Score", f"{score:.0f}%")

                ca, cb = st.columns(2, gap="medium")
                with ca:
                    st.markdown(f"<div style='color:{CYAN};font-weight:600;margin-bottom:6px;'>✅ Skills You Have</div>", unsafe_allow_html=True)
                    tags = " ".join(f'<span class="skill-tag have">{s}</span>' for s in sorted(matched)) if matched else "<i style='color:#555;'>None matched</i>"
                    st.markdown(tags, unsafe_allow_html=True)
                with cb:
                    st.markdown(f"<div style='color:{PINK};font-weight:600;margin-bottom:6px;'>❌ Skills to Learn</div>", unsafe_allow_html=True)
                    tags = " ".join(f'<span class="skill-tag miss">{s}</span>' for s in sorted(missing)) if missing else "<i style='color:#555;'>All covered!</i>"
                    st.markdown(tags, unsafe_allow_html=True)
            else:
                st.info("No cluster keyword data available for this career.")

        # ── Model Comparison (moved from Analytics) ─────────────────────────
        st.markdown(f'<div class="section-title" style="margin-top:28px;">Model Comparison</div>', unsafe_allow_html=True)

        # Show Live prediction comparison if available
        if st.session_state.get("live_table_data") is not None and st.session_state.get("live_fig") is not None:
            _live_headers = ["Model", "Predicted Career", "Confidence", "Inference Time", "Status"]
            _header_html = "".join(f"<th>{h}</th>" for h in _live_headers)
            _rows_html = ""
            for r in st.session_state["live_table_data"]:
                _model_display = "Custom DistilBERT" if r["Model"] == "custom_distilbert" else ("BERT Base" if r["Model"] == "bert_base" else "ML Baseline")
                _rows_html += f"""<tr>
                    <td><b>{_model_display}</b></td>
                    <td>{r['Prediction']}</td>
                    <td>{r['Confidence']}</td>
                    <td>{r['Inference (ms)']} ms</td>
                    <td>{r['Status']}</td>
                </tr>"""
                
            st.markdown(f"""
            <div class="chart-card" style="padding:16px 10px; margin-bottom: 20px;">
              <div class="chart-title">⚡ Live 3-Model Prediction Results</div>
              <div style="overflow-x:auto;">
                <table class="model-cmp-table">
                  <thead><tr>{_header_html}</tr></thead>
                  <tbody>{_rows_html}</tbody>
                </table>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(st.session_state["live_fig"], use_container_width=True, config={"displayModeBar": False})
            st.markdown("<hr style='border:0;border-top:1px solid rgba(255,255,255,0.1);margin:24px 0;'>", unsafe_allow_html=True)
        else:
            st.info("💡 Enter your skills in the box above and click 'Analyse & Predict' to run live 3-model inference comparison.")

        st.markdown(f'<div class="chart-title" style="font-size:1.1rem;margin-bottom:12px;">📊 Baseline Model Evaluation (Static Training Metrics)</div>', unsafe_allow_html=True)

        _metrics_rows = load_model_metrics()

        def _parse_pct(v: str):
            """Return float from '92.34%' or '92.34', else None."""
            if v == "N/A":
                return None
            try:
                return float(v.rstrip("%"))
            except (ValueError, AttributeError):
                return None

        _numeric_cols = ["accuracy", "f1", "precision", "recall"]
        _best_per_col = {}
        for _col in _numeric_cols:
            _vals = [(_parse_pct(r[_col]), i) for i, r in enumerate(_metrics_rows)]
            _numeric_vals = [(v, i) for v, i in _vals if v is not None]
            _best_per_col[_col] = max(_numeric_vals, key=lambda x: x[0])[1] if _numeric_vals else None

        _header_labels = ["Model", "Accuracy", "F1 Score", "Precision", "Recall", "Model Size"]
        _col_keys      = ["Model", "accuracy", "f1", "precision", "recall", "model_size"]
        _header_html   = "".join(f"<th>{h}</th>" for h in _header_labels)
        _rows_html     = ""
        for _i, _row in enumerate(_metrics_rows):
            _cells = ""
            for _j, _key in enumerate(_col_keys):
                _val = _row[_key]
                _is_best = (
                    _key in _best_per_col and
                    _best_per_col[_key] == _i and
                    _val != "N/A"
                )
                _badge = '<span class="best-badge">✅ Best</span>' if _is_best else ""
                if _val == "N/A":
                    _na_cls = ' class="na-cell" title="Metrics file not found in model directory"'
                    _cells += f"<td{_na_cls}>—{_badge}</td>"
                else:
                    _cells += f"<td>{_val}{_badge}</td>"
            _rows_html += f"<tr>{_cells}</tr>"

        # ── Grouped bar chart: Accuracy, F1, Precision, Recall ──────────────────
        _chart_models = [r["Model"] for r in _metrics_rows]
        _metric_defs  = [
            ("Accuracy",  "accuracy",  CYAN),
            ("F1 Score",  "f1",        PINK),
            ("Precision", "precision", PURPLE),
            ("Recall",    "recall",    LIME),
        ]
        _any_data = any(_parse_pct(r["accuracy"]) is not None for r in _metrics_rows)
        if _any_data:
            _cmp_fig = go.Figure()
            for _label, _key, _color in _metric_defs:
                _yvals = [_parse_pct(r[_key]) for r in _metrics_rows]
                _cmp_fig.add_trace(go.Bar(
                    name=_label,
                    x=_chart_models,
                    y=[v if v is not None else 0 for v in _yvals],
                    text=[f"{v:.1f}%" if v is not None else "—" for v in _yvals],
                    textposition="outside",
                    textfont=dict(color=_CHART_FONT_COLOR),
                    marker=dict(color=_color, line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
                    hovertemplate=f"%{{x}}<br>{_label}: <b>%{{y:.2f}}%</b><extra></extra>",
                ))
            _layout = {
                **_LAYOUT_BASE,
                "height": 340,
                "font": dict(color=_CHART_FONT_COLOR),
            }
            _cmp_fig.update_layout(
                **_layout,
                barmode="group",
                bargap=0.2,
                bargroupgap=0.05,
                xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=_CHART_FONT_COLOR)),
                yaxis=dict(
                    gridcolor=_CHART_GRID_COLOR,
                    title="Score (%)",
                    title_font=dict(color=_CHART_AXIS_COLOR),
                    tickfont=dict(color=_CHART_FONT_COLOR),
                    range=[0, 115],
                ),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=_CHART_FONT_COLOR),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                ),
            )
            st.plotly_chart(_cmp_fig, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  Analytics":
    st.markdown(f"<h2 style='color:{TEXT_PRI};margin-bottom:4px;'>📊 Analytics Dashboard</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_SEC};margin-bottom:20px;'>Live exploration of the {stats['total_jobs']:,}-job dataset across {stats['total_careers']} careers and {stats['total_clusters']} clusters.</p>", unsafe_allow_html=True)

    # ── Stat cards ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Total Jobs",    stats["total_jobs"],     "pink",   "💼"), unsafe_allow_html=True)
    c2.markdown(stat_card("Careers",       stats["total_careers"],  "cyan",   "🎯"), unsafe_allow_html=True)
    c3.markdown(stat_card("Clusters",      stats["total_clusters"], "purple", "🔵"), unsafe_allow_html=True)
    c4.markdown(stat_card("Model",         stats["model_name"],     "orange", "🤖"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Career bar + Donut ────────────────────────────────────────────
    st.markdown('<div class="section-title">Career Distribution</div>', unsafe_allow_html=True)
    r1l, r1r = st.columns([3, 2], gap="medium")
    with r1l:
        chart_card("Job Postings by Career", chart_career_distribution(df_jobs))
    with r1r:
        chart_card("Career Share (Donut)", chart_career_donut(df_jobs))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Top Skills + Skill count per career ───────────────────────────
    st.markdown('<div class="section-title">Skill Intelligence</div>', unsafe_allow_html=True)
    r2l, r2r = st.columns(2, gap="medium")
    with r2l:
        chart_card("Top 15 Skills Across All Jobs", chart_top_skills(df_jobs))
    with r2r:
        chart_card("Avg Skill Count by Career", chart_skills_by_career(df_jobs))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Heatmap + Cluster Scatter (TASK 2) ────────────────────────────
    st.markdown('<div class="section-title">Cluster Analysis</div>', unsafe_allow_html=True)
    r3l, r3r = st.columns([2, 3], gap="medium")

    with r3l:
        chart_card("Career × Cluster Heatmap", chart_cluster_heatmap(df_jobs))

    with r3r:
        st.markdown('<div class="chart-card"><div class="chart-title">t-SNE Cluster Scatter (Convex Hulls)</div>', unsafe_allow_html=True)
        with st.spinner("Computing t-SNE projection …"):
            scatter_fig = build_cluster_scatter(
                CLUSTERED_DATA_PATH,
                st.session_state.get("theme", "dark"),  # UI Change 2: theme-keyed cache
            )
        st.plotly_chart(scatter_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ROADMAP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️  Roadmap":
    st.markdown(f"<h2 style='color:{TEXT_PRI};margin-bottom:4px;'>🗺️ Learning Roadmap</h2>", unsafe_allow_html=True)

    career    = st.session_state.get("predicted_career", None)
    missing   = st.session_state.get("missing_skills", [])

    if career is None:
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;
            padding:32px;text-align:center;color:{TEXT_SEC};">
            <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
            No prediction yet. Go to <b style="color:{PINK};">🔮 Predict & Roadmap</b> first,
            enter your skills and click Analyse.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {PINK}33;border-radius:12px;
            padding:20px 24px; margin-bottom:20px;">
            <span style="color:{TEXT_SEC};font-size:0.8rem;">Your predicted career</span><br>
            <span style="color:{PINK};font-size:1.6rem;font-weight:800;">{career}</span>
        </div>
        """, unsafe_allow_html=True)

        cluster_id  = str(career_map.get(career, -1))
        all_skills  = cluster_kws.get(cluster_id, [])
        n           = len(all_skills)
        third       = max(1, n // 3)

        beginner     = all_skills[:third]
        intermediate = all_skills[third:2*third]
        advanced     = all_skills[2*third:]

        def level_card(level, skills, color, icon, timeline):
            tags = " ".join(f'<span class="skill-tag have" style="border-color:{color};color:{color};background:{color}18;">{s}</span>'
                            for s in skills) if skills else "<i style='color:#555;'>—</i>"
            st.markdown(f"""
            <div style="background:{CARD_BG};border:1px solid {color}44;border-radius:12px;
                padding:20px;margin-bottom:14px;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <span style="font-size:1.3rem;">{icon}</span>
                <span style="color:{color};font-weight:700;font-size:1rem;">{level}</span>
                <span style="color:{TEXT_SEC};font-size:0.78rem;margin-left:auto;">⏱ {timeline}</span>
              </div>
              <div>{tags}</div>
            </div>
            """, unsafe_allow_html=True)

        level_card("Beginner — Foundation",     beginner,     CYAN,   "🟢", "0 – 3 months")
        level_card("Intermediate — Growth",     intermediate, PURPLE, "🟡", "3 – 8 months")
        level_card("Advanced — Specialisation", advanced,     PINK,   "🔴", "8 – 18 months")

        if missing:
            st.markdown(f'<div class="section-title">Priority Skills to Learn (your gaps)</div>', unsafe_allow_html=True)
            tags = " ".join(f'<span class="skill-tag miss">{s}</span>' for s in missing)
            st.markdown(f'<div style="background:{CARD_BG};border-radius:10px;padding:16px;border:1px solid {BORDER};">{tags}</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="margin-top:40px;border-top:1px solid {BORDER};padding-top:12px;
    text-align:center;color:{TEXT_SEC};font-size:0.75rem;">
    Career Guidance AI &nbsp;·&nbsp; Custom DistilBERT &nbsp;·&nbsp; &copy; 2026
</div>
""", unsafe_allow_html=True)
