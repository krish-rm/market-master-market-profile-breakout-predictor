import sys
from pathlib import Path

# Ensure project root on sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

from scripts.predict import PredictionService


def compute_feature_importance(model, feature_names: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Return (feature_names_sorted, importance_sorted).
    Supports tree-based models with feature_importances_ and linear models via |coef_|.
    """
    # Tree-based models (RandomForest, XGBoost's sklearn API, LightGBM's sklearn API)
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    # Linear models (e.g., LogisticRegression)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        # Handle multiclass: average absolute coefficients across classes
        if coef.ndim > 1:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
    else:
        raise ValueError(
            "This model does not expose feature_importances_ or coef_. "
            "Consider training a tree-based model or add SHAP for interpretability."
        )

    # Guard against length mismatches
    if len(feature_names) != len(importance):
        # Try to align by truncation to the shortest length
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]

    # Sort ascending for a nicer horizontal bar chart
    order = np.argsort(importance)
    features_sorted = [feature_names[i] for i in order]
    importance_sorted = importance[order]
    return features_sorted, importance_sorted


def build_bar_chart(features: List[str], importances: np.ndarray, title: str = "Feature Importance"):
    fig = go.Figure(
        data=go.Bar(
            x=importances,
            y=features,
            orientation="h",
            marker_color="#1f77b4",
            text=[f"{v:.3f}" for v in importances],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, len(features) * 28),
        template="plotly_dark",
        margin=dict(l=150, r=40, t=60, b=40),
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize model feature importance.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory containing best_model.pkl and feature_names.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/feature_importance.html",
        help="Output HTML file path for the chart.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only top-N most important features (0 = show all).",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print sorted feature importances to stdout as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model and metadata using existing service
    service = PredictionService(args.model_dir)
    model = service.model
    feature_names = service.feature_names

    features_sorted, importance_sorted = compute_feature_importance(model, feature_names)

    # Optionally reduce to top-N
    if args.top and args.top > 0:
        features_sorted = features_sorted[-args.top :]
        importance_sorted = importance_sorted[-args.top :]

    fig = build_bar_chart(features_sorted, importance_sorted, title="Model Feature Importance")
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved feature importance chart to {output_path}")

    if args.print_json:
        as_json = [
            {"feature": f, "importance": float(v)} for f, v in zip(features_sorted, importance_sorted)
        ]
        print(json.dumps(as_json, indent=2))

    # Try to open in a browser for convenience (non-blocking)
    try:
        import webbrowser

        webbrowser.open_new_tab(output_path.resolve().as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()


