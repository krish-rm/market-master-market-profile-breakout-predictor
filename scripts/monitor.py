import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1D distributions."""
    # Replace inf/nan
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    # Ensure at least 2 bins
    if breaks.size < 2:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=breaks)
    actual_counts, _ = np.histogram(actual, bins=breaks)
    expected_perc = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, 1)
    actual_perc = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, 1)
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


def build_report_fig(df: pd.DataFrame, metrics: dict, psi_value: Optional[float]) -> go.Figure:
    rows = 2
    cols = 2
    fig = go.Figure()

    # Row 1 Col 1: Probabilities over time
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["probability"], mode="lines+markers",
        name="Probability", line=dict(color="#1f77b4")
    ))
    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Probability", range=[0, 1])

    # Row 1 Col 2: Histogram of probabilities
    fig.add_trace(go.Histogram(
        x=df["probability"], name="Probability Dist", marker_color="#ff7f0e", opacity=0.7
    ))

    # Row 2 Col 1: If labels present, show rolling accuracy (window=50)
    if "true_label" in df.columns and df["true_label"].notna().any():
        valid = df.dropna(subset=["true_label"])
        preds = (valid["probability"] >= 0.5).astype(int)
        rolling_acc = (preds == valid["true_label"]).rolling(50, min_periods=5).mean()
        fig.add_trace(go.Scatter(
            x=valid["timestamp"], y=rolling_acc, mode="lines", name="Rolling Accuracy (50)",
            line=dict(color="#2ca02c")
        ))

    # Row 2 Col 2: Metrics text box
    text_lines = [f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))]
    if psi_value is not None and np.isfinite(psi_value):
        text_lines.append(f"PSI (prob.): {psi_value:.4f}")
    fig.add_annotation(
        xref="paper", yref="paper", x=1.02, y=0.5, showarrow=False,
        text="<br>".join(text_lines), align="left", bordercolor="#ccc", borderwidth=1, bgcolor="#111", font=dict(color="#fff")
    )

    fig.update_layout(
        title="Model Monitoring Report",
        template="plotly_dark",
        height=700
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Model performance monitoring and drift report.")
    parser.add_argument("--log", type=str, default="logs/predictions.csv", help="Predictions log CSV")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline predictions CSV for drift comparison")
    parser.add_argument("--output", type=str, default="reports/monitor_report.html", help="Output HTML report")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Predictions log not found: {log_path}")

    df = pd.read_csv(log_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    # Compute basic metrics if labels present
    metrics = {}
    if "true_label" in df.columns and df["true_label"].notna().any():
        valid = df.dropna(subset=["true_label"])
        y_true = valid["true_label"].astype(int).values
        y_prob = np.clip(valid["probability"].values, 0.0, 1.0)
        y_pred = (y_prob >= 0.5).astype(int)
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Drift vs baseline (probability distribution)
    psi_value = None
    if args.baseline and Path(args.baseline).exists():
        baseline = pd.read_csv(args.baseline)
        psi_value = compute_psi(
            expected=baseline.get("probability", pd.Series(dtype=float)).values,
            actual=df.get("probability", pd.Series(dtype=float)).values,
            bins=10
        )

    # Build report
    output_path = Path(args.output)
    parent_dir = output_path.parent
    # Handle case where parent path exists as a file (e.g., a file named 'reports')
    if parent_dir.exists() and parent_dir.is_file():
        fallback_dir = Path("reports_out")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"Parent path '{parent_dir}' is a file. Writing report to '{fallback_dir}'.")
        output_path = fallback_dir / output_path.name
        parent_dir = fallback_dir
    # Ensure directory exists
    parent_dir.mkdir(parents=True, exist_ok=True)
    fig = build_report_fig(df, metrics, psi_value)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved monitoring report to {output_path}")

    # Save metrics as JSON alongside HTML
    metrics_path = output_path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump({"metrics": metrics, "psi_probability": psi_value}, f, indent=2)
    print(f"Saved monitoring summary to {metrics_path}")


if __name__ == "__main__":
    main()


