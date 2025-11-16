"""
Streamlit Dashboard for Market Profile Breakout Predictor

Run with: streamlit run scripts/dashboard.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scripts.predict import PredictionService

# Page configuration
st.set_page_config(
    page_title="Market Master - Breakout Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_service' not in st.session_state:
    try:
        st.session_state.prediction_service = PredictionService(model_dir="models")
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.model_error = str(e)

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


@st.cache_data
def load_metrics():
    """Load model performance metrics."""
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def create_confusion_matrix_plot(cm):
    """Create confusion matrix visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Breakout', 'Predicted: Breakout'],
        y=['Actual: No Breakout', 'Actual: Breakout'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    fig.update_layout(
        title="Confusion Matrix",
        width=500,
        height=400
    )
    return fig


def create_metrics_radar_chart(metrics):
    """Create radar chart for model metrics."""
    categories = ['ROC AUC', 'Precision', 'Recall', 'F1-Score']
    values = [
        metrics['auc'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Model Performance',
        line_color='#1f77b4'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Metrics",
        width=500,
        height=400
    )
    return fig


def get_feature_importance(prediction_service):
    """Extract feature importance from the model."""
    try:
        model = prediction_service.model
        feature_names = prediction_service.feature_names
        
        # Check if model has feature_importances_ attribute (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            # Create DataFrame for better visualization
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            return importance_df
        else:
            # For models without feature importance (e.g., Logistic Regression)
            return None
    except Exception as e:
        return None


def create_feature_importance_chart(importance_df):
    """Create feature importance visualization."""
    if importance_df is None:
        return None
    
    fig = go.Figure(data=go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker_color='#1f77b4',
        text=[f'{imp:.3f}' for imp in importance_df['importance']],
        textposition='auto'
    ))
    fig.update_layout(
        title="Feature Importance (Random Forest)",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(400, len(importance_df) * 30),
        width=700
    )
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Market Master – Breakout Predictor Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔮 Predictions", "📊 Model Performance", "📈 Feature Explorer"]
        )
        
        st.divider()
        
        # Model Status
        st.header("Model Status")
        if st.session_state.model_loaded:
            st.success("✅ Model Loaded")
            st.info(f"Model: Random Forest")
        else:
            st.error("❌ Model Not Loaded")
            if 'model_error' in st.session_state:
                st.error(f"Error: {st.session_state.model_error}")
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Predictions":
        show_predictions_page()
    elif page == "📊 Model Performance":
        show_performance_page()
    elif page == "📈 Feature Explorer":
        show_feature_explorer_page()


def show_home_page():
    """Display home page with overview."""
    st.header("Welcome to Market Master Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What is Market Profile?")
        st.markdown("""
        Market Profile is a trading methodology that visualizes price action 
        and volume distribution to identify:
        - **Point of Control (POC)**: Price with most trading volume
        - **Value Area High (VAH)**: Upper boundary of fair value
        - **Value Area Low (VAL)**: Lower boundary of fair value
        
        Our ML model predicts whether the next session will break above VAH.
        """)
    
    with col2:
        st.subheader("📊 Model Overview")
        metrics = load_metrics()
        if metrics:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("ROC AUC", f"{metrics['auc']:.3f}")
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col_b:
                st.metric("Recall", f"{metrics['recall']:.3f}")
                st.metric("F1-Score", f"{metrics['f1']:.3f}")
        else:
            st.warning("Metrics not available")
    
    st.divider()
    
    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. **Make Predictions**: Go to the Predictions page to input Market Profile features
    2. **View Performance**: Check Model Performance page for detailed metrics
    3. **Explore Features**: Use Feature Explorer to understand Market Profile concepts
    """)


def _load_latest_defaults():
    """Load latest engineered features as defaults and latest close price."""
    defaults = {
        "session_poc": 95000.0,
        "session_vah": 97000.0,
        "session_val": 93000.0,
        "va_range_width": 4000.0,
        "balance_flag": 1,
        "volume_imbalance": 0.55,
        "one_day_return": 0.012,
        "three_day_return": 0.030,
        "atr_14": 1500.0,
        "rsi_14": 60.5,
        "session_volume": 1_500_000.0,
    }
    latest_close = None
    try:
        df = pd.read_parquet("data/processed/market_profile.parquet").sort_values("date")
        row = df.iloc[-1]
        for k in defaults.keys():
            if k in row.index and pd.notna(row[k]):
                defaults[k] = float(row[k])
    except Exception:
        pass
    # Try to load latest close from cached raw data
    try:
        raw = pd.read_parquet("data/raw/BTC-USD_2y_1h.parquet")
        latest_close = float(raw['close'].iloc[-1])
        latest_ts = str(raw.index.max())
    except Exception:
        latest_ts = None
    return defaults, latest_close, latest_ts


def show_predictions_page():
    """Display predictions page with input form."""
    st.header("🔮 Make Predictions")
    
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Please ensure models are available in the models/ directory.")
        return

    # Load latest defaults
    defaults, latest_close, latest_ts = _load_latest_defaults()
    if latest_close is not None:
        st.info(f"Latest cached price: {latest_close:.2f} USD at {latest_ts}")
    st.button("↻ Use latest engineered features", on_click=lambda: None)
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Market Profile Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            session_poc = st.number_input(
                "Session POC (Point of Control)",
                min_value=0.0,
                value=defaults["session_poc"],
                step=100.0,
                help="Price level with most trading volume"
            )
            session_vah = st.number_input(
                "Session VAH (Value Area High)",
                min_value=0.0,
                value=defaults["session_vah"],
                step=100.0,
                help="Upper boundary of value area (70th percentile)"
            )
            session_val = st.number_input(
                "Session VAL (Value Area Low)",
                min_value=0.0,
                value=defaults["session_val"],
                step=100.0,
                help="Lower boundary of value area (30th percentile)"
            )
            va_range_width = st.number_input(
                "VA Range Width",
                min_value=0.0,
                value=defaults["va_range_width"],
                step=10.0,
                help="Width of value area (VAH - VAL)"
            )
        
        with col2:
            balance_flag = st.selectbox(
                "Balance Flag",
                [0, 1],
                index=1 if int(defaults["balance_flag"]) == 1 else 0,
                help="1 if balanced session, 0 if unbalanced"
            )
            volume_imbalance = st.slider(
                "Volume Imbalance",
                min_value=0.0,
                max_value=1.0,
                value=float(np.clip(defaults["volume_imbalance"], 0.0, 1.0)),
                step=0.01,
                help="Ratio of upside volume to total volume"
            )
            one_day_return = st.number_input(
                "1-Day Return",
                min_value=-1.0,
                max_value=1.0,
                value=defaults["one_day_return"],
                step=0.001,
                format="%.3f",
                help="1-day return percentage"
            )
            three_day_return = st.number_input(
                "3-Day Return",
                min_value=-1.0,
                max_value=1.0,
                value=defaults["three_day_return"],
                step=0.001,
                format="%.3f",
                help="3-day return percentage"
            )
        
        with col3:
            atr_14 = st.number_input(
                "ATR (14-period)",
                min_value=0.0,
                value=defaults["atr_14"],
                step=10.0,
                help="Average True Range - volatility measure"
            )
            rsi_14 = st.slider(
                "RSI (14-period)",
                min_value=0.0,
                max_value=100.0,
                value=float(np.clip(defaults["rsi_14"], 0.0, 100.0)),
                step=0.1,
                help="Relative Strength Index - momentum oscillator"
            )
            session_volume = st.number_input(
                "Session Volume",
                min_value=0.0,
                value=defaults["session_volume"],
                step=10000.0,
                format="%.0f",
                help="Total trading volume for the session"
            )
        
        submitted = st.form_submit_button("🔮 Predict Breakout", use_container_width=True)
    
    # Make prediction
    if submitted:
        try:
            features = {
                "session_poc": session_poc,
                "session_vah": session_vah,
                "session_val": session_val,
                "va_range_width": va_range_width,
                "balance_flag": balance_flag,
                "volume_imbalance": volume_imbalance,
                "one_day_return": one_day_return,
                "three_day_return": three_day_return,
                "atr_14": atr_14,
                "rsi_14": rsi_14,
                "session_volume": session_volume
            }
            
            result = st.session_state.prediction_service.predict_single(features)
            prediction = result['prediction']
            probability = result['probability']
            confidence = max(probability, 1 - probability)
            
            # Display prediction
            st.divider()
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>✅ Breakout Predicted</h2>
                        <p><strong>Probability:</strong> {probability:.1%}</p>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p>The model predicts the next session will break above the Value Area High (VAH).</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h2>❌ No Breakout Predicted</h2>
                        <p><strong>Probability:</strong> {probability:.1%}</p>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p>The model predicts the next session will NOT break above the Value Area High (VAH).</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Prediction", "Breakout" if prediction == 1 else "No Breakout")
                st.metric("Probability", f"{probability:.1%}")
            
            with col3:
                st.metric("Confidence", f"{confidence:.1%}")
                # Confidence indicator
                if confidence > 0.7:
                    st.success("High Confidence")
                elif confidence > 0.6:
                    st.info("Moderate Confidence")
                else:
                    st.warning("Low Confidence")
            
            # Visualize probability
            fig = go.Figure(data=go.Bar(
                x=['No Breakout', 'Breakout'],
                y=[1 - probability, probability],
                marker_color=['#ff6b6b', '#51cf66'],
                text=[f'{(1-probability):.1%}', f'{probability:.1%}'],
                textposition='auto'
            ))
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add to history
            st.session_state.prediction_history.append({
                **features,
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence
            })
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    # Prediction history
    if st.session_state.prediction_history:
        st.divider()
        st.subheader("📜 Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(
            history_df[['session_poc', 'session_vah', 'prediction', 'probability', 'confidence']].tail(10),
            use_container_width=True
        )
        
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()


def show_performance_page():
    """Display model performance metrics."""
    st.header("📊 Model Performance")
    
    metrics = load_metrics()
    if not metrics:
        st.error("Performance metrics not available. Please train the model first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC AUC", f"{metrics['auc']:.3f}", help="Area Under ROC Curve")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}", help="True Positives / (True Positives + False Positives)")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}", help="True Positives / (True Positives + False Negatives)")
    with col4:
        st.metric("F1-Score", f"{metrics['f1']:.3f}", help="Harmonic mean of Precision and Recall")
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        cm = np.array(metrics['confusion_matrix'])
        fig_cm = create_confusion_matrix_plot(cm)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Calculate additional metrics from confusion matrix
        tn, fp = cm[0]
        fn, tp = cm[1]
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        # Radar chart
        fig_radar = create_metrics_radar_chart(metrics)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.divider()
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['ROC AUC', 'Precision', 'Recall', 'F1-Score', 'Model Type'],
        'Value': [
            f"{metrics['auc']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            metrics['model_name'].replace('_', ' ').title()
        ]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Confusion matrix breakdown
    st.subheader("Confusion Matrix Breakdown")
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=['Actual: No Breakout', 'Actual: Breakout'],
        columns=['Predicted: No Breakout', 'Predicted: Breakout']
    )
    st.dataframe(cm_df, use_container_width=True)


def show_feature_explorer_page():
    """Display feature explorer with explanations."""
    st.header("📈 Market Profile Feature Explorer")
    
    st.markdown("""
    ### Understanding Market Profile Features
    
    Market Profile features help identify where price and volume cluster during a trading session.
    """)
    
    # Feature categories
    tab1, tab2, tab3 = st.tabs(["Market Profile", "Technical Indicators", "Feature Importance"])
    
    with tab1:
        st.subheader("Market Profile Features")
        
        features_info = {
            "session_poc": {
                "name": "Point of Control (POC)",
                "description": "The price level where the most trading volume occurred during the session.",
                "interpretation": "Represents the 'fair value' price where buyers and sellers agreed most.",
                "range": "Any positive price value"
            },
            "session_vah": {
                "name": "Value Area High (VAH)",
                "description": "The upper boundary of the Value Area, containing 70% of session volume.",
                "interpretation": "Prices above VAH are considered 'expensive'. Breakouts above VAH indicate strong buying pressure.",
                "range": "Typically 0.5-2% above POC"
            },
            "session_val": {
                "name": "Value Area Low (VAL)",
                "description": "The lower boundary of the Value Area, containing 70% of session volume.",
                "interpretation": "Prices below VAL are considered 'cheap'. Prices below VAL indicate selling pressure.",
                "range": "Typically 0.5-2% below POC"
            },
            "va_range_width": {
                "name": "Value Area Range Width",
                "description": "The difference between VAH and VAL.",
                "interpretation": "Wider ranges suggest more volatility and uncertainty. Narrow ranges suggest consolidation.",
                "range": "Typically 1-5% of price"
            },
            "balance_flag": {
                "name": "Balance Flag",
                "description": "Indicates if the session was balanced (POC in middle 50% of price range).",
                "interpretation": "1 = Balanced (consolidation), 0 = Unbalanced (directional move).",
                "range": "0 or 1"
            },
            "volume_imbalance": {
                "name": "Volume Imbalance",
                "description": "Ratio of upside volume (volume when price moved up) to total volume.",
                "interpretation": ">0.5 = Bullish pressure, <0.5 = Bearish pressure.",
                "range": "0.0 to 1.0"
            },
            "session_volume": {
                "name": "Session Volume",
                "description": "Total trading volume for the session.",
                "interpretation": "Higher volume indicates more participation and conviction.",
                "range": "Any positive value"
            }
        }
        
        for key, info in features_info.items():
            with st.expander(f"📊 {info['name']}"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Interpretation:** {info['interpretation']}")
                st.write(f"**Typical Range:** {info['range']}")
    
    with tab2:
        st.subheader("Technical Indicators")
        
        tech_info = {
            "atr_14": {
                "name": "Average True Range (ATR-14)",
                "description": "14-period Average True Range measures volatility by averaging price ranges.",
                "interpretation": "Higher ATR = more volatility = larger expected price movements.",
                "range": "Typically 1-5% of price"
            },
            "rsi_14": {
                "name": "Relative Strength Index (RSI-14)",
                "description": "14-period RSI measures momentum and overbought/oversold conditions.",
                "interpretation": "<30 = Oversold (buying opportunity), >70 = Overbought (potential reversal), 30-70 = Normal.",
                "range": "0 to 100"
            },
            "one_day_return": {
                "name": "1-Day Return",
                "description": "Percentage change from yesterday's close to today's close.",
                "interpretation": "Positive = upward momentum, Negative = downward momentum.",
                "range": "Typically -10% to +10%"
            },
            "three_day_return": {
                "name": "3-Day Return",
                "description": "Percentage change from 3 days ago to today's close.",
                "interpretation": "Measures medium-term trend direction.",
                "range": "Typically -20% to +20%"
            }
        }
        
        for key, info in tech_info.items():
            with st.expander(f"📈 {info['name']}"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Interpretation:** {info['interpretation']}")
                st.write(f"**Typical Range:** {info['range']}")
    
    with tab3:
        st.subheader("Feature Importance")
        
        if st.session_state.model_loaded:
            try:
                importance_df = get_feature_importance(st.session_state.prediction_service)
                
                if importance_df is not None:
                    # Create visualization
                    fig = create_feature_importance_chart(importance_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display as table
                    st.subheader("Importance Scores")
                    st.dataframe(
                        importance_df.sort_values('importance', ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Most Important", importance_df.loc[importance_df['importance'].idxmax(), 'feature'])
                    with col2:
                        st.metric("Max Importance", f"{importance_df['importance'].max():.4f}")
                    with col3:
                        st.metric("Avg Importance", f"{importance_df['importance'].mean():.4f}")
                else:
                    st.info("""
                    **Note:** This model type doesn't provide feature importance scores.
                    Feature importance is available for tree-based models (Random Forest, XGBoost, etc.).
                    """)
            except Exception as e:
                st.error(f"Error loading feature importance: {e}")
        else:
            st.warning("Model not loaded. Cannot display feature importance.")


if __name__ == "__main__":
    main()

