"""
ml_models.py
============
Machine-learning and forecasting utilities for the prediction page.

Models implemented
------------------
1. Linear Regression        – simple trend extrapolation
2. Polynomial Regression    – captures non-linear trend
3. Prophet-style decomposition via sklearn + numpy (no Facebook Prophet dep)
4. Rolling-window forecast  – last-N-day average projection
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─── Colour constants ────────────────────────────────────────────────────────
ACTUAL_COLOR    = "#2563EB"
PREDICT_COLOR   = "#DC2626"
FUTURE_COLOR    = "#059669"
INTERVAL_COLOR  = "rgba(5,150,105,0.15)"

FRIENDLY = {
    "Children_Apprehended": "Children Apprehended",
    "Children_in_CBP":      "Children in CBP Custody",
    "Children_Transferred": "Children Transferred",
    "Children_in_HHS":      "Children in HHS Care",
    "Children_Discharged":  "Children Discharged",
}


def _layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color="#1E293B"), x=0.01),
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Inter, sans-serif", color="#475569"),
        xaxis=dict(showgrid=False, linecolor="#CBD5E1"),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=70, b=50),
        hovermode="x unified",
    )


def _date_to_ordinal(dates: pd.Series) -> np.ndarray:
    """Convert pandas datetime series to integer day ordinals for regression."""
    return dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)


# ─── 1. Linear Regression Forecast ──────────────────────────────────────────
def linear_regression_forecast(
    df: pd.DataFrame,
    column: str,
    forecast_days: int = 30,
) -> tuple[go.Figure, dict]:
    """
    Fits a linear regression model on the full dataset and forecasts
    `forecast_days` into the future.

    Returns:
        fig    : Plotly figure
        metrics: dict with MAE, RMSE, R²
    """
    df = df.dropna(subset=["Date", column]).sort_values("Date").reset_index(drop=True)
    X = _date_to_ordinal(df["Date"])
    y = df[column].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Build future dates
    last_date   = df["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    X_future    = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    y_future    = model.predict(X_future)

    # Residual std for prediction interval
    residuals = y - y_pred
    std_resid = residuals.std()

    # Metrics
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    label = FRIENDLY.get(column, column)

    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(
        x=df["Date"], y=y,
        mode="lines", name="Actual",
        line=dict(color=ACTUAL_COLOR, width=1.5),
    ))

    # In-sample fitted line
    fig.add_trace(go.Scatter(
        x=df["Date"], y=y_pred,
        mode="lines", name="Fitted (Linear)",
        line=dict(color=PREDICT_COLOR, width=2, dash="dash"),
    ))

    # Forecast + confidence band
    upper = y_future + 1.96 * std_resid
    lower = y_future - 1.96 * std_resid
    lower = np.maximum(lower, 0)  # no negative children

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor=INTERVAL_COLOR,
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=y_future,
        mode="lines+markers", name="Forecast",
        line=dict(color=FUTURE_COLOR, width=2.5),
        marker=dict(size=5),
    ))

    fig.update_layout(**_layout(f"Linear Regression Forecast – {label}"))

    return fig, {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R²": round(r2, 4)}


# ─── 2. Polynomial Regression Forecast ──────────────────────────────────────
def polynomial_regression_forecast(
    df: pd.DataFrame,
    column: str,
    degree: int = 3,
    forecast_days: int = 30,
) -> tuple[go.Figure, dict]:
    """
    Fits a polynomial regression model and forecasts `forecast_days` ahead.
    Degree 2-4 typically captures trend inflections better than linear.
    """
    df = df.dropna(subset=["Date", column]).sort_values("Date").reset_index(drop=True)
    X = _date_to_ordinal(df["Date"])
    y = df[column].values

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr",   LinearRegression()),
    ])
    model.fit(X, y)
    y_pred = model.predict(X)

    last_date    = df["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    X_future     = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    y_future     = np.maximum(model.predict(X_future), 0)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    label = FRIENDLY.get(column, column)
    fig   = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=y,
        mode="lines", name="Actual",
        line=dict(color=ACTUAL_COLOR, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=y_pred,
        mode="lines", name=f"Fitted (Poly deg={degree})",
        line=dict(color=PREDICT_COLOR, width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=y_future,
        mode="lines+markers", name="Forecast",
        line=dict(color=FUTURE_COLOR, width=2.5),
        marker=dict(size=5),
    ))

    fig.update_layout(**_layout(f"Polynomial Regression (deg={degree}) Forecast – {label}"))

    return fig, {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R²": round(r2, 4)}


# ─── 3. Rolling-Window Average Forecast ─────────────────────────────────────
def rolling_average_forecast(
    df: pd.DataFrame,
    column: str,
    window: int = 30,
    forecast_days: int = 30,
) -> tuple[go.Figure, dict]:
    """
    Projects the next `forecast_days` as a flat line equal to the rolling
    mean of the last `window` observations.  Simple but highly interpretable.
    """
    df = df.dropna(subset=["Date", column]).sort_values("Date").reset_index(drop=True)
    y = df[column].values
    baseline = df[column].tail(window).mean()

    last_date    = df["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    y_future     = np.full(forecast_days, baseline)

    label = FRIENDLY.get(column, column)
    fig   = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=y,
        mode="lines", name="Actual",
        line=dict(color=ACTUAL_COLOR, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=y_future,
        mode="lines", name=f"{window}-Day Rolling Avg Forecast",
        line=dict(color=FUTURE_COLOR, width=2.5, dash="dashdot"),
    ))

    fig.update_layout(**_layout(f"Rolling Average ({window}-Day) Forecast – {label}"))
    return fig, {"Projected Value": round(baseline, 1), "Window": window}


# ─── 4. Feature importance from multi-variable regression ───────────────────
def feature_importance(df: pd.DataFrame, target: str, features: list) -> go.Figure:
    """
    Fits a multiple linear regression with the given features predicting
    the target, then displays normalised absolute coefficients as a bar chart.
    Useful for explaining which variables drive children in HHS care, etc.
    """
    valid_features = [f for f in features if f in df.columns and f != target]
    if not valid_features:
        return go.Figure()

    sub = df[valid_features + [target]].dropna()
    X   = sub[valid_features].values
    y   = sub[target].values

    # Standardise X to make coefficients comparable
    X_std   = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    model   = LinearRegression().fit(X_std, y)
    coefs   = np.abs(model.coef_)
    coefs   = coefs / coefs.sum()   # normalise to sum = 1

    labels = [FRIENDLY.get(f, f) for f in valid_features]
    colors = ["#2563EB", "#7C3AED", "#059669", "#D97706"]

    fig = go.Figure(go.Bar(
        x=coefs,
        y=labels,
        orientation="h",
        marker=dict(color=colors[:len(labels)]),
        text=[f"{v:.1%}" for v in coefs],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(
            text=f"Feature Importance → {FRIENDLY.get(target, target)}",
            font=dict(size=16, color="#1E293B"), x=0.01,
        ),
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Inter, sans-serif", color="#475569"),
        xaxis=dict(title="Normalised |Coefficient|", tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=80, t=60, b=50),
        showlegend=False,
    )
    return fig
