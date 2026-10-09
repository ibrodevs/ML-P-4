from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def calculate_vif(X: pd.DataFrame) -> pd.Series:
    """Calculate VIF for each feature without requiring statsmodels."""
    values: dict[str, float] = {}
    for column in X.columns:
        y_feature = X[column]
        other = X.drop(columns=column)
        if other.empty:
            values[column] = 1.0
            continue
        r2 = LinearRegression().fit(other, y_feature).score(other, y_feature)
        values[column] = np.inf if 1.0 - r2 <= 1e-12 else 1.0 / (1.0 - r2)
    return pd.Series(values, name="VIF").sort_values(ascending=False)


def reduce_by_vif(X: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Remove the highest-VIF feature until every VIF is <= threshold."""
    reduced = X.copy()
    removed: list[dict[str, float]] = []
    while reduced.shape[1] > 1:
        vif = calculate_vif(reduced)
        feature, value = str(vif.index[0]), float(vif.iloc[0])
        if value <= threshold:
            break
        removed.append({"feature": feature, "vif_before_removal": value})
        reduced = reduced.drop(columns=feature)
    return reduced, removed


def evaluate(name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
             y_train: pd.Series, y_test: pd.Series):
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "model": name,
        "features": X_train.shape[1],
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    }
    return model, pred, metrics


def plot_correlation(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for row in range(len(corr.index)):
        for col in range(len(corr.columns)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation matrix of numeric features")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_residuals(pred: np.ndarray, residuals: np.ndarray, path: Path) -> tuple[float, float]:
    slope, intercept = np.polyfit(pred, residuals, 1)
    x_line = np.linspace(pred.min(), pred.max(), 100)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred, residuals, alpha=0.75)
    ax.axhline(0, linewidth=1)
    ax.plot(x_line, slope * x_line + intercept, linestyle="--", label="Linear trend")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals for the reduced linear regression model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    corr = float(np.corrcoef(pred, residuals)[0, 1])
    return float(slope), corr


def load_data(csv_path: str | None, target: str) -> tuple[pd.DataFrame, pd.Series, str]:
    if csv_path:
        data = pd.read_csv(csv_path)
        label = csv_path
    else:
        data = load_diabetes(as_frame=True).frame.copy()
        label = "sklearn.datasets.load_diabetes"

    numeric = data.select_dtypes(include=np.number).dropna().copy()
    if target not in numeric.columns:
        raise ValueError(f"Numeric target column '{target}' was not found in {label}")
    X = numeric.drop(columns=target)
    if X.empty:
        raise ValueError("No numeric features found")
    return X, numeric[target], label


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare feature sets for linear regression")
    parser.add_argument("--data", default=None, help="Optional CSV; Diabetes is used by default")
    parser.add_argument("--target", default="target")
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    X, y, dataset_label = load_data(args.data, args.target)

    # Baseline uses all available numeric features.
    corr = X.corr()
    initial_vif = calculate_vif(X)

    # Remove multicollinear features with iterative VIF elimination.
    X_reduced, removed = reduce_by_vif(X, args.vif_threshold)
    final_vif = calculate_vif(X_reduced)

    # Identical rows in train/test for a fair comparison.
    train_idx, test_idx = train_test_split(
        X.index, test_size=args.test_size, random_state=args.random_state
    )
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    _, _, baseline_metrics = evaluate(
        "Baseline: all numeric features",
        X.loc[train_idx], X.loc[test_idx], y_train, y_test,
    )
    reduced_model, reduced_pred, reduced_metrics = evaluate(
        "Reduced: VIF-selected features",
        X_reduced.loc[train_idx], X_reduced.loc[test_idx], y_train, y_test,
    )

    metrics = pd.DataFrame([baseline_metrics, reduced_metrics])
    metrics.to_csv(output / "metrics.csv", index=False)
    initial_vif.rename_axis("feature").reset_index().to_csv(output / "vif_initial.csv", index=False)
    final_vif.rename_axis("feature").reset_index().to_csv(output / "vif_final.csv", index=False)
    corr.to_csv(output / "correlation_matrix.csv")
    pd.DataFrame({"feature": X_reduced.columns, "coefficient": reduced_model.coef_}).to_csv(
        output / "final_coefficients.csv", index=False
    )

    plot_correlation(corr, output / "correlation_matrix.svg")
    residuals = y_test.to_numpy() - reduced_pred
    slope, residual_corr = plot_residuals(reduced_pred, residuals, output / "residuals.svg")

    summary = {
        "dataset": dataset_label,
        "dataset_rows": int(len(X)),
        "initial_features": list(X.columns),
        "removed_by_vif": removed,
        "final_features": list(X_reduced.columns),
        "vif_threshold": args.vif_threshold,
        "residual_trend_slope": slope,
        "residual_prediction_correlation": residual_corr,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Initial VIF ===")
    print(initial_vif.round(3).to_string())
    print("\nRemoved:", removed or "none")
    print("\n=== Final VIF ===")
    print(final_vif.round(3).to_string())
    print("\n=== Metrics ===")
    print(metrics.round(4).to_string(index=False))
    print("\nFinal features:", ", ".join(X_reduced.columns))
    print(f"Residual trend slope: {slope:.4f}")
    print(f"Correlation(prediction, residual): {residual_corr:.4f}")


if __name__ == "__main__":
    main()
