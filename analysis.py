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


def calculate_vif(frame: pd.DataFrame) -> pd.Series:
    """Calculate VIF without an extra statsmodels dependency."""
    values: dict[str, float] = {}
    for column in frame.columns:
        y_feature = frame[column]
        other_features = frame.drop(columns=column)
        if other_features.empty:
            values[column] = 1.0
            continue

        r2 = LinearRegression().fit(other_features, y_feature).score(
            other_features, y_feature
        )
        denominator = 1.0 - r2
        values[column] = np.inf if denominator <= 1e-12 else 1.0 / denominator

    return pd.Series(values, name="VIF").sort_values(ascending=False)


def reduce_by_vif(
    frame: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Iteratively remove the feature with the highest VIF above threshold."""
    reduced = frame.copy()
    removed: list[dict[str, float]] = []

    while reduced.shape[1] > 1:
        vif = calculate_vif(reduced)
        highest_feature = str(vif.index[0])
        highest_vif = float(vif.iloc[0])
        if highest_vif <= threshold:
            break

        removed.append({"feature": highest_feature, "vif_before_removal": highest_vif})
        reduced = reduced.drop(columns=highest_feature)

    return reduced, removed


def evaluate_model(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[LinearRegression, np.ndarray, dict[str, float | int | str]]:
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics: dict[str, float | int | str] = {
        "model": name,
        "features": X_train.shape[1],
        "R2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
    }
    return model, predictions, metrics


def save_correlation_matrix(correlation: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.index)), correlation.index)

    for row in range(len(correlation.index)):
        for col in range(len(correlation.columns)):
            value = correlation.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation matrix of numeric features")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_residual_plot(predictions: np.ndarray, residuals: np.ndarray, path: Path) -> tuple[float, float]:
    slope, intercept = np.polyfit(predictions, residuals, 1)
    x_line = np.linspace(predictions.min(), predictions.max(), 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predictions, residuals, alpha=0.75)
    ax.axhline(0, linewidth=1)
    ax.plot(x_line, slope * x_line + intercept, linestyle="--", label="Linear trend")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals for the reduced linear regression model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    corr = float(np.corrcoef(predictions, residuals)[0, 1])
    return float(slope), corr


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-set comparison for linear regression")
    parser.add_argument("--data", default=None, help="Optional CSV file path; built-in Diabetes data is used by default")
    parser.add_argument("--target", default="target", help="Target column")
    parser.add_argument("--vif-threshold", type=float, default=10.0, help="VIF cutoff")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test-set share")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="results", help="Directory for tables and plots")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data:
        data_path = Path(args.data)
        data = pd.read_csv(data_path)
        data_label = str(data_path)
    else:
        data = load_diabetes(as_frame=True).frame.copy()
        data_label = "sklearn.datasets.load_diabetes"

    if args.target not in data.columns:
        raise ValueError(f"Target column '{args.target}' not found in {data_label}")

    numeric = data.select_dtypes(include=np.number)
    if args.target not in numeric.columns:
        raise ValueError("The target must be numeric for linear regression")

    numeric = numeric.dropna().copy()
    y = numeric[args.target]
    X = numeric.drop(columns=args.target)
    if X.empty:
        raise ValueError("No numeric features found")

    # 1. Baseline: every available numeric feature.
    correlation = X.corr()
    initial_vif = calculate_vif(X)

    # 2. Remove redundant features by iterative VIF elimination.
    X_reduced, removed = reduce_by_vif(X, args.vif_threshold)
    final_vif = calculate_vif(X_reduced)

    # Use exactly the same row split for a fair baseline/reduced comparison.
    train_index, test_index = train_test_split(
        X.index,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]

    _, baseline_predictions, baseline_metrics = evaluate_model(
        "Baseline: all numeric features",
        X.loc[train_index],
        X.loc[test_index],
        y_train,
        y_test,
    )
    reduced_model, reduced_predictions, reduced_metrics = evaluate_model(
        "Reduced: VIF-selected features",
        X_reduced.loc[train_index],
        X_reduced.loc[test_index],
        y_train,
        y_test,
    )

    metrics = pd.DataFrame([baseline_metrics, reduced_metrics])
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    initial_vif.rename_axis("feature").reset_index().to_csv(output_dir / "vif_initial.csv", index=False)
    final_vif.rename_axis("feature").reset_index().to_csv(output_dir / "vif_final.csv", index=False)
    correlation.to_csv(output_dir / "correlation_matrix.csv")

    save_correlation_matrix(correlation, output_dir / "correlation_matrix.png")

    residuals = y_test.to_numpy() - reduced_predictions
    residual_slope, residual_prediction_corr = save_residual_plot(
        reduced_predictions,
        residuals,
        output_dir / "residuals.png",
    )

    coefficients = pd.DataFrame(
        {
            "feature": X_reduced.columns,
            "coefficient": reduced_model.coef_,
        }
    )
    coefficients.to_csv(output_dir / "final_coefficients.csv", index=False)

    summary = {
        "dataset": data_label,
        "dataset_rows": int(len(numeric)),
        "initial_features": list(X.columns),
        "removed_by_vif": removed,
        "final_features": list(X_reduced.columns),
        "vif_threshold": args.vif_threshold,
        "residual_trend_slope": residual_slope,
        "residual_prediction_correlation": residual_prediction_corr,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Initial VIF ===")
    print(initial_vif.round(3).to_string())
    print("\nRemoved:", removed or "none")
    print("\n=== Final VIF ===")
    print(final_vif.round(3).to_string())
    print("\n=== Metrics ===")
    print(metrics.round(4).to_string(index=False))
    print("\nFinal features:", ", ".join(X_reduced.columns))
    print(f"Residual trend slope: {residual_slope:.4f}")
    print(f"Correlation(prediction, residual): {residual_prediction_corr:.4f}")
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
