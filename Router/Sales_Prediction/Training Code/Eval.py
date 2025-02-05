from Efficiency import ClothingSalesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Training import df

predictor = ClothingSalesPredictor()
def evaluate_model(predictor, df, target='Sale_QTY'):
    y_actual = df[target]
    y_pred = predictor.predict(df)
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)
    mean_target = y_actual.mean()
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Context: The MAE is {mae:.2f}, which is approximately {mae / mean_target * 100:.1f}% of the average target value ({mean_target:.2f}). Lower values indicate better performance.\n")

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Context: RMSE is {rmse:.2f}, representing the average magnitude of errors. Compare it to the target range to understand its significance.\n")

    print(f"R-squared (R²): {r2:.3f}")
    print(f"Context: An R² of {r2:.3f} means the model explains {r2 * 100:.1f}% of the variance in the target variable. Higher values are better.\n")
    residuals = y_actual - y_pred
    print(f"Residual Mean: {residuals.mean():.2f}")
    print("Context: The residual mean should ideally be close to 0. Skewness indicates systematic prediction errors.\n")
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30, color='blue')
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
    plt.legend()
    plt.show()
    print("Context: Residuals should ideally be symmetrically distributed around 0. Skewness or heavy tails suggest potential bias in predictions.\n")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.6, color='green')
    plt.plot([y_actual.min(), 1500], [y_actual.min(), 1500], 'r--', lw=2, label="Ideal Predictions")
    plt.title("Prediction vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()
    print("Context: Points should lie close to the red diagonal line. Deviations indicate prediction errors.\n")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color='purple')
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()
    print("Context: No clear pattern should appear in residuals vs. predicted values. Patterns may indicate model inadequacies.\n")
    print("Evaluation complete.\n")
    plt.figure(figsize=(8, 5))
    # stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot for Residuals")
    plt.show()
    print("Context: Q-Q plot checks if residuals are normally distributed. Deviations indicate potential bias.")
evaluate_model(predictor,df)
y_actual = df['Sale_QTY']
y_pred = predictor.predict(df)
plt.plot(range(len(y_actual)), y_actual, label='Actual')
plt.plot(range(len(y_pred)), y_pred, label='Predicted')
plt.xlabel('Data Points')
plt.ylabel('Target Variable')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()
y_actual_cumsum = y_actual.cumsum()
y_pred_cumsum = y_pred.cumsum()
plt.plot(range(len(y_actual_cumsum)), y_actual_cumsum, label='Actual Cumulative')
plt.plot(range(len(y_pred_cumsum)), y_pred_cumsum, label='Predicted Cumulative')
plt.xlabel('Data Points')
plt.ylabel('Cumulative Target Variable')
plt.title('Actual vs. Predicted Cumulative Values')
plt.legend()
plt.show()
def count_predictions_in_ranges(y_pred, ranges=[0.8, 0.9, 1, 1.1, 1.2, 1.3]):
  counts = {}
  for i in range(len(ranges) - 1):
    lower_bound = ranges[i]
    upper_bound = ranges[i + 1]
    count = np.sum((y_pred >= lower_bound) & (y_pred <= upper_bound))
    counts[f'{lower_bound}-{upper_bound}'] = count
  return counts
prediction_counts = count_predictions_in_ranges(y_pred)
print(prediction_counts)