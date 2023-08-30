import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    r2score = r2_score(actual, pred)
    return print("RMSE:", rmse, "\n", "MAE:", mae, "\n", "MSE:", mse, "\n", "R2_SCORE: ", r2score)

