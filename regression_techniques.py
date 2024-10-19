import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def save_plot(fig, filename):
    plt.show()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Linear Regression
def plot_linear_regression():
    np.random.seed(0)
    X = np.random.rand(100, 1)
    y = 2 + 3 * X + np.random.randn(100, 1) * 0.1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Data')
    ax.plot(X_test, y_pred, color='red', label='Prediction')
    ax.set_title('Linear Regression')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    
    save_plot(fig, 'linear_regression.png')

    print("Linear Regression:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# Polynomial Regression
def plot_polynomial_regression():
    X = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * X) + np.random.randn(100, 1) * 0.1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    degree = 3
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Data')
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    ax.plot(X_plot, model.predict(X_plot), color='red', label='Prediction')
    ax.set_title(f'Polynomial Regression (degree={degree})')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    
    save_plot(fig, 'polynomial_regression.png')

    print("\nPolynomial Regression:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# Ridge and Lasso Regression
def ridge_lasso_regression():
    X = np.random.randn(100, 3)
    X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1
    y = 2 + 3 * X[:, 0] + 4 * X[:, 1] + 5 * X[:, 2] + np.random.randn(100) * 0.1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)

    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)

    print("\nRidge Regression:")
    for i, coef in enumerate(ridge_model.coef_):
        print(f"Feature {i+1}: {coef:.4f}")
    print(f"MSE: {mean_squared_error(y_test, ridge_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, ridge_pred):.4f}")

    print("\nLasso Regression:")
    for i, coef in enumerate(lasso_model.coef_):
        print(f"Feature {i+1}: {coef:.4f}")
    print(f"MSE: {mean_squared_error(y_test, lasso_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, lasso_pred):.4f}")

# Random Forest Regression
def plot_random_forest():
    X = np.random.rand(100, 5)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2]**2 + X[:, 3]**3 + np.exp(X[:, 4])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nRandom Forest Regression:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Feature Importances (Random Forest)")
    ax.bar(range(X.shape[1]), importances[indices])
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels([f"Feature {i+1}" for i in indices])
    plt.tight_layout()
    save_plot(fig, 'random_forest_importances.png')

if __name__ == "__main__":
    plot_linear_regression()
    plot_polynomial_regression()
    ridge_lasso_regression()
    plot_random_forest()
    print("All visualizations have been saved as PNG files.")