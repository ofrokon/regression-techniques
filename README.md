# Regression Techniques: When to Use Which Model

This repository contains Python scripts demonstrating various regression techniques, from simple Linear Regression to more complex models like Random Forests. It accompanies the Medium post "Regression Techniques: When to Use Which Model".

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Visualizations](#visualizations)
4. [Techniques Covered](#techniques-covered)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

To run these scripts, you need Python 3.6 or later. Follow these steps to set up your environment:

1. Clone this repository:
   ```
   git clone https://github.com/ofrokon/regression-techniques.git
   cd regression-techniques
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To generate all visualizations and see the performance metrics, run:

```
python regression_techniques.py
```

This will create PNG files for visualizations and print performance metrics in the console.

## Visualizations

This script generates the following visualizations:

1. `linear_regression.png`: Demonstrates simple linear regression
2. `polynomial_regression.png`: Shows polynomial regression fitting a non-linear relationship
3. `random_forest_importances.png`: Displays feature importances from Random Forest Regression

## Techniques Covered

1. Linear Regression
2. Polynomial Regression
3. Ridge Regression
4. Lasso Regression
5. Random Forest Regression

Each technique is explained in detail in the accompanying Medium post, including:
- When to use each technique
- Pros and cons
- Python implementation using scikit-learn
- Performance metrics and interpretation

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your suggested changes. If you're planning to make significant changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).

---

For a detailed explanation of these regression techniques and their applications, check out the accompanying Medium post: [Regression Techniques: When to Use Which Model](https://medium.com/@mroko001/regression-techniques-when-to-use-which-model-96b36e859b46)

For questions or feedback, please open an issue in this repository.
