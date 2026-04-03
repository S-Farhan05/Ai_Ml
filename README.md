# ML Projects - EDA, Regression & Classification

A collection of three machine learning projects covering data exploration, regression, and classification using Python, pandas, scikit-learn, and visualization libraries.

---

## Project 1: Exploring and Visualizing the Iris Dataset

### Task Objective
Load, inspect, and visualize a dataset to understand data trends and distributions.

### Dataset Used
**Iris Dataset** — Classic multi-class dataset containing measurements of 150 iris flowers across 3 species (*setosa*, *versicolor*, *virginica*). Features include sepal length, sepal width, petal length, and petal width.

### Models Applied
No predictive models — focused on **Exploratory Data Analysis (EDA)**:
- Descriptive statistics via `.info()` and `.describe()`
- Scatter plots for feature relationships
- Histograms for value distributions
- Box plots for outlier detection

### Key Results & Findings
- Clear separation of *setosa* from the other two species in feature space
- Strong correlations observed between petal length/width and flower species
- Visualization revealed natural clustering, confirming the dataset's suitability for classification tasks
- No significant outliers detected within individual species groups

---

## Project 2: Predict Future Stock Prices (Short-Term)

### Task Objective
Use historical stock market data to predict the next day's closing price.

### Dataset Used
**Stock Market Data** — Historical stock prices fetched via the **yfinance** library (e.g., Apple, Tesla). Features include Open, High, Low, Volume, Adj Close, and Close prices over time.

### Models Applied
- **Linear Regression** — Baseline model for predicting closing price
- **Random Forest Regressor** — Non-linear alternative for capturing complex patterns

### Key Results & Findings
- The model captures the general trend of stock prices over short-term windows
- Features like Open and High prices show the strongest correlation with the Close price
- Accuracy varies with market volatility — performs better on stable, trending stocks
- Visual comparison of actual vs. predicted prices highlights the model's ability to follow price direction, though exact value prediction remains limited due to market randomness

---

## Project 3: Heart Disease Prediction

### Task Objective
Build a model to predict whether a person is at risk of heart disease based on their health data.

### Dataset Used
**Heart Disease UCI Dataset** — Medical dataset with 920 patient records across 4 datasets (Cleveland, Hungary, VA Long Beach, Switzerland). Contains 16 features including age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, and more. Target variable (`num`) indicates severity of heart disease (0–4), converted to binary classification (0 = no disease, 1+ = disease).

### Models Applied
- **Logistic Regression** — Primary classification model with L2 regularization
- **StandardScaler** — Feature scaling applied to ensure convergence and fair feature comparison
- Evaluation metrics: **Accuracy, Confusion Matrix, ROC-AUC Curve**

### Key Results & Findings
| Metric | Value |
|--------|-------|
| Accuracy | ~90% |
| ROC-AUC | ~0.94 |
| Dataset (after cleaning) | 299 samples (from 920, after dropping NaN) |

- **Chest pain type** and **max heart rate** emerged as the most influential features
- The model achieved strong true positive detection with minimal false positives
- ROC-AUC of 0.94 indicates excellent discriminative ability between healthy and at-risk patients
- Feature importance analysis aligned with clinical intuition — heart rate, chest pain, and cholesterol are key indicators

---

## Skills Demonstrated

| Skill | Project |
|-------|---------|
| Data loading & inspection with pandas | All |
| Descriptive statistics & EDA | Iris, Heart Disease |
| Visualization with matplotlib & seaborn | All |
| Time series data handling | Stock Prices |
| Data fetching via API (yfinance) | Stock Prices |
| Binary & multi-class classification | Heart Disease |
| Regression modeling (Linear & Random Forest) | Stock Prices |
| Model evaluation (Accuracy, ROC, Confusion Matrix) | Heart Disease |
| Feature scaling & importance analysis | Heart Disease |

---

## How to Run

```bash
# 1. Clone or download this repository
# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run any notebook in Jupyter
jupyter notebook Heart.ipynb
jupyter notebook iris.ipynb
jupyter notebook Stocks.ipynb
```

## Dependencies
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- yfinance (Stocks project only)
- Jupyter Notebook / JupyterLab
