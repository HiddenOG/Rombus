ASEMIC_WORKFLOW = """
ASEMIC Workflow (Acquire, Scrub, Explore, Model, Interpret, Communicate):

1. Acquire: Collect raw data from sources like databases, APIs, or CSV files.
   - Example: Loading Iris dataset from UCI Repository.
2. Scrub: Clean data by fixing errors, missing values, or outliers.
   - Example: Encoding categorical species, scaling features.
3. Explore: Analyze data with visualizations and statistics to find patterns.
   - Example: Scatter plots of sepal/petal dimensions, correlation heatmaps.
4. Model: Build machine learning models to predict or classify.
   - Example: Training an SVM classifier.
5. Interpret: Understand model results in the context of the problem.
   - Example: Checking accuracy, precision, recall, and F1-score.
6. Communicate: Share findings with stakeholders through reports or dashboards.
   - Example: Building a UI to display predictions.
"""

IRIS_CASE_STUDY = """
Case Study: Predicting Iris Flower Species
- Goal: Predict species (Setosa, Versicolor, Virginica) based on sepal/petal measurements.
- Data: 150 samples, 4 features (Sepal_Length, Sepal_Width, Petal_Length, Petal_Width).
- Steps:
  1. Acquire: pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=col_names)
  2. Scrub: LabelEncoder for Species, StandardScaler for features.
  3. Explore: sns.scatterplot, sns.heatmap.
  4. Model: SVC(kernel='linear', C=1).
  5. Evaluate: accuracy_score, classification_report.
"""
