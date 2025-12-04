# Million Song Dataset – Genre Classification

This project builds and evaluates several machine learning models to predict musical genre using features from the Million Song Dataset.

## Project Structure

- `MillionSong_Model.ipynb` – main notebook:
  - Loads and inspects the dataset
  - Performs exploratory data analysis (EDA)
  - Preprocesses features and labels
  - Trains and evaluates multiple models
  - Runs SHAP explanations and additional experiments
- `msd_genre_dataset.txt` – preprocessed feature + genre label file, the dataset that we use.

## Data

We use `msd_genre_dataset.txt`, which contains:

- **Target**: `genre`
- **Metadata**: `track_id`, `artist_name`, `title`
- **Audio features**:
  - Global features: `loudness`, `tempo`, `time_signature`, `key`, `mode`, `duration`
  - Timbre statistics: `avg_timbre1`–`avg_timbre12`, `var_timbre1`–`var_timbre12`

Classes include genres such as:

- classic pop and rock
- folk
- dance and electronica
- jazz and blues
- soul and reggae
- punk
- metal
- classical
- pop
- hip-hop

## Methods

### Exploratory Data Analysis (EDA)

The notebook explores:

- Class distribution across genres
- Boxplots of loudness, tempo, and duration by genre
- A correlation heatmap of numerical features

### Preprocessing

- Drops metadata columns: `track_id`, `artist_name`, `title`
- Encodes `genre` with `LabelEncoder`
- Splits into **train / validation / test**:
  - 70% train, 15% validation, 15% test (via two-stage split)
- Standardizes features with `StandardScaler` (fit on train only)

### Models

Trained and evaluated models:

- **Logistic Regression**
  - Multinomial, `lbfgs` solver
  - Reports accuracy, precision/recall/F1, and normalized confusion matrix
  - Global feature importance from coefficients

- **Decision Tree**
  - Baseline tree classifier
  - Reports classification metrics and confusion matrix
  - Plots top-10 feature importances

- **XGBoost (XGBClassifier)**
  - Gradient-boosted trees
  - Reports metrics and top-10 feature importances
  - Confusion matrix visualization
  - SHAP

- **CatBoost (CatBoostClassifier) -- Extra**
  - Gradient boosting with ordered boosting
  - Reports metrics and top-10 feature importances
  - Confusion matrix visualization

### Model Evaluation and Experiments

- **Validation performance**:
  - Compares Logistic Regression, Decision Tree, XGBoost, and CatBoost on a held-out validation set.
- **Feature ablation**:
  - Retrains Logistic Regression without `avg_timbre1` to test its importance.
- **Class imbalance handling (SMOTE)**:
  - Applies SMOTE to oversample minority classes on the training data.
  - Retrains all four models on resampled data and compares test accuracy.

### Explainability (SHAP for XGBoost)

- Uses **SHAP (TreeExplainer)** on the XGBoost model.
- Produces:
  - Bar summary plot (mean |SHAP| per feature)
  - Beeswarm summary plot of feature impact across samples

## Requirements

Install dependencies (example using `pip`):

```bash
pip install \
  numpy pandas seaborn matplotlib scikit-learn \
  xgboost catboost imbalanced-learn shap
```

You also need Jupyter to run the notebook:

```bash
pip install notebook
```

> Note: SHAP and XGBoost can be slower on large datasets; running with a subset of samples for SHAP is supported in the notebook.

## How to Run

1. Clone this repository and navigate to the project folder:

   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-folder>
   ```

2. (Optional but recommended) create and activate a virtual environment.

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   or, if you are not using a `requirements.txt`, install the packages listed above manually.

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `MillionSong_Model.ipynb` in the browser.
6. Run all cells from top to bottom (**Kernel → Restart & Run All**) to:
   - Load and explore the data
   - Train all models
   - Generate evaluation metrics, plots, and SHAP explanations

## Results (High-Level)

- Tree-based models (XGBoost and CatBoost) achieve **higher accuracy** than Logistic Regression and Decision Tree.
- Important features across models consistently include:
  - **`avg_timbre1`, `avg_timbre2`, `avg_timbre6`, `loudness`, `duration`**
- SMOTE improves class balance but may reduce accuracy for some models while slightly improving minority-class performance.

## Authors

- **Shishi Jiang**


