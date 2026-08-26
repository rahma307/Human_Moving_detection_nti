# Human Activity Recognition Using Smartphones

A supervised multiclass machine learning project that predicts human physical activities using smartphone accelerometer and gyroscope sensor features.

## 📌 Project Overview

This project builds a machine learning pipeline for **Human Activity Recognition (HAR)** using the UCI Human Activity Recognition Using Smartphones dataset.

The goal is to classify each observation into one of six human activities based on engineered smartphone sensor features:

* 🚶 Walking
* ⬆️ Walking Upstairs
* ⬇️ Walking Downstairs
* 🪑 Sitting
* 🧍 Standing
* 🛏️ Laying

The project covers the complete machine learning workflow, including:

* Dataset understanding and validation
* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature scaling
* Multiple machine learning models
* Model comparison
* PCA dimensionality analysis
* Feature importance
* Error analysis
* Hyperparameter tuning
* Final model evaluation
* Artifact generation for the Streamlit dashboard

---

## 📊 Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

Dataset source:

https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

### Dataset Characteristics

| Property           |        Value |
| ------------------ | -----------: |
| Subjects           |           30 |
| Activities         |            6 |
| Features           |          561 |
| Sampling Frequency |        50 Hz |
| Window Size        | 2.56 seconds |
| Window Overlap     |          50% |
| Training Samples   |        7,352 |
| Testing Samples    |        2,947 |

The data was collected from 30 volunteers aged 19–48 using a **Samsung Galaxy S II** smartphone mounted on the waist.

The smartphone recorded:

* 3-axis linear acceleration
* 3-axis angular velocity

The original UCI train/test split is preserved:

* **21 subjects → training**
* **9 subjects → testing**

The split is performed by subject to avoid leakage between training and testing data.

---

## 🧠 Features

The dataset contains **561 engineered features** derived from time-domain and frequency-domain sensor signals.

Examples include:

* Mean
* Standard deviation
* Minimum and maximum values
* Signal energy
* Correlations between axes
* Frequency-domain / FFT-based features
* Body acceleration features
* Gravity acceleration features
* Gyroscope features
* Magnitude-based features

Each row represents a statistical summary of a **2.56-second sensor window**, rather than raw sensor readings.

---

## 🔍 Exploratory Data Analysis

The notebook performs several EDA steps to understand the dataset, including:

* Class distribution analysis
* Feature distribution visualization
* Selected meaningful feature analysis
* Correlation analysis
* Investigation of relationships between sensor features and activities

The six classes are reasonably balanced, so accuracy is a meaningful evaluation metric. Precision, recall, and macro F1-score are also reported to provide a more complete evaluation.

---

## ⚙️ Preprocessing

The feature matrix is standardized using `StandardScaler`.

The scaler is fitted **only on the training data**:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This prevents information from the test set from leaking into the training process.

No random row-level train/test split is performed because observations from the same subject can be highly correlated.

---

## 🤖 Machine Learning Models

Five classical machine learning algorithms are trained and compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)

Each model is evaluated using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* Confusion Matrix

### Model Performance

The baseline comparison produced approximately:

| Model               |   Accuracy |
| ------------------- | ---------: |
| Logistic Regression | **95.49%** |
| SVM                 | **95.18%** |
| Random Forest       |     92.74% |
| KNN                 |     88.36% |
| Decision Tree       |     86.22% |

The exact values are generated from the notebook's `comparison_df`.

---

## 📉 PCA Analysis

Principal Component Analysis (PCA) is used to investigate the redundancy and structure of the 561-dimensional feature space.

Two PCA analyses are performed:

1. Cumulative explained variance
2. Two-dimensional PCA visualization

The 2D projection shows a clear separation between:

* **Dynamic activities:** Walking, Walking Upstairs, Walking Downstairs
* **Static activities:** Sitting, Standing, Laying

However, activities within the same group overlap more strongly, particularly Sitting vs. Standing and the three walking activities.

---

## 🌲 Feature Importance

Random Forest feature importance is used to identify the engineered features that contribute most to classification.

Important features are mainly related to:

* Gravity orientation
* Body acceleration magnitude
* Motion energy
* Time-domain acceleration statistics
* Frequency-domain acceleration statistics

These features are physically meaningful because they capture both body orientation and movement intensity.

---

## 🔎 Error Analysis

The confusion matrix is used to investigate classification errors.

The main difficult distinction is:

**Sitting ↔ Standing**

These activities produce very similar low-motion sensor signals and phone orientations.

Some confusion also occurs between:

* Walking
* Walking Upstairs
* Walking Downstairs

because these activities share similar gait patterns.

Laying is generally easier to distinguish because its gravity orientation differs substantially from the other activities.

---

## 🎯 Hyperparameter Tuning

The most promising models are further optimized using `GridSearchCV`.

### SVM

The following parameters are explored:

```python
C = [1, 10]
gamma = ["scale", 0.01]
```

### Random Forest

The following parameters are explored:

```python
n_estimators = [200, 300]
max_depth = [None, 30]
```

Three-fold cross-validation is performed **using the training set only**.

The tuned model with the best cross-validation accuracy is selected as the final model.

---

## 🏆 Final Evaluation

After model selection and hyperparameter tuning, the final model is evaluated once on the untouched test set.

The final evaluation reports:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* Classification Report
* Confusion Matrix

The final test set is not used during hyperparameter tuning.

This provides an unbiased estimate of the model's performance on unseen subjects.

---

## 📁 Project Structure

A typical project structure is:

```text
Human-Activity-Recognition/
│
├── data/
│   ├── train/
│   ├── test/
│   ├── features.txt
│   └── activity_labels.txt
│
├── artifacts/
│   ├── scaler.joblib
│   ├── final_model.joblib
│   ├── final_model_name.joblib
│   ├── all_models.joblib
│   ├── model_comparison.csv
│   ├── sample_test_X.csv
│   ├── sample_test_y.csv
│   ├── activity_distribution.csv
│   ├── final_confusion_matrix.csv
│   └── pca_2d.csv
│
├── HAR_Project.ipynb
├── app.py
└── README.md
```

> The exact dashboard filename should match the Streamlit application file in the project repository.

---

## 💾 Dashboard Artifacts

The notebook exports trained models and analysis results to the `artifacts/` directory so that the dashboard can display results without retraining the models.

Generated artifacts include:

| Artifact                     | Purpose                      |
| ---------------------------- | ---------------------------- |
| `scaler.joblib`              | Fitted feature scaler        |
| `final_model.joblib`         | Selected final trained model |
| `final_model_name.joblib`    | Name of the final model      |
| `all_models.joblib`          | All trained baseline models  |
| `model_comparison.csv`       | Model performance comparison |
| `sample_test_X.csv`          | Sample test features         |
| `sample_test_y.csv`          | Sample test labels           |
| `activity_distribution.csv`  | Activity distribution        |
| `final_confusion_matrix.csv` | Final confusion matrix       |
| `pca_2d.csv`                 | PCA visualization data       |

The dashboard loads these artifacts instead of retraining the models.

---

## 🖥️ Dashboard

The project includes a dashboard that summarizes the machine learning workflow and results.

The dashboard presents information such as:

* Project description
* Dataset statistics
* Number of features
* Number of activities
* Number of subjects
* Sample test-set features
* Model performance
* PCA analysis
* Feature importance
* Final evaluation
* Confusion matrix

The dashboard is intended for **visualization and demonstration**; the complete modeling pipeline is implemented in `HAR_Project.ipynb`.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd Human-Activity-Recognition
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter streamlit
```

### 3. Run the notebook

Open:

```bash
jupyter notebook HAR_Project.ipynb
```

Run the notebook from top to bottom.

This will:

1. Load the dataset
2. Validate the data
3. Perform EDA
4. Preprocess the features
5. Train the models
6. Compare the models
7. Perform PCA
8. Analyze feature importance
9. Perform error analysis
10. Tune the best models
11. Evaluate the final model
12. Generate the dashboard artifacts

### 4. Run the dashboard

If the Streamlit application is named `app.py`:

```bash
streamlit run app.py
```

---

## 📈 Key Findings

The project shows that the engineered smartphone sensor features are highly informative for human activity recognition.

Key findings include:

* Classical machine learning models can achieve around **95% accuracy** on this dataset.
* Logistic Regression and SVM perform particularly well with the 561 engineered features.
* PCA reveals substantial redundancy in the feature space.
* Two PCA components already provide a useful visual separation between static and dynamic activities.
* Random Forest feature importance highlights gravity orientation and movement-energy features.
* The main classification challenge is distinguishing **Sitting from Standing**.
* Walking activities can also be confused with one another because of their similar motion patterns.
* Hyperparameter tuning provides an additional improvement over default model configurations.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook
* Streamlit

---

## 📚 References

**UCI Machine Learning Repository**

Human Activity Recognition Using Smartphones Dataset:

https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

---

## 👩‍💻 Project Type

**Machine Learning — Multiclass Classification**

**Task:** Human Activity Recognition

**Input:** Smartphone accelerometer and gyroscope features

**Output:** One of six human activity classes

**Models:** Logistic Regression, KNN, Decision Tree, Random Forest, SVM
