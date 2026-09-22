# INX Future Inc: Employee Performance Analysis and Prediction (IABAC Project 10281)

An end-to-end Data Science and Machine Learning project developed for INX Future Inc under IABAC Project Code: 10281. The primary objective was to investigate an approximate 8% decline in client satisfaction caused by dropping employee performance, identify the root drivers, evaluate department-level trends, and build an automated classification pipeline to predict employee performance ratings.

---

## Business Problem and Objectives

INX Future Inc observed an approximate 8% dip in overall client satisfaction tied directly to workforce performance challenges. This project addresses the 4 core deliverables mandated by IABAC:

1. Department-Wise Performance Analysis: Measure and contrast performance ratings across Development, Data Science, Research & Development, Sales, HR, and Finance.
2. Identification of Top Influencing Factors: Quantify feature importance and correlation metrics to find the primary drivers of employee performance.
3. Machine Learning Predictive Modeling: Develop a multi-class classifier using Random Forest to accurately predict employee performance tiers.
4. Actionable Business Recommendations: Provide evidence-based strategies for leadership and HR to boost morale, retention, and client satisfaction.

---

## Dataset Profile

* Total Records: 1,200 employees
* Feature Count: 28 features (19 numerical, 9 categorical, 0 missing values)
* Target Variable: PerformanceRating
  * Class 2 (Low Performer): Employee underperforming relative to standard benchmarks.
  * Class 3 (Medium / Average Performer): Majority class meeting standard expectations.
  * Class 4 (High Performer): Employee consistently exceeding performance goals.
* Class Imbalance Strategy: Handled via SMOTE (Synthetic Minority Over-sampling Technique) to balance minority classes (2 and 4).

---

## Key Insights and Core Findings

### 1. Department-Wise Trends (Requirement 1)
* Top Performers: Development and Data Science departments recorded the highest average performance ratings.
* Lowest Performer: Finance department recorded the lowest overall performance rating, requiring targeted intervention.

### 2. Top 3 Factors Affecting Performance (Requirement 2)
1. EmpEnvironmentSatisfaction: Demonstrates the strongest positive correlation with high performance ratings.
2. EmpLastSalaryHikePercent: Shows a clear positive relationship; employees receiving competitive increments deliver superior performance.
3. EmpWorkLifeBalance: Critical stability factor reducing burnout and sustaining high performance.

---

## Machine Learning Model and Evaluation (Requirement 3)

### Why Random Forest?
Random Forest Classifier was selected as the champion model because of its ability to handle non-linear relationships across 27+ mixed feature types, prevent overfitting, and provide transparent feature importances.

### Performance Comparison (Before vs After SMOTE):

| Metric | Before SMOTE | After SMOTE | Impact |
| :--- | :--- | :--- | :--- |
| Accuracy | 84.0% | 85.0% | Slightly Improved |
| Macro Precision | 0.94 | 0.78 | Balanced across classes |
| Macro Recall | 0.54 | 0.69 | +15% Boost in minority detection |
| Macro F1-Score | 0.61 | 0.72 | Significant overall improvement |
| Weighted F1-Score | 0.80 | 0.84 | Robust generalization |

* Class 2 F1-Score: Improved from 0.65 to 0.75 after SMOTE.
* Class 4 F1-Score: Improved from 0.26 to 0.51 after SMOTE.

---

## Strategic Business Recommendations (Requirement 4)

* Enhance Work Environment Satisfaction: Conduct regular internal pulse surveys and upgrade physical/remote workspace tooling.
* Align Salary Increments with Merit: Re-evaluate compensation bands; last salary hike percentage is a decisive motivator.
* Work-Life Balance Safeguards: Implement reasonable limits on unnecessary overtime and establish flexible working hours.
* Focused Departmental Upskilling: Design custom training and leadership enablement programs specifically for the Finance team.
* Recognition and Reward Framework: Incentivize Class 4 high performers to prevent attrition while mentoring Class 2 employees.

---

## Tech Stack and Tools

* Programming Language: Python 3.10+
* Data Processing and Analytics: Pandas, NumPy
* Data Visualization: Matplotlib, Seaborn
* Machine Learning and Preprocessing: Scikit-learn, Imbalanced-learn (SMOTE)
* Model Serialization: Joblib / Pickle (model.pkl, feature_columns.pkl)

---

## Repository Structure

```text
├── data/                               # Raw & processed workforce datasets
├── src/                                # Modular pipeline scripts & notebooks
│   ├── data_cleaning.ipynb             # Data inspection and null validation
│   ├── feature_engineering.ipynb      # One-hot encoding & feature transformation
│   ├── train_test_split.ipynb         # Stratified training/testing partitioning
│   ├── train_model(main_file).ipynb   # Core Random Forest training & SMOTE evaluation
│   ├── predict_model.ipynb.ipynb      # End-to-end inference and prediction test
│   └── visualization/                  # Exploratory analysis & saved visuals
│       ├── eda_visualization.ipynb    # EDA code
│       └── Saved Images/               # Exported charts & heatmaps
│           ├── correlation_heatmap.png
│           ├── department_analysis.png
│           └── performance_distribution.png
├── Project summary/                    # Executive summary (Project_Summary.pdf)
├── references/                         # IABAC project specifications (Analysis.pdf, Requirement.txt)
├── model.pkl                           # Serialized Random Forest classification model
├── feature_columns.pkl                 # Persisted feature registry
├── requirements.txt                    # Project environment dependencies
└── README.md                           # Comprehensive documentation
```

---

## Setup and Execution

### 1. Environment Setup
```bash
python -m venv .venv
# Activate environment (Windows):
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Running the Pipeline
Open Jupyter Notebook and execute the notebooks in src/ sequentially:
1. data_cleaning.ipynb
2. feature_engineering.ipynb
3. eda_visualization.ipynb
4. train_test_split.ipynb
5. train_model(main_file).ipynb
6. predict_model.ipynb.ipynb
