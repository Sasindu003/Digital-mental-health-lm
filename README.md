# **AI-based Stress Level Prediction from Digital Device Usage**

2025-Y2-S1-MLB-B5G2-01
IT24101066	Nethmika N.G.D
IT24100928	Nilame D.V.P.B.D
IT24100926	Pathirana E.P.D.N
IT24101068	Digoarachchi S.A
IT24100945	Liyanage S.L.U
IT24101005	Weerasena H.P.P.R.

Overview of the Project

This project focuses on AI-based Stress Level Prediction from Digital Device Usage, aiming to predict an individual’s stress level based on their interaction with digital devices such as phones, laptops, tablets, and televisions. The dataset captures daily screen time, device usage patterns, and demographic factors like age and gender.
The goal is to build a predictive machine learning model that can classify or estimate stress levels, providing insights into how digital lifestyle and screen exposure contribute to mental health outcomes.

Key stages of the project include:
•	Data Cleaning and Preprocessing: Handling missing values, outliers, and categorical encodings.
•	Feature Engineering: Constructing new derived features such as total screen time or social-to-work usage ratios.
•	Feature Selection: Identifying relevant predictors through statistical tests and variance analysis.
•	Standardization and Data Balancing: Scaling numerical features and applying techniques to handle imbalanced stress classes.
•	Pipeline Integration: Organizing the workflow for reproducibility and model readiness.

Dataset Details
The dataset is named Digital Diet Mental Health (originally collected as digital_diet_mental_health.csv). It captures demographic information (such as age and gender) and detailed digital device usage patterns (daily screen time, phone, laptop, tablet, and TV hours, as well as social media and work-related usage). The dataset is designed for educational and research purposes, focusing on how patterns of technology consumption correlate with mental well-being and stress levels.



Key Statistics:
•	Shape: Approximately 1000 rows and 8–10 columns (after preprocessing).
•	Columns:
 o	Demographics:
  	Age (numerical, range: ~13–64).
  	Gender (categorical: Male, Female, Other → later label/one-hot encoded).
 o	Digital Usage Features (numerical/continuous, in hours):
  	daily_screen_time_hours (e.g., 0.1 – 12.2).
  	phone_usage_hours.
  	laptop_usage_hours.
  	tablet_usage_hours.
  	tv_usage_hours.
  	social_media_hours.
  	work_related_hours.
 o	Target Variable:
  	stress_level (categorical: e.g., Low, Medium, High).
  	Encoded versions (label-encoded integers and one-hot encoded stress categories) are created      for model training.
 •	Data Types:
   •	Numerical (float values for screen/device usage, integer for age).
   •	Categorical (gender, stress level → encoded).

Challenges:
•	Imbalanced classes: Some stress categories dominate (e.g., "Medium"), while others ("High") are underrepresented.
•	Missing Values: Handled by imputation (mean for continuous, mode for categorical).
•	Outliers: Removed using IQR method (extreme screen times beyond realistic limits).
•	Encoding: Applied Label Encoding (for gender, stress level) and One-Hot Encoding (for categorical targets).
•	Scaling: Standardization (z-score) applied to numerical features.
•	Feature Selection: Low-variance features and highly correlated features were removed to improve model quality.

Preprocessing Steps
1.	Handling Missing Data and Encoding
•	Identified and imputed missing values.
•	Encoded categorical features such as gender into numeric form.
2.	Outlier Detection
•	Used the IQR method to detect extreme outliers in usage hours.
•	Outliers removed to reduce skewness in distributions.
3.	Feature Engineering
•	Created aggregated features like total_screen_time and work_vs_social_ratio.
•	Normalized device usage values relative to daily screen time.
4.	Feature Selection
•	Applied variance threshold to drop low-variance features.
•	Used correlation analysis and statistical tests to select relevant predictors.
5.	Standardization
•	StandardScaler applied to all continuous features to normalize ranges.
6.	Data Balancing
•	Detected imbalance in stress level classes.
•	Balanced using SMOTE and oversampling to ensure fair model training.

Group Member Roles
•	IT24100926: Responsible for Standerdization Scaling. Applied scaling to normalize continuous features and visualized distributions with box plots before and after scaling.
•	IT24101005: Focused on Variance Threshold Feature Selection. Removed lowvariance features (threshold=0.0) to eliminate redundant columns.

•	Group Pipeline (No Specific ID): Integrated all steps into a single pipeline notebook. This combines loading, cleaning, encoding, scaling, feature selection, and saving the final dataset. It uses tools like SimpleImputer for missing values and joblib for model persistence.

How to Run the Code
1.	Setup Environment
•	Recommended: Google Colab or Jupyter Notebook.
•	Install required libraries:
•	pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
2.	Running the Pipeline
•	Upload digital_diet_mental_health.csv as input.
•	Execute preprocessing notebooks in logical order:
Missing Data → Outlier Detection → Encoding → Standardization → Feature Selection → Data Balancing.
•	Final processed dataset saved as Done.csv.
3.	Outputs
•	Visualizations: Histograms, Boxplots, Correlation Heatmaps, Class Distribution.
•	Final Dataset: Balanced and standardized data ready for machine learning model training.
