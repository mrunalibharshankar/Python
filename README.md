# Logistic Regression and Tree Based Model for employee retention

Providing data-driven suggestions for HR using a regression and machine learning model to predict whether or not an employee will leave the company.

# Understand the business scenario and problem
The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They are asking to provide data-driven suggestions based on understanding of the data. 

They have the following question: what’s likely to make the employee leave the company?

The goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.

If we can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

# Overview

## Dataset 
The data file has 10 columns with names:
1. satisfaction_level
2. last_evaluation
3. number_project
4. average_montly_hours
5. time_spend_company
6. Work_accident
7. left
8. promotion_last_5years
9. Department
10. salary

The original files were exported from the [Kaggle](https://www.kaggle.com/datasets/mrunalibharshankar/hr-employee-retention#:~:text=HR_capstone_dataset.-,csv,-Summary), and is available in this repository as an [.CSV file](https://github.com/mrunalibharshankar/Python/blob/98905f38ef3704a651371c66b1cb6c6f71452c46/HR_capstone_dataset.csv) document.


## Importing relevant libraries and packages
We have used Jupiter Notebook of Anaconda to evaluate and build the model in python. Started off with importing relevant libraries and packages:
1. For data manipulation: Numpy and Pandas
2. For data visualization: Matplotlib.pyplot and Seaborn
3. For data modeling: XGboost(XGBClassifier, XGBRegressor, plot_importance), sklearn.linear_model(LogisticRegression), sklearn.tree(DecisionTreeClassifier), sklearn.ensemble(RandomForestClassifier)
4. For metrics and helpful functions: sklearn.model_selection(GridSearchCV, train_test_split), sklearn.metrics(accuracy_score, precision_score, recall_score,roc_auc_score, roc_curve, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report), sklearn.tree(plot_tree)
5. For saving models: pickle

The data is looking like this,
![Alt Text](https://github.com/mrunalibharshankar/Python/blob/bb2702859046781e845ed7bdbb66d134d9039946/Dataset.png)

## Data Exploration(Initial EDA, data cleaning and data visualization)
Examining and visualizing data to understand its characteristics, uncover patterns, and identify potential insights.
1. Basic Info
2. Descriptive Statistic
3. Missing Values
4. Univariate Analysis
5. Bivariate Analysis

## Correlations between variables
The correlation heatmap confirms that the number of projects, monthly hours, and evaluation scores and to some extend tenure all have some positive correlation with each other, and whether an employee leaves is negatively correlated with their satisfaction level.

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/Heatmap.png)

## Logistic Regression model and Classification Report
Goal is to predict whether an employee leaves the company, which is a categorical outcome variable. So this task involves classification. More specifically, this involves binary classification, since the outcome variable left can be either 1 (indicating employee left) or 0 (indicating employee didn't leave).

Will build Logistics regression model first and will compare which one serves the best results.

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/LR.png)

## Cross Validation with k-fold

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/LR_CV.png)

The classification report above shows that the logistic regression cv model achieved a precision of 47%, recall of 25%, f1-score of 32% (all weighted averages), and accuracy of 89%. However, if it's most important to predict employees who leave, then the scores are significantly lower.


## Build Decision Tree Model and Classification Report

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/DT.png)

## Cross-validated with grid-search

![Alt Text]()

## Built Random Forest Model and Classification Report

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/RF.png)

## Cross-validated with grid-search

![Alt Text]()


## Final Results of Regression and Tree Based Models
![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/Result_tb.png)

### Random forest feature importance
![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/RF_FI.png)
### Decision Tree feature importance
![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/DT_Feature_I.png)

## Conclusion and Summary


  















