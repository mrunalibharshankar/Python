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

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/bb2702859046781e845ed7bdbb66d134d9039946/Imports.png)

The data is looking like this,
![Alt Text](https://github.com/mrunalibharshankar/Python/blob/bb2702859046781e845ed7bdbb66d134d9039946/Dataset.png)

## Data Exploration (Initial EDA and data cleaning)

## Data Exploration

## Data visualizations

## Correlations between variables

## Splitting data into X and y variables


## Logistic Regression model and Classification Report

## Cross Validation with k-fold


## Build Decision Tree Model and Classification Report

## Cross-validated with grid-search

## Built Random Forest Model and Classification Report

## Cross-validated with grid-search


## Final Results of Regression and Tree Based Models

### Random forest feature importance
### Decision Tree feature importance


## Conclusion and Summary


- Added [total_counter] column by using countif() function of excel to count the duplicate records against the CustomerId column which represents the unique records.
  
![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/94fa03a3d1f4c10b5b602dc80acde987232b5910/total_counter.png)

# Pivot Analysis
- Created new field list ChurnRate by dividing [Exited]/[total_counter] to find out [Sum_of_Churnrate]
- First analysis was to draw conclusion of [NoOfProducts] against [ChurnRate] and [CreditScore]
  
![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/6d9b6a9250941794aa4b910593eb6d9804a586f7/Pivot1.png)

- Likewise we have created pivot table for [Tenure], [Gender] and [Georgraphy] with [ChurnRate] and [CreditScore]

![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/4e3b8f96549da6884b37e2921602a80bc79d9732/Pivot2.png)

![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/4e3b8f96549da6884b37e2921602a80bc79d9732/Pivot3.png)

![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/4e3b8f96549da6884b37e2921602a80bc79d9732/Pivot4.png)

- Conclusion drawn is,
  - the highest no of products holder are tend to exit.
  - Initial years(0, 1) customers ChurnRate is the highest.
  - Female customers ChurnRate compared to Male is highest.
  - Geographically, Germany's ChurnRate is highest against Spain and France.

With this information we can come up with the best subscription plans for these users.  


# Regression Analysis
- The target variable(dependent variable) is [Exited] to analysis the best independent variable explains about this variable we performed statistic regression using Data analysis in built tool in Excel.

![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/8eacec2c83b182301b9bfa681adfbde0396cc589/RegressionS1.png)

![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/8eacec2c83b182301b9bfa681adfbde0396cc589/RegressionS2.png)

![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/8eacec2c83b182301b9bfa681adfbde0396cc589/RegM1.png)
![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/fd64072f167827c116dd25cb6589694c1304abc2/RegOutput.png)


# Conclusion:  
1. Statistical Significance: A coefficient is considered statistically significant if its p-value is below a predetermined significance level (e.g., 0.05).
   In this case, **p-value of CreditScore is 0.007** (< 0.05), it determined the significant relationship between CredtiScore(independent variable) and Exited (dependent variable).

![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/8eacec2c83b182301b9bfa681adfbde0396cc589/RegFinalM.png)

2. Coefficient Interpretation: The sign and magnitude of the coefficients are interpreted.
   In our case, **coefficients is -0.0002** which indicates a negative relationship.
3. R-squared metrics: A higher R-squared value(ranges from 0 to 1) suggests that the independent variables explain a larger proportion of the variance in the dependent variable. R-squared of CreditScore is **0.0008** is 0.08% proportion of variance in Exited variable which is very very less but better than other variables to consider.
   
4. Model Fit: To determine the coefficients that result in the smallest residual sum of squares (the sum of the squared differences between observed and predicted values). 
![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/fd64072f167827c116dd25cb6589694c1304abc2/BestfitlineGraph.png)

    
# Further Analysis
 The target variable was in binary form(0,1) so the better analysis can be done in Logistic Regression with different evaluation metrics.

  

  















