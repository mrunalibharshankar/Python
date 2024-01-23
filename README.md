# Regression and Tree Based Model for employee retention

Providing data-driven suggestions for HR using a regression and machine learning model to predict whether or not an employee will leave the company


# Understand the business scenario and problem
The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They are asking to provide data-driven suggestions based on understanding of the data. 

They have the following question: what’s likely to make the employee leave the company?

The goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.

If we can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

- _RowNumber_, unique no.
- _CustomerId_, unique id represents the individual customer.
- _Surname_, last name of the customer
- _CreditScore_, number that measures customers risk.
- _Geography_, country the customer is based, and its categorical variable.
- _Gender_, Female or Male, its categorical variable.
- _Age_, age of the person.
- _Tenure_, for how many years the customer is with the bank.
- _Balance_, available amount for withdrawals.
- _NumOfProducts_, no of products the customer has and, its a categorical variable.
- _HasCrCard_, if customer owes credit card. 1 for yes and 0 for no.
- _IsActiveMember_, 1 for he/she is active and 0 for not active.
- _EstimatedSalary_, Salary.
- _Exited, 1 if the client has left the bank and 0 if he/she has not. Its the target 
       variable.

The original files were exported from the [Kaggle](https://www.kaggle.com/datasets/mrunalibharshankar/hr-employee-retention#:~:text=HR_capstone_dataset.-,csv,-Summary), and are available in this repository as an [.CSV file](https://github.com/mrunalibharshankar/Python/blob/98905f38ef3704a651371c66b1cb6c6f71452c46/HR_capstone_dataset.csv) document.

Regression analysis is a statistical technique used to examine the relationship between one dependent variable and one or more independent variables. The goal is to understand how the independent variables impact the dependent variable.

For Regression Analysis to approximate how two variables in a dataset are related. In the following dataset for regression, we’ll have:
- An independent variable, which is explanatory variable. This variable can be directly controlled.
- A dependent variable, which is target variable. This variable cannot be directly controlled, and can depend on independent variable.

The general form of a regression equation is:
 **Y= β0 + β1*X1 +β2X2+…+βnXn + ϵ**

- Y is the dependent variable.
- X1,X2,..Xn are independent variables.
- β0 is the y-intercept.
- β1, β2,..βn are coefficients representing the relationship between the independent and dependent variables.
- ϵ is the error term.

The general line graph, that independent variable runs along the x-axis, and dependent variable runs along the y-axis
![Alt Text](https://github.com/mrunalibharshankar/RegressionAnalysis/blob/24c43203a0b58bc66dd831b17bf089f5d2a44fe4/line_example.png)


The regression analysis aims to estimate the coefficients (β) that minimize the difference between the observed and predicted values of the dependent variable.

Key concepts associated with regression analysis include:

- _R-squared (R²)_: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
- _Coefficients_: Represent the strength and direction of the relationship between the independent and dependent variables.
- _Residuals_: The differences between the observed and predicted values.


# Transforming Variable
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

  

  















