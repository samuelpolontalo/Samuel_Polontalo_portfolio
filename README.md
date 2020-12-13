# Samuel_Polontalo_portfolio
Samuel's Data science project

# [Project 1: Rock, Paper and Scissor image recognition using Tensor Flow](https://github.com/samuelpolontalo/rock-paper-scissors-recognition-using-TensorFlow)
# [Project 2: Housing Price Advanced](https://github.com/samuelpolontalo/Housing-Price-Advanced)
## Project Overview
* Exploratory data analysis
* Apply data cleaning:
  * solve the categorical data('object') read as numerical data('int64' and 'float64').
  * solve the numerical data('int64' and 'float64') read as categorical data('object').
  * solve missing value.
  * removing the outlier.
  * Create One-Hot-Encoder for handle categorical data.
* Feature Engineering.
* Build regression model using XGBOOST (`XGBRegressor`).
* Evaluate model using `cross_val_score`.
* Tuning hyper parameter using `RandomizedSearchCV`.
* Save the prediction results in csv.
# [Project 3: Jane Street Market Prediction](https://github.com/samuelpolontalo/Jane-Street-Market-Prediction)
## Project Overview
* Data preparation:
 * Read big dataset using `dask.dataframe` then convert it to pandas dataframe (there is 137 columns and over 20 million row)
 * Create column action based on column 'resp' and 'weight'
 * Fill missing value with 0
* Build Model using XGBOOST ('XGBClassifier'):
 * Activate the GPU on kaggle notebook.
* Predict the unseen data test.
