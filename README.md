# Samuel_Polontalo_portfolio
Samuel's Data science project

# [Project 1: Rock, Paper and Scissor image recognition using Tensor Flow](https://github.com/samuelpolontalo/rock-paper-scissors-recognition-using-TensorFlow)
## Project Overview
* Data preparation:
  * Read dataset using `os` and `zipfile`.
  * Create directory.
  * Split data into data train and data test using sklearn `train_test_split`.
  * Create new directory and copy the data train and data test using `shutil`.
* Build model:
  * Process images using `ImageDataGenerator`.
  * Create several layer Neural Network using TensorFlow.
* Plot training, validation accuracy and training, validation loss to evaluate the model.
* Create a file upload feature to test the prediction model.

# [Project 2: Housing Price Advanced](https://github.com/samuelpolontalo/Housing-Price-Advanced)
## Project Overview
* Exploratory data analysis
* Apply data cleaning:
  * Solve the categorical data('object') read as numerical data('int64' and 'float64').
  * Solve the numerical data('int64' and 'float64') read as categorical data('object').
  * Solve missing value.
  * Removing the outlier.
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
* Build Model using XGBOOST (`XGBClassifier`):
  * Activate the GPU on kaggle notebook.
* Predict the unseen data test.
