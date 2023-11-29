# Samuel Polontalo Portfolio
**About Me**: I am a scientist. A planner and conceptor. Work in a structured and systematic manner. Adaptive, intuitive, able to learn fast, and can work under pressure. Able to work in teams and be able to work independently. Love a challenge and exploring new things. Having a competition in collaboration spirit. Interested in data science, big data, and artificial intelligence. 

# Here My Data Science Project :

## [Lintasarta CSR Program, SMKN 4 Kupang: Weather Prediction](https://github.com/samuelpolontalo/Weather_Prediction)
The "Weather Prediction" project by Al for Youth from SMK Negeri 4 Kupang addresses the recurring issue of crop failures among NTT farmers due to unpredictable weather conditions. Using Artificial Intelligence, specifically Gradient Boosting, the team developed a software tool that leverages data from BMKG for accurate weather predictions in Kupang. The "Typing and Knowing" feature allows users to input instructions in Bahasa Indonesia to receive real-time weather forecasts. The success of the project underscores the potential of artificial intelligence to assist farmers, and the team expresses gratitude to PT Lintasarta and Intel AI For Youth for their support.

## [Personal Project: Rock, Paper and Scissor image recognition using Tensor Flow](https://github.com/samuelpolontalo/rock-paper-scissors-recognition-using-TensorFlow)
In this data science project, I created a program to recognize rock, paper, and scissors images using the TensorFlow neural network.
### [Project Overview]
  * Read dataset using `os` and `zipfile`.
  * Create directory.
  * Split data into data train and data test using sklearn `train_test_split`.
  * Create new directory and copy the data train and data test using `shutil`.
* Build model:
  * Process images using `ImageDataGenerator`.
  * Create several layer Neural Network using TensorFlow.
* Plot training, validation accuracy and training, validation loss to evaluate the model.
* Create a file upload feature to test the prediction model.

## [Kaggle Competition: Jane Street Market Prediction](https://github.com/samuelpolontalo/Jane-Street-Market-Prediction)
In this challenge, i built my own quantitative trading model to maximize returns using market data from a major global stock exchange. Next, i tested the predictiveness of my models against future market returns and receive feedback on the leaderboard.

in this challenge, I used the historical data, mathematical tools, and technological tools at my disposal to create a model that gets as close to certainty as possible.
To increase accuracy, i build 2 statistical model using XGBOOST and Neural Network.
### [Project Overview]
* Data preparation:
  * Read big dataset using `dask.dataframe` then convert it to pandas dataframe (there is 137 columns and over 20 million row)
  * Create column action based on column 'resp' and 'weight'
  * Fill missing value with 0
* Build Model using XGBOOST (`XGBClassifier`):
  * Activate the GPU on kaggle notebook.
  * Optimize the parameters by trial and errors.
* [Build Model using NeuralNetwork TensorFlow]:
  * Apply `BatchNormalization()` to improve performance model.
  * Apply `Dropout()` to improve performance model.
  * Using `EarlyStopping` to prevent model from overfitting.
* Predict the unseen data test.

## [Kaggle Competition: Housing Price Advanced](https://github.com/samuelpolontalo/Housing-Price-Advanced)
In this competition I solved the problem of predicting house prices. On this data science project, I demonstrated my ability to perform data cleaning, create EDA(Exploratory data analysis) and build statistical models.

### [Project Overview]
* Exploratory data analysis
* Apply data cleaning:
  * Solve the categorical data('object') read as numerical data('int' and 'float').
  * Solve the numerical data('int64' and 'float64') read as categorical data('object').
  * Solve missing value.
  * Removing the outlier.
  * Create One-Hot-Encoder for handle categorical data.
* Feature Engineering.
* Build regression model using XGBOOST (`XGBRegressor`).
  * Evaluate model using `cross_val_score`.
  * Tuning hyper parameter using `RandomizedSearchCV`.
* Save the prediction results in csv.

## [Kaggle Competition: People Analitic BRI data Hackaton 2021](https://github.com/samuelpolontalo/People-analytic-hackaton-BRI)
In this science data competition, I created a statistical model using machine learning to predict employee performance based on the given key performance indication (KPI)
### [Project Overview]
* Exploratory data analysis using Power BI
* Data Cleaning
* Build classifier model using XGBOOST (`XGBClassifier`) with AUC Metric evaluation.
* Tuning hyper parameter using `RandomizedSearchCV`.
* Save the prediction results in csv.

## [Kaggle Competition: Cassava Leaf Disease Classification](https://github.com/samuelpolontalo/cassava-leaf-disease)
As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated.
### [Project Overview]
* Data overview
* Data Preparation
  * Streaming data using API.
  * Installing Kaggle on google colab.
  * Read dataset using `os` and `zipfile`.
  * Create directory.
  * Read the train.csv as reference label using `json`
  * Create directory.
  * Split data into data train and data test using sklearn `train_test_split`.
  * Create new directory and copy the data train and data test using `shutil`.
* Build Model
  * Process images using `ImageDataGenerator`.
  * Create several layer Neural Network using TensorFlow.
* Plot training, validation accuracy and training, validation loss to evaluate the model.
