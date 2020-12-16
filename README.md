# Samuel Polontalo Portfolio
I am a scientist. A planner and conceptor. Work in a structured and systematic manner. Adaptive, intuitive, able to learn fast, and can work under pressure. Able to work in teams and be able to work independently. Love a challenge and exploring new things. Having a competition in collaboration spirit. Interested in quantum information sciences, big data, and artificial intelligence. 

# [Project 1: Rock, Paper and Scissor image recognition using Tensor Flow](https://github.com/samuelpolontalo/rock-paper-scissors-recognition-using-TensorFlow)
In this data science project we created a program to recognize rock, paper and scissors images using the TensorFlow neural network.
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
**Competition Description**: Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
## Project Overview
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

# [Project 3: Jane Street Market Prediction](https://github.com/samuelpolontalo/Jane-Street-Market-Prediction)
In the first three months of this challenge, you will build your own quantitative trading model to maximize returns using market data from a major global stock exchange. Next, you’ll test the predictiveness of your models against future market returns and receive feedback on the leaderboard.

Your challenge will be to use the historical data, mathematical tools, and technological tools at your disposal to create a model that gets as close to certainty as possible. You will be presented with a number of potential trading opportunities, which your model must choose whether to accept or reject.
## Project Overview
* Data preparation:
  * Read big dataset using `dask.dataframe` then convert it to pandas dataframe (there is 137 columns and over 20 million row)
  * Create column action based on column 'resp' and 'weight'
  * Fill missing value with 0
* Build Model using XGBOOST (`XGBClassifier`):
  * Activate the GPU on kaggle notebook.
  * Optimize the parameters by trial and errors.
* Build Model using NeuralNetwork TensorFlow:
  * Apply `BatchNormalization()` to improve performance model.
  * Apply `Dropout()` to improve performance model.
  * Using `EarlyStopping` to prevent model from overfitting.
* Predict the unseen data test.

# [Project 4: Cassava Leaf Disease Classification](https://github.com/samuelpolontalo/cassava-leaf-disease)
As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated.
## Project Overview
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
  

# [Project 5: Titanic - Machine Learning from Disaster](https://github.com/samuelpolontalo/Titanic-Survival)
## The Challenge
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
## Project Overview
* Simple Exploratory data analysis
* Apply data cleaning:
  * Solve missing value.
  * Create One-Hot-Encoder for handle low cardinality categorical data.
* Build several classifier model machine learning then compare them each other using `cross_val_score`.
* Save the prediction results in csv.

# [Project 6: Exploratory Data Analysis COVID-19 Around The World and Southeast Asia.](https://github.com/samuelpolontalo/Simple-Exploratory-Data-Analysis-COVID-19)
In this data science project we will visualize fatality rate around the world and compare total case indonesia against 4 other countries.

# [Project 7: Housing Price Beginer](https://github.com/samuelpolontalo/Housing-Prices-Beginer)
**Competition Description**: Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
