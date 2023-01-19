## Modelling Airbnb Property Listing Dataset Project
&nbsp;

The aim of this project is to develop a framework for a wide range of machine learning models that can be applied to various datasets. The main dataset used is the
Airbnb property listing dataset.<br />

&nbsp;


## Milestone 1 - Data Preparation
&nbsp;

__Tabular Data__ 

The first step is to download and save images and tabular data folder. The tabular data folder contains the AirBnbData.csv. The tabular data has the following colums:

* ID: Unique identifier for the listing
* Category: The category of the listing
* Title: The title of the listing
* Description: The description of the listing
* Amenities: The available amenities of the listing
* Location: The location of the listing
* guests: The number of guests that can be accommodated in the listing
* beds: The number of available beds in the listing
* bathrooms: The number of bathrooms in the listing
* Price_Night: The price per night of the listing
* Cleanliness_rate: The cleanliness rating of the listing
* Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
* Location_rate: The rating of the location of the listing
* Check-in_rate: The rating of check-in process given by the host
* Value_rate: The rating of value given by the host
* amenities_count: The number of amenities in the listing
* url: The URL of the listing
* bedrooms: The number of bedrooms in the listing
* Unamed: 19: empty column
  
Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,built on top of the Python programming language. In order to
process the tabular data pandas was installed (pip install pandas). 

The tabular_data.py script contains the functions coded in order to clean and process the tabular data. The  next step is to code __read_csv()__ function which reads in the tabular data csv and converts it to a pandas data frame(pd). The __copy()__ method is then called to create a copy of the df. This copy of the df can now be cleaned and manipulated. 

_missing ratings_ 
&nbsp;

The __isna().sum()__ function is used to calculate the sum of NaN values in the pd data frame. Figure 1 belows shows that the rating columns contained missing values. 

![NaN values count](project_images/Figure_1_sum_of_NaN_values_in_DF.PNG)

*Figure 1 - Sum of NaN values in the pd data frame

This would cause problems when trying to train the machine learning models therefore, a function was coded (figure 2) which passes the df as an argument and creates a new data frame (df1) with the __dropna(subset)__. The ratings columns now contain 0 missing values as shown in Figure 3 below. The function returns the new data frame.

![Alt text](project_images/Figure_2_remove_rows_with_missing_ratings_func.PNG)

*Figure 2 - Remove rows with missing ratings function


![Alt text](project_images/Figure_3_sum_of_NaN_values_in_DF.PNG)

*Figure 3 - Sum of NaN values in the pd data frame

_Description strings_ 
&nbsp;

The __combine_description_strings__ function (figure 4) is coded which passes the data frame which is returned from the  __remove_rows_with_missing_ratings__ function and processes and cleans the strings within the description column. The process includes the:

* Creates a new data frame(df2) using df1 and calls the __dropna(subset)__ to remove the NaN values in the description column. 
* Calling the __str.replace()__ function to replace the strings 'About this space' and other strings not needed and replace this with ''.
* Calling the __str.join()__ function to join the strings in the description column.
* Calling the __to_list()__ function to combine the list items into the same string.
* returns df2

&nbsp;



![Alt text](project_images/Figure_4_combine_description_string_func.PNG)

*Figure 4 - Combine description string_function

_Feature values_ 

The __set_default_feature_values(__ function (figure 5) is coded which passes the data frame which is returned from the __combine_description_strings__ function and creates a new df (df3)
using df2 and calls the __fillna(1)__ function which replaces the NaN values in the "guests", "beds", "bathrooms" and "bedrooms" columns with the value 1. the function returns the data frame(df3).

&nbsp;


![Alt text](project_images/Figure_5_set_default_features_values_func.PNG)

*Figure 5 - Set default features values function 



_clean tabular data_ 

The functions mentioned above are wrapped into the __clean_tabular_data__(figure 6) hence when called a new df is created with the cleaned tabular data then the __to_csv__ function is called to create a new csv labeled as 'clean_tabular_data.csv' from the new df.  

![Alt text](project_images/Figure_6_clean_tabular_csv_func.PNG)
![Alt text](project_images/Figure_6_clean_tabular_csv_func_pt2.PNG)

*Figure 6 - clean tabular data function 

&nbsp;

__Format Image Data__ 

The prepare prepare_image_data.py script contains the code which processes the image data. The images are saved in a folder which is named as the UUID of the listing which it contains the images for.

_downloadDirectoryFroms3 function_ 

The __downloadDirectoryFroms3__ (Figure 7) takes the aws S3 bucket name and Remote directory name as an argument  and downloads the images from the 'airbnb-property-listings' and saves into to the images folder.

![Alt text](project_images/Figure_7_downloadDirectoryFroms3.PNG)

*Figure 7 - downloadDirectoryFroms3 function

_create_directory function_ 

The __create_directory function__ (Figure 8) creates a new directory for the images that will be processed.The function returns the path of the directory.


![Alt text](project_images/Figure_8_create_directory.PNG)

*Figure 8 - Create directory function

_calculate_smallest_image_height function_

The __calculate_smallest_image_height function__ takes in the path returned from the __create_directory function__ and for loop is coded which iterates through each image and calculates the images height and appends the height to the image_height list.The smallest image height is calculated using the __min__ function. The function returns the smallest image height.

_delete_image_by_mode_

The __delete_image_by_mode__ function (Figure 9) takes in the  image mode (set to RBG), the file path of the resized image and the image and codes a for loop which check if the image mode of the image is not set to RBG and deletes the image file path if this is the case.

![Alt text](project_images/Figure_9_deletes_images_by_mode_func.PNG)

*Figure 9 - deletes images by mode function

_resize_images_

The __resize_images function__ takes in the path returned from the __create_directory function__ and codes a for loop which iterates through the images and  calls the __calculate_smallest_image_height function__ to return the smallest image height (base_height) then calculates the image aspect ratio (width/height) and resizes the image height based a ratio of the smallest image height (base_height*aspect_ratio) and saves the image in a directory located in the processed_images folder. This function is called  within the __name__ == "__main__"  block

![Alt text](project_images/Figure_10_resize_images_func.PNG)

Figure 10 - resize images function

&nbsp;

_get data in the right format_

For the first batch of modelling the features are the numerical tabular data and the label is the "price_night" feature. Within the tabular_data.py script the __load_airbnb function__ is coded which takes in the the new df created from the 'clean_tabular_data.csv' and assigns the features a new df which calls the select_dtypes(include='float64') hence the  features df contains the float numerical data. The labels df is then created using the new df and extracts the 'Price_Night' column. The features and labels are returned as a tuple.
&nbsp;

## Milestone 2 - Create a regresion model
&nbsp;

_simple regression model to predict the nightly cost of each listing_ 

The modelling.py script was created an contains the main code  for the various models. The first step is create a simple regression model to predict the nightly cost  of each listing. the __load_airbnb function_ is imported from the tabular_data.py which contains the df for the  features (numerical data - nightly cost) and label (nightly cost). This model is trained using the SKlearn __Stochastic Gradient Descent (SGD)Regressor class__. The __test split function__  is used to split the data into training and testing sets. The training set is used to train the model and the test set is used to provide an unbiased evaluation of a final model fit on the training data set. SGD model is imported from Sklearn and The next step to create a variable which calls the __SGDRegresssor__ class. The __fit()__ function is  then called to fit the training data with SGD. Now that the model is fitted,  the __prediction function__ is called to make a prediction of the nightly cost based on the fitted  training data set. The __np.random.seed function__ ensures(look it up in videos again)(FIGURE 11)




_Evaluation the regression model performance_

SKlearn is then used to evaluate the key measures of performance regression. This is done by importing the mean sqaure error and R2 score functions from the Sklearn metrics. The __mean_square_error__ and __r2score__ functions are then called to calculate the mean square error and the R2 score on the trained data set (Figure 11). 

![Alt text](project_images/Figure_11_SGD_model.PNG)

Figure 11 - SGD model code

_Evaluation the regression model performance_
&nbsp;



_Tune the hyperparameters of the model using methods from SKLearn_

In order to tune the accuracy of the model, the hyperparameters need to be tuned. This is done by implementing sklearns __GridSearchCV libary function__. The __GridSearchCV libary function__ helps loop through predefined hyperparameters and fits the model on the training set. The  __tune_regression_model_hyperparameters functions__(Figure 12) passes the model, X,y,X_test,y_test and a dictionary of the hyperparameters to be tuned and calls the __GridSearchCV libary function__ which loops through the dictionary of hyperparameters then the __fit function__ fits the model, then the __predict function__ makes a prediction on the test data set. The __mean_squared_error function__ calculates the mean squared error bestween the y_test and the predictions. The function returns the the best model, the best model hyperparameters and the hyperparameters.

![Alt text](project_images/Figure_12_CV_diagram.PNG)
Figure 12 - Cross validation diagram


&nbsp;

_Cross validation_
In general ML models the features dataset would be split into Training,Test and validation sets, however __GridSearchCV libary function__ contains a parameter called cv which stands for cross validation which  is a resampling method that uses different portions of the data to test and train a model on different iterations.In SKlearn the model is trained using k-1 of the folds as training data and then then resulting model is validated on the remaining part of the model(Figure 12).
&nbsp;

&nbsp;
![Alt text](project_images/Figure_13_Regression_Tune.PNG)

Figure 13 - Cross validation diagram

_Remaining GridSearchCV parameters_

The remaining paramters called in the __GridSearchCV libary function__ is the estimator which is the model, n_jobs which is set to -1 which means all proccessors are being used (this reduces the run time of the tuning proccess) and the verbose is set to 1 (hence no progress metrics are shown)

&nbsp;
&nbsp;

_hyperparameter selection_

In general The first value/boolean statement/option to be tuned for each hyperparameter was chosen based on the defaults provided in the SKlearn manual. Then a range of values were tested by increasing/decreasing from the default value. In general each hyperparameter provided in the SKlearn manual was chosen to be tuned. After a few model runs the some hyperparameters were removed due to time constraints.

&nbsp;
&nbsp;

_Save the model_

The __save_model function__(Figure 14) takes in the model,best_parameters,performance_metrics, and a key work argument folder and saves the best model and its hyperparameters and performance metrics.

&nbsp;

![Alt text](project_images/Figure_14_Save_model.PNG)
Figure 13 - Save model function


_Beat the baseline regression model_

In order to improve the baseline regression model, it was decided to apply different regression models provided by Sklearn. This includes decision trees, random forests, and gradient boosting. In order to run these addtional models they are first imported from Sklearn. The __evaluate_all_models function__ calls the __tune_regression_model_hyperparameters functions__ for each model scenario sequentially and the __save_model function__ is called to save the best model, hyperparameters and performance metrics.

&nbsp;

__find the best overall regression model__

In order to find the best overall regression model the performance metric(RMSE) needs to be compared against each model scenario. The __find_best_model function__ loads the RMSE for each model scenario and appends to a list, then a for loop is coded to find the lowest RMSE which will decide the best overall regression model and then returns the model, hyperparameters and performance metrics. The gradient boosting  algorithm contained the lowest RMSE and was the best regression model for price night model scenario.


## Milestone 3 - Create a classification model 

&nbsp;

_simple classification model to predict the category of the airbnb properties_ 

In order run the simple classification model the __load_airbnb function__  is used to generate the features (tabular data) and label ('category').

![Alt text](project_images/Figure_15_occurance_of_each_airbnb_category.PNG)

Figure 15 - Occurance of each type of ainbnb property 

in order to pass the data into the model, the label needs to be encoded to its numerical representation by using label Encoder. This is done by importing the __label Encoder function__
from the Sklearn and calling the an instance of the __label Encoder function__ then the __transform function__ to encode the label data. As with the regression model the next step is then to split the data into test and train datasets. The model is then trained using Sklearn __Logistic regression class__. Now that the model is fitted,  the __prediction function__ is called to make a prediction of the category of the air bnb apartments based on  training data set(Figure 16). The __np.random.seed function__ ensures(look it up in videos again)(FIGURE 11)


![Alt text](project_images/Figure_16_Logistic_model.PNG)

Figure 16 - Logistic regression model code

_Evaluation the regression model performance_

SKlearn is then used to evaluate the key measures of performance of the logistic reg. This is done by importing the __precision, recall and f1_score functions__ from the Sklearn metrics. The __precision, recall and f1_score functions__ functions are then called to calculate the mean square error and the R2 score on the test data set (Figure 16. 


&nbsp;


_Tune the hyperparameters of the model using methods from SKLearn_

Just like the regression models the hyperparameters for the logistic regression needs to be tuned. The steps follow the same process as mentioned within the Milestone but instead the 
code is wrapped in the __tune_classification_model_hyperparameters functions__(Figure 17).

&nbsp;


![Alt text](project_images/Figure_17_classification_tune.PNG)
Figure 17 - __tune_classification_model_hyperparameters()__ function

__save the classification model__

Similar to the regression model in milstone 2 the logistic regression model is saved along with its hyperparameters and performance metrics.

__beat the baseline classifcation model__

&nbsp;

Similar to the regression model in milstone 2 the performance of the baseline model can be improved by using different models provided by SKlearn. This is done by using the classification version of the decision trees, random forest and gradient boosting algorithms. The __evaluate_all_models__(Figure 18) calls the __tune_classification_model_hyperparameters functions__ for each scenario sequentially and returns the best model, hyperparameters and performance metrics and the _Save_model function__ saves this data accordingly.


![Alt text](project_images/Figure_18_evaluate_all_models.PNG)
Figure 17 - __evaluate_all_models__ function




_Find the overall classification model_

The __find_best_model function__ is adapted to take in a keyword argument called task_folder. This is to ensure the function finds the correct models (i.e regression or classification). 
The precision score was decided as the performance metric to find the best model. The __find_best_model function__ loads the precision for each model scenario and appends to a list, then a for loop is coded to find the highest precision score and then returns the model, hyperparameters and performance metrics. The gradient boosting  algorithm contained the lowest RMSE and was the best regression model for price night model scenario.

__find the best overall regression model__

In order to find the best overall regression model the performance metric(RMSE) needs to be compared against each model scenario. The __find_best_model function__ loads the RMSE for each model scenario and appends to a list, then a for loop is coded to find the lowest RMSE which will decide the best overall regression model and then returns the model, hyperparameters and performance metrics. The gradient boosting  algorithm contained the lowest RMSE and was the best regression model for price night model scenario.



&nbsp;

__Refactoring__

The first step was to review and refractor the code written in milestone 2. This included;

* Renaming methods and variables so that they are clear and concise to any who reads the script.
* Ensuring that the appropriate methods were made private.
* Re-ordering the sequence of the imports required for the code to run in alphabetical order.
* Adding docstrings to methods.

 These improvements makes the code look clearer and more user friendly.

&nbsp;

__Unit testing__

The second step was to set up unit tests for each public method. This was done by creating a test.py file which contains &ensp;__class producttestcase method__&ensp; to test each method. The main  purpose of tests is to ensure each public method returns the expected data type (string,list,dictionary) and to ensure the scrapper is correctly scrapping all the books from each page. This is to ensure that the code is processing the correct data as expected. Each unit test passed for each method.


__Project management__

The last step is to organise and add the relevant files which will ensure the scripts is packaged correctly. This includes adding;

* Renaming the python script as 'WaterstonesScrapper.py' and placing the script into a project folder.
* Placing the test file into a test folder.
* Creating a requirements.txt file which contains the external dependencies and versions.
* Creating a setup.py and setup.cfg which contains the meta data of the project and packages which need to be installed.
* Creating README.md file 
* Creating a license file which describes the license of the project.
* Creating a gitignore file.

## Milestone 4 - Containerising the scraper

&nbsp;

__Headless mode__

After confirming the unit tests still run, the next step was to run the scraper file in headless mode without the GUI. This was done so that the script could be run correctly in docker. The correct&ensp;__options arguments__&ensp; were coded into the __init method__ to allow the headless mode to work.

![Alt text](project_images/Milstone%204%20-%20options%20arguments.PNG)
*Figure 11 - Options arguments*


&nbsp;

__Docker image__

In order to build the docker image a docker file which contains the instructions on how to build the image is first created. A docker account was also created in order to upload the image file. The desktop app was downloaded.

The docker file contains the following;

* From - The base image for the docker image(python).
* Copy - Copies everything in the docker file directory (requirements.txt, scraper folder) into the container.
* Run -  Installs the required dependencies for the script to run. 
* CMD - Specifies the instruction that is to be executed when a Docker container starts.

&nbsp;

![Alt text](project_images/Milstone%204%20-%20docker%20file.PNG)
*Figure 12 - Dockerfile*


&nbsp;


The next step is to build the image using the docker build command.

&nbsp;

__Docker container__

&nbsp;

Now that the docker image is built the next step is to run the docker container using the docker run command. The script within the container ran fine with no issues. The container is then pushed onto docker hub.

&nbsp;

## Milestone 5 - Set up a CI/CD pipeline for your docker image

&nbsp;

in order to fully automate the docker image build and container run, it was first required to set up Github actions on the repository. 

__Create repository__
&nbsp;

The first step is to go yo the actions section in the repository on github and create two GitHub secrets actions. 

The first is a secret is called DOCKER_HUB_USERNAME which containes the name of the dockerhub account created and the second is called OCKER_HUB_ACCESS_TOKEN which contained a Personal Access Token (PAT) generated on dockerhub.

__Set up the workflow__
&nbsp;

The next step is to set up the GitHub Actions workflow for building and pushing the image to Docker Hub. This is done by going to the actions section on the repo and selecting set up workflow which creates a Github actions work file contained in yaml format.

&nbsp;

__Define the workflow steps__
&nbsp;

The  last step includes setting up the build context within the yaml file. The contains all the information for docker hub to copy to files mentioned in the dockerfile then build an image and automatically push to docker hub.

The last step is to commit the changes in the repo which would automatically start workflow. In order to make sure the workflow worked the image pushed on to docker hub was downloaded and a container was created and ran to ensure the script ran correctly.  A docker compose file which contains commands to self automate running containers was also created.