# Neural_Network_Charity_Analysis
Applying knowledge in Neural Networks and Deep Learning models using the Python TensorFlow and Pandas libraries to preprocess and create an optimal predictive binary classifier.


## Overview of the Analysis
![neural_image](https://user-images.githubusercontent.com/94148420/166151677-8db65c82-5bdf-47e1-b5d4-8c1efc5f15d0.gif)

Charitable foundations and orgainzations that provide funding for worthy causes are coming under increased scrutiny.  Theses foundations and organizations, as well as their stakeholders, are looking to optimize their funding dollars.  With newly gained knowledge of machine learning and neural networks, features in the provided charity dataset are used to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

Alphabet Soup’s business team provided a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

The ultimate goal is to utilize the newly created binary classifier to predict whether an organization that is applying for funding will be successful.

### Resources
#### Code:
* AlphabetSoupCharity.ipynb
* AlphabetSoupCharity_Optimization.ipynb
* AlphabetSoupCharity_Optimization_v1.ipynb
* AlphabetSoupCharity_Optimizaiton_v2.ipynb
* AlphabetSoupCharity_Optimizaiton_v3.ipynb

#### Data:
* https://github.com/1on1pt/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv

#### Software/Data Tools/Libraries:
* Jupyter Notebook 6.4.6
* Python 3.7.11
* scikit-learn
* StandardScaler
* tensorflow
* OneHotEncoder
* ModelCheckpoint
* keras_tuner

## Results
The goal was to optimize the deep neural model in order to achieve a target predictive accuracy higher than 75%.  The original model was set up as to not achieve this level of accuracy. There were three additional versions of the deep neural model created in attempt to reach this goal.  There was also an attempt to use keras_tuner to identify the best hyperparameters.
* AlphabetSoupCharity.ipynb (original model)
* AlphabetSoupCharity_Optimization_v1.ipynb
* AlphabetSoupCharity_Optimizaiton_v2.ipynb
* AlphabetSoupCharity_Optimization_v3.ipynb
* AlphabetSoupCharity_Optimization.ipynb (keras_tuner version)

### Data Preprocessing
In the original model:
* The **EIN** and **NAME** columns were removed as these columns offered no added value
* **APPLICATION_TYPE** was binned; unique values with less than 500 records were categorized as "Other"
* **CLASSIFICATION** was also binned; value counts less than 1800 were categorized as "Other"
* The *target* was **IS_SUCCESSFUL**
* The *features* were the 43 remaining variables, i.e. STATUS, ASK_AMOUNT, APPLICATION_TYPE_Other, APPLICATION_TYPE_T10, etc.

Here is the resulting X_train.shape:

![X_train_shape](https://user-images.githubusercontent.com/94148420/166154489-d30ab652-138f-41bf-af78-efac175b142f.PNG)


### Compiling, Training, and Evaluating the Model
In the original model:
* 5981 parameters; 43 inputs; 2 hidden layers; 1 output layer
* The Rectified Linear Unit (RELU) activation function was used for the first and second layers
* The Sigmoid activation function was used for the output layer
* Epochs = 100
* nn.summary()

![nn_summary_original](https://user-images.githubusercontent.com/94148420/166155170-a8fa1080-eefa-479b-81fe-58855836018e.PNG)


Target performance for accuracy was set at 75%.  The original model was only able to achieve an accuracy rate of 72.5%.

![accuracy_original](https://user-images.githubusercontent.com/94148420/166155299-fa9ead10-1866-48da-a68e-f789d989dbd8.PNG)


### Optimizing the Model to Improve Accuracy Rate

An initial attempt to use keras_tuner to identify hyperparameters proved to be very time consuming, so this strategy was abandoned.  Here is an example of one of the attempts to use keras_tuner, which only resulted in an accuracy rate of 72.7%.

![keras_tuner_attempt](https://user-images.githubusercontent.com/94148420/166155602-398feefe-9c1d-4455-94d7-71573694eabc.PNG)

Various strategies were attempted to improve the accuracy score, but none of these attempts were successful.  See the summary of results below:

![model_comparison](https://user-images.githubusercontent.com/94148420/166156620-314f4101-e09c-4a43-bb56-d30bce525ccc.PNG)

#### Optimization_v1
* Increased neurons in layer 1 to **100**
* Increased neurons in layer 2 to **80**
* **Added third layer with 40 neurons**
* Activation for layers = 'relu'
* Activation for output = 'sigmoid'
* Epochs = 100
* Overall parameters increased to **14,181**
* Overall decreased performance with accuracy rate **decreased by 0.19%** compared to the original model

![nn_v1_summary](https://user-images.githubusercontent.com/94148420/166156835-d3e5c071-2498-4bad-9f08-ce74a4ce5655.PNG)


#### Optimization_v2
* **Dropped 'ORGANIZATION' column**
* Neurons in layer 1 changed to **80**
* Neurons in layer 2 changed to **60**
* Neurons in layer 3 remained at 40
* **Added fourth layer with 20 neurons**
* Activation for layers = 'relu'
* Activation for output = 'sigmoid"
* Epochs **decreased to 90**
* Overall parameters now at **12,181**
* Overall decreased performance with accuracy rate **decreased by 0.33%** compared to the original model

![nn_v2_summary](https://user-images.githubusercontent.com/94148420/166157033-aa8bb0da-f989-4770-824e-14e25fdf46a3.PNG)


#### Optimization_v3
* APPLICATION_TYPE binning was changed; unique **values with less than 1000** records were categorized as "Other"
* CLASSIFICATION binning was changed; **value counts less than 2100** were categorized as "Other"
* Two hidden layers were used for this model
* Neurons in layer 1 = **100**
* Neurons in layer 2 = **60**
* **Activation for layers changed to 'sigmoid'**
* Activation for output remained at 'sigmoid'
* Epochs **increased to 125**
* Overall parameters now at **10,021**
* Overall decreased performance with accuracy rate **decreased by 0.44%** compared to the original model

![nn_v3_summary](https://user-images.githubusercontent.com/94148420/166157259-109258a5-a403-4eff-a3c6-dc9da8c4ba89.PNG)


## Summary
Developing neural network and deep learning models to achieve a high degree of accuracy while minimizing loss can prove to be quite challenging.  There are so many options to consider for a model with an end result of maximizing accuracy.  Those options can be changed and influenced in the **data preprocessing phase** of developing the model or in the **compiling, training, and evaluating the model** phase.  The keras_tuner was attempted to identify hyperparameters, but this proved to be very time consuming.  Adding hidden layers and neurons, and thus increasing the overall parameters did not improve the accuracy with this data set.  It appears that a model with fewer parameters than what was used in the optimizing phase of this analysis might help to increase accuracy.  Recommendations would also include continued fine tuning in the data preprocessing phase as well as another attempt with using the keras_tuner to identify hyperparameters.
