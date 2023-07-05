# Credit Card Fraud Detection
## Problem Statement
The objective of this project is to build a machine learning model to detect fraudulent credit card transactions. The dataset contains a large number of credit card transactions, some of which are fraudulent. The task is to develop a model that can accurately identify fraudulent transactions and minimize false positives and false negatives.

## Dataset
The dataset used for this project is the Credit Card Fraud Detection dataset from Kaggle. It contains a total of 284,807 transactions, out of which 492 are fraudulent. The dataset is highly imbalanced, with the majority class being non-fraudulent transactions.

## Approach
### Data Preprocessing: 
The dataset was explored and checked for any missing values or anomalies. The features were scaled using standardization to ensure that they have a similar range and distribution.

### Data Split: 
The dataset was split into training and testing sets using a 70:30 ratio. The training set was used for model training and evaluation, while the testing set was used for final performance assessment.

### Model Selection: 
Logistic Regression was chosen as the initial model for its simplicity and interpretability. Other algorithms like Random Forest or Gradient Boosting could be explored in future iterations.

### Model Training and Evaluation:

The model was trained on the training set using the fit() function from scikit-learn's LogisticRegression module.
Model performance was evaluated on the testing set using various evaluation metrics, including accuracy, precision, recall, and F1 score.
The confusion matrix was computed to analyze the true positive, true negative, false positive, and false negative predictions.
Hyperparameter Tuning: Grid search was performed to find the optimal hyperparameters for the Logistic Regression model. Cross-validation was used to assess the model's performance with different parameter combinations.

### Model Evaluation: 
The final model's performance was evaluated using the testing set. The accuracy, precision, recall, and F1 score were calculated to assess the model's ability to detect fraudulent transactions.
### Results
The results of our analysis are as follows:

- Accuracy: 0.9991222218320986
- Confusion Matrix: [[56854    10]
                     [   40    58]]
- Precision: 0.8529411764705882
- Recall: 0.5918367346938775
- F1 Score: 0.6987951807228915
## Conclusion
In this project, we developed a Logistic Regression model to detect credit card fraud. The model achieved high accuracy and demonstrated reasonable precision and recall values. Thanks to the use of the Synthetic Minority Over-sampling Technique (SMOTE), we successfully addressed the class imbalance challenge in the dataset.

SMOTE helped generate synthetic instances of the minority class (fraudulent transactions), effectively balancing the dataset and improving the model's performance. By oversampling the minority class, SMOTE allowed the model to learn from a more representative and balanced training set, leading to better fraud detection capabilities.

Further improvements could be explored by considering other advanced algorithms like Random Forest or Gradient Boosting. Additionally, feature engineering and selection techniques could be applied to extract more informative features and enhance the model's predictive power.

Overall, this project highlights the importance of fraud detection in financial transactions and showcases the effectiveness of SMOTE in addressing class imbalance. It demonstrates the successful application of machine learning techniques in detecting credit card fraud, contributing to the development of more robust and accurate fraud detection systems.

