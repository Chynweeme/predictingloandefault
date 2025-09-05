# Predicting Loan Defaults: A Machine Learning Approach
1. Introduction
In the dynamic world of finance, accurately predicting loan defaults is paramount. It directly impacts risk management and informs crucial lending decisions. This project dives into a machine learning methodology to tackle this challenge, leveraging historical loan data to build a model capable of classifying loan applications as ‘Good’ (non-default) or ‘Bad’ (default).

2. Data Loading and Initial Exploration
Data sets:
A. "https://raw.githubusercontent.com/Oyeniran20/axia_cohort_8/refs/heads/main/trainperf.csv"
B. "https://raw.githubusercontent.com/Oyeniran20/axia_cohort_8/refs/heads/main/trainperf.csv"
C. "https://raw.githubusercontent.com/Oyeniran20/axia_cohort_8/refs/heads/main/traindemographics.csv"
Our journey began by loading three datasets. These datasets contained vital loan performance information, including ‘customerid’, ‘systemloanid’, ‘loannumber’, ‘approveddate’, ‘creationdate’, ‘loanamount’, ‘totaldue’, ‘termdays’, ‘referredby’, and our target variable, ‘good_bad_flag’. We noted missing values primarily in the ‘referredby’ column.

Complementing this, the traindemographics.csv dataset offered demographic details such as ‘customerid’, ‘birthdate’, ‘bank_account_type’, ‘longitude_gps’, ‘latitude_gps’, ‘bank_name_clients’, ‘bank_branch_clients’, ‘employment_status_clients’, and ‘level_of_education_clients’.

This dataset presented a more significant challenge with substantial missing values in ‘bank_branch_clients’, ‘employment_status_clients’, and ‘level_of_education_clients’.

3. Data Cleaning and Feature Engineering

To prepare the data for robust modeling, we undertook several cleaning steps. Columns with a high proportion of missing values (‘referredby’, ‘bank_branch_clients’, ‘level_of_education_clients’) were dropped, as attempting to impute such a large amount of data could introduce considerable noise and potentially skew our results.

We also decided to drop ‘loannumber’ and ‘creationdate’ from the performance data, considering them less impactful for predicting loan outcomes. The two ‘trainperf.csv’ dataframes (df1 and df2) were concatenated to form a unified performance dataset (df4).

Similarly, the demographics dataframe (df3) was streamlined by removing location-specific columns (‘longitude_gps’, ‘latitude_gps’, ‘bank_name_clients’, ‘bank_branch_clients’) which were deemed less relevant for our prediction task. A key part of our feature engineering involved extracting the ‘birthyear’ from the ‘birthdate’ column in the demographics data and subsequently calculating the ‘age’ of each customer.

Once ‘age’ was computed, the original ‘birthdate’ and ‘birthyear’ columns were no longer needed and were dropped. The cleaned performance and demographics dataframes were then merged using ‘customerid’ as the common key, resulting in a comprehensive dataframe (df5). We identified and removed duplicate rows within df5 to ensure data integrity. Missing values in the newly created ‘age’ column were imputed with the mean age of the dataset.

For the remaining categorical columns with missing values (‘bank_account_type’, ‘employment_status_clients’), we applied a forward fill (‘ffill’) strategy to propagate the last valid observation forward. Crucially, our target variable ‘good_bad_flag’ initially represented as text (‘Good’ or ‘Bad’), was transformed into a numerical format (1 for ‘Good’, 0 for ‘Bad’). This binary representation is essential for compatibility with most machine learning algorithms.

4. Data Preparation

To rigorously evaluate our models’ ability to generalize to unseen data, the preprocessed and engineered dataset (df5) was partitioned into training and testing sets using ‘train_test_split’. A standard 20% of the data was allocated to the testing set. To mitigate the impact of the observed class imbalance, we employed ‘stratify=y’, which ensures that the proportion of ‘Good’ and ‘Bad’ loans is preserved in both the training and testing splits.

5. Data Processing

Prior to feeding the data into our models, feature scaling and encoding were necessary. Numerical features (‘loanamount’, ‘totaldue’, ‘termdays’, ‘age’) were scaled using ‘StandardScaler’. This process standardizes the features to have a mean of zero and a standard deviation of one, preventing features with larger values from dominating the learning process. Categorical features (‘approveddate’, ‘bank_account_type’, ‘employment_status_clients’) were converted into a numerical format using ‘OneHotEncoder’. This technique creates binary columns for each category, making them suitable for inclusion in our models. The scaled numerical features and the one-hot encoded categorical features were then combined to form the final feature sets for training (`X_train`) and testing (`X_test`).

6. Model Training

With the data prepared, we proceeded to train a suite of classification models to predict loan defaults. The models selected included: — Logistic Regression — Decision Tree — Random Forest — Gradient Boosting — XGBoost Each of these models was trained independently on the preprocessed training data (‘X_train’) and its corresponding target variable (‘y_train’).

7. Model Evaluation and Comparison

Following training, the performance of each model was assessed using key metrics: accuracy, precision, and recall, on both the training and testing datasets. Initially, models such as Decision Tree and Random Forest exhibited perfect accuracy on the training data but showed a noticeable drop in performance on the test data, a clear indication of overfitting. Logistic Regression, in contrast, displayed a more consistent performance across both sets. However, the initial evaluation revealed a high precision score (1.00) for Logistic Regression, which, while seemingly positive, raised concerns about the model being overly cautious and potentially missing actual default cases (false negatives).

8. Handling Class Imbalance

The initial evaluation results underscored the presence of a significant class imbalance, where the ‘Good’ loan instances far outnumbered the ‘Bad’ loan instances. To address this, we implemented the Synthetic Minority Over-sampling Technique (SMOTE) on the training data. SMOTE works by generating synthetic samples for the minority class (‘Bad’ loans), effectively balancing the class distribution within the training set (‘X_train_resampled’, ‘y_train_resampled’).

9. Results After Addressing Class Imbalance

Re-evaluating the models after training on the SMOTE-resampled data provided a more informative picture of their performance. While overall accuracy might have seen slight adjustments, the precision and recall metrics offered a deeper understanding, as visually represented by the confusion matrices and detailed in the metrics table below:

Press enter or click to view image in full size

Post-SMOTE, Logistic Regression and Random Forest continued to demonstrate strong performance with high test accuracy and precision. Gradient Boost and XGBoost, while showing improved recall (better at identifying potential defaults), exhibited lower precision, indicating a higher rate of false positives. Considering the trade-offs, Logistic Regression emerges as a robust choice, striking a good balance between identifying potential defaults (recall) and minimizing incorrect positive predictions (precision) after addressing the class imbalance. Random Forest also presents a compelling option with its high precision. The ultimate selection of the best model would necessitate considering the specific costs associated with false positives and false negatives within the business context.

10. Conclusion

This project successfully navigated the machine learning pipeline for predicting loan defaults, from the initial stages of data loading and cleaning through to model training and comprehensive evaluation. We effectively addressed common data challenges, including missing values, duplicate entries, and crucially, class imbalance.

Our exploration of various classification models revealed that Logistic Regression and Random Forest delivered promising results, particularly after employing SMOTE to balance the dataset.

Future enhancements to this project could involve more sophisticated feature engineering techniques, exploring alternative resampling or cost-sensitive learning methods to further refine the handling of class imbalance, hyperparameter tuning of the selected models for optimal performance, and potentially investigating more advanced machine learning architectures. The insights gleaned from this analysis offer valuable guidance for financial institutions aiming to enhance their lending decisions and effectively mitigate associated risks.

Summary:
Data Analysis Key Findings
The project involved building a machine learning model to predict loan defaults based on historical performance and demographic data.
Initial data exploration revealed missing values in several columns, particularly in demographic information, and duplicate entries.
Data cleaning involved dropping columns with a high percentage of missing values and irrelevant identifiers or location data.
Feature engineering included calculating customer age from birthdate and transforming the target variable into a numerical format.
The cleaned datasets were merged, and missing values in age were imputed with the mean, while categorical missing values were filled using a forward fill strategy.
The data was split into training and testing sets using stratification to maintain the proportion of loan outcomes.
Numerical features were scaled using StandardScaler, and categorical features were one-hot encoded before model training.
Several classification models were trained, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
Initial model evaluation indicated potential overfitting in some models and highlighted a class imbalance issue.
The Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data to address class imbalance.
After applying SMOTE, Logistic Regression and Random Forest demonstrated strong performance with high test accuracy and precision, while Gradient Boost and XGBoost showed improved recall but lower precision.
Logistic Regression was identified as a strong candidate due to its balance between identifying defaults and minimizing false positives after resampling.
Insights or Next Steps
The choice of the best model should consider the specific business costs associated with false positives versus false negatives.
Further work could explore more advanced feature engineering, alternative resampling techniques, hyperparameter tuning, and potentially more complex modeling approaches to improve predictive performance.




