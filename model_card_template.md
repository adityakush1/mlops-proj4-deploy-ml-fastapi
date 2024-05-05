# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Our mission is to predict whether someone's income exceeds the 50K mark annually. For this endeavor, we've harnessed the power of a GradientBoostingClassifier, fine-tuned with the utmost precision using the cutting-edge hyperparameter optimization capabilities of scikit-learn version 1.2.0. Through meticulous exploration via GridSearchCV, we've unearthed the ideal combination of hyperparameters, ensuring peak performance in our predictive model.
- learning_rate: 0.3
- max_depth: 5
- min_samples_split: 100
## Intended Use
The predictive prowess of this model extends to estimating an individual's salary level, drawing insights from a curated set of attributes. Tailored for students, academics, or research enthusiasts, its application transcends traditional boundaries, offering a versatile tool for exploration and analysis in various educational and research contexts.
## Training Data
The Census Income Dataset, sourced from the UCI Machine Learning Repository, arrives in CSV format with 32,561 rows and 15 columns. These columns encompass the target label "salary," comprising two classes ('<=50K', '>50K'), alongside 8 categorical features and 6 numerical features. Further insights into each feature can be gleaned from the UCI link provided.

Notably, the target label "salary" exhibits class imbalance, with a distribution ratio of approximately 75% to 25%. A preliminary data cleansing procedure was enacted to eliminate leading and trailing whitespaces. For a comprehensive overview of data exploration and cleansing steps, refer to the "data_cleaning.ipynb" notebook.

Subsequently, the dataset underwent an 80-20 split, segregating it into distinct training and testing sets. Stratification based on the target label "salary" was implemented to ensure a balanced representation of classes across both sets. To prepare the data for training, categorical features were subjected to One Hot Encoding, while the target label underwent label binarization.
## Evaluation Data
A segment comprising 20% of the dataset was reserved for model evaluation purposes. To prepare the categorical features and target label for modeling, transformations were applied using the One Hot Encoder and label binarizer, respectively. Notably, these transformations were fitted solely on the training set to maintain consistency and prevent data leakage.
## Metrics
- Precision: 0.783
- Recall: 0.662
- fbeta: 0.717
- Confusion Matrix: [[4657  288]
 [ 530 1038]]

## Ethical Considerations
It's important to note that the dataset may not accurately reflect the true distribution of salaries and should not be extrapolated to make assumptions about the salary levels of specific population categories. The dataset's limitations, biases, and potential sampling issues should be taken into consideration when interpreting any findings or conclusions drawn from it.
## Caveats and Recommendations
The dataset was extracted from the 1994 Census database, indicating its outdated nature. As such, it cannot be deemed a statistically representative sample of the population. While its applicability for statistical analyses may be limited, it remains a valuable resource for training machine learning classification models or tackling related problems in the field.