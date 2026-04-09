 Task 1: Data Cleaning & Preprocessing (Titanic Dataset)

This project focuses on cleaning and preprocessing the Titanic dataset to make it suitable for machine learning.

In this task, the dataset is first loaded using Python and Pandas. Basic information such as data types, structure, and missing values is explored to understand the dataset.

Missing values are handled by filling the Age column with the median value and the Embarked column with the mode value. The Cabin column is removed because it contains too many missing values. Any remaining null values are also removed.

Categorical columns like Sex and Embarked are converted into numerical form using Label Encoding, which helps machine learning models understand the data.

Numerical features such as Age and Fare are standardized using StandardScaler to bring all values to a similar scale.

Outliers in the dataset are detected using boxplots and removed using the IQR (Interquartile Range) method to improve data quality.

Finally, the cleaned dataset is saved as a new file named "cleaned_titanic.csv".

This task helps in understanding the importance of data preprocessing, handling missing data, encoding categorical variables, feature scaling, and outlier removal in machine learning.
