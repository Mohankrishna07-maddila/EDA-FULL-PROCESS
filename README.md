# 🌸 Iris Dataset – Exploratory Data Analysis & Preprocessing

## 1. Objective
The goal of this notebook is to explore and understand the Iris dataset by:
- Inspecting data structure and quality
- Checking missing and duplicate values
- Performing basic statistical analysis
- Visualizing feature distributions
- Preparing the dataset for further analysis or modeling

This notebook focuses on **data understanding**, not machine learning models.

---

## 2. Dataset Description

The Iris dataset is a well-known dataset used for classification and statistical analysis.

### Features

| Column Name     | Description                    |
|-----------------|--------------------------------|
| sepal_length    | Sepal length (cm)              |
| sepal_width     | Sepal width (cm)               |
| petal_length    | Petal length (cm)              |
| petal_width     | Petal width (cm)               |
| species         | Type of iris flower            |

### Target Classes
- Setosa  
- Versicolor  
- Virginica  

---

## 3. Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Purpose

pandas → data handling

numpy → numerical operations

matplotlib / seaborn → data visualization

4. Loading the Dataset
python
Copy code
df = pd.read_csv("iris.csv")
Initial inspection:

python
Copy code
df.head()
df.shape
df.info()
Why this is important

Confirms column names

Checks data types

Verifies number of rows and columns

5. Data Quality Checks
5.1 Missing Values
python
Copy code
df.isnull().sum()
Explanation

isnull() identifies missing values

sum() counts missing values per column

✔ Result: No missing values found.

5.2 Duplicate Records
python
Copy code
df.duplicated().sum()
Explanation

duplicated() flags repeated rows

sum() gives the total count of duplicates

Duplicate records can distort analysis and should be removed if present.

6. Statistical Summary
python
Copy code
df.describe()
Provides:

Mean

Standard deviation

Minimum and maximum values

Quartiles (Q1, Median, Q3)

Used to understand data spread and detect anomalies.

7. Species Distribution
python
Copy code
df["species"].value_counts()
Purpose

Checks class balance

Important for future modeling steps

8. Data Visualization
8.1 Histogram (Distribution Analysis)
python
Copy code
plt.hist(df["petal_length"], bins=20)
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()
Interpretation

Shows how values are distributed

Reveals skewness and gaps

Indicates possible class separation

8.2 Box Plot (Outlier Detection)
python
Copy code
sns.boxplot(x=df["sepal_width"])
What it shows

Median

Quartiles

Potential outliers

9. Species-wise Visualization
python
Copy code
sns.histplot(data=df, x="petal_length", hue="species", kde=True)
Purpose

Compares feature distributions across species

Helps identify discriminative features

10. Outlier Detection Using IQR Method
python
Copy code
Q1 = df["sepal_width"].quantile(0.25)
Q3 = df["sepal_width"].quantile(0.75)
IQR = Q3 - Q1
Filtering valid values:

python
Copy code
df_filtered = df[
    (df["sepal_width"] >= Q1 - 1.5 * IQR) &
    (df["sepal_width"] <= Q3 + 1.5 * IQR)
]
Why 1.5 × IQR?
Standard statistical rule

Captures most normal data

Flags extreme values effectively

11. Key Insights
Dataset is clean and well-structured

No missing values

Petal features separate species better than sepal features

Setosa is clearly distinguishable

Versicolor and Virginica show partial overlap

12. Conclusion
This analysis:

Validates dataset quality

Explores feature distributions

Identifies patterns and relationships

Prepares the data for modeling

Next steps

Feature scaling

Train-test split

Classification models (KNN, SVM, Logistic Regression)