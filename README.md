# %% [markdown]
#             imports

# %%
import pandas as pd 
import numpy as np

# %%
# csv reading 
df = pd.read_csv('iris.csv')
# showing sample data
print(df.head())

# %%
# shape and columns
print(f"shape : {df.shape}\n")
print(f"features : {df.columns}")

# %%
# dimension of the dataframe
print(f"the dataframe is {df.ndim} dimensional")

# %%
# size of the data set
print(f"size of the data set is {df.size}")

# %%
# transpose of the data frame    
transposed_df = df.T
df.T


# %%
# info of the data set 
print(f"DATASET INFO \n{df.info()}")

# %%
# sample data 
print(df.sample())

# %%
# memory usage of each feature
print(f"\nmemory usage of each feature : \n{df.memory_usage()}")

# %%


# %%
print(f"DATA TYPES :\n{df.dtypes}")

# %%
# count of null and duplicate values using count that returns total count of rows
a=df.isnull().count()
print(f"\nMISSING VALUES :\n{a}")
b=df.duplicated().count()
print(f"\nDUPLICATE VALUES : {b}")

# %%
# count of null and duplicate values using sum that returns total sum of nulls and duplicates
a=df.isnull().sum()
print(f"\nMISSING VALUES :\n{a}")
b=df.duplicated().sum()
print(f"\nDUPLICATE VALUES : {b}")

# %% [markdown]
#                                 Core difference of count and sum 
# 
# count():-
# 
# 👉 Counts rows, NOT problems
# 
# sum():-
# 
# 👉 Counts actual problems (True values)

# %%
# describing a data set like mean, min, max, std, percentiles
print(df.describe())

# %% [markdown]
#             mean of individual

# %%
print(df["sepal_length"].mean())

# %%
print(df["sepal_width"].mean())

# %%
print(df["petal_length"].mean())

# %%
print(df["petal_width"].mean())

# %% [markdown]
# max = maximum value ;;;;;;;
# min = minimum value ;;;;;;;
# std_dev = Standard Deviation. This tells you how spread out the numbers are. A high std (like 1.76 for petal length) means the sizes vary a lot; a low std (like 0.43 for sepal width) means most flowers are very similar in that category.;;;;;;;
# here 25% = ist quartile;;;;;;
# 75% = 3rd quartile ;;;;;;;
# 50% = median ;;;;;;;;

# %% [markdown]
# this data set is univariety data set 

# %% [markdown]
#                     individual skews for different features

# %% [markdown]
# Interpretation (memorize this):
# 
# ≈ 0 → Normal distribution
# 
# > +0.5 → Right-skewed (positively skewed)
# 
# < -0.5 → Left-skewed (negatively skewed)

# %%
print(df["sepal_length"].skew())

# %%
print(df["sepal_width"].skew())

# %%
print(df["petal_length"].skew())

# %%
print(df["petal_width"].skew())

# %% [markdown]
# logical check of skewing using mean and median for column - petal length

# %%
a=df["petal_length"].mean()#mean of the petal length
b=df["petal_length"].median()#median of the petal length
if a > b:
    print("right skewed")
elif a < b:
    print("left skewed")
else:
    print("normal distribution")
print(f"mean : {a}\nmedian : {b}")

# %% [markdown]
# for petal width

# %%
a=df["petal_width"].mean()#mean of the petal width
b=df["petal_width"].median()#median of the petal width
if a > b:
    print("right skewed")
elif a < b:
    print("left skewed")
else:
    print("normal distribution")
print(f"mean : {a}\nmedian : {b}")

# %% [markdown]
# for sepal length

# %%
a=df["sepal_length"].mean()#mean of the sepal length
b=df["sepal_length"].median()#median of the sepal length
if a > b:
    print("right skewed")
elif a < b:
    print("left skewed")
else:
    print("normal distribution")
print(f"mean : {a}\nmedian : {b}")

# %% [markdown]
# for sepal width

# %%
a=df["sepal_width"].mean()#mean of the sepal width
b=df["sepal_width"].median()#median of the sepal width
if a > b:
    print("right skewed")
elif a < b:
    print("left skewed")
else:
    print("normal distribution")
print(f"mean : {a}\nmedian : {b}")

# %%
df["species"].value_counts()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.figure(figsize=(10,6))

for species in df["species"].unique():
    data = df[df["species"] == species]["petal_length"]

    # Histogram (density=True for probability)
    plt.hist(data, bins=15, density=True, alpha=0.5, label=species)
    # Gaussian curve
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    plt.plot(x, norm.pdf(x, mu, sigma), linewidth=2)

plt.xlabel("Petal Length")
plt.ylabel("Density")
plt.title("Petal Length Distribution by Species (True Bell Curves)")
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt
plt.boxplot(df["petal_length"])
plt.show()


# %%
import matplotlib.pyplot as plt
plt.boxplot(df["petal_width"])
plt.show()


# %%
import matplotlib.pyplot as plt
plt.boxplot(df["sepal_length"])
plt.show()

# %%
import matplotlib.pyplot as plt
plt.boxplot(df["sepal_width"])
plt.show()


# %%
import matplotlib.pyplot as plt

data = df["sepal_width"]   # or the column you used for the box plot

plt.figure(figsize=(12,5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(data, bins=15, edgecolor="black")
plt.xlabel("sepal Width")
plt.ylabel("Frequency")
plt.title("Histogram")

# Box plot
plt.subplot(1, 2, 2)
plt.boxplot(data, vert=True)
plt.ylabel("sepal Width")
plt.title("Box Plot")

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

plt.hist(df["petal_length"], bins=20)
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# %%
plt.hist(df["petal_width"], bins=20)
plt.xlabel("Petal width")
plt.ylabel("Frequency")
plt.show()


# %%
plt.hist(df["sepal_length"], bins=20)
plt.xlabel("sepal Length")
plt.ylabel("Frequency")
plt.show()


# %%
plt.hist(df["sepal_width"], bins=20)
plt.xlabel("sepal Width")
plt.ylabel("Frequency")
# plt.margins(x=0, y=0)
plt.show()


# %%
print(df.head())

# %% [markdown]
# here we understand that petals have 2 groups of the range 1-2 is one group and 3-7 is another group for only petals for sepals all are bell shaped 

# %%
# %pip install nbformat

# %%
import seaborn as sns
import plotly.express as px

# 1. LOAD THE DATA (This defines 'df')
df = sns.load_dataset('iris')

# 2. CREATE THE PLOT
fig = px.scatter(df, 
                 x="sepal_width", 
                 y="sepal_length", 
                 color="species")
fig.show()


# %%
fig=px.box(df,x="sepal_width")
fig.show()

# %%
# data cleaning to remove outliers using IQR method
Q1 = df["sepal_width"].quantile(0.25)
Q3 = df["sepal_width"].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[
    (df["sepal_width"] >= Q1 - 1.5 * IQR) &
    (df["sepal_width"] <= Q3 + 1.5 * IQR)
]
fig=px.box(df_clean,x="sepal_width")
fig.show()

# %%
# finding the outlier ranges to see which rows have those outlier values
outliers = df[df["sepal_width"].isin([2, 4.1, 4.2, 4.4])]
print(outliers)

# %%
# using the not operation to remove those outlier rows
df_clean = df[~df["sepal_width"].isin([2, 4.1, 4.2, 4.4])]
fig=px.box(df_clean,x="sepal_width")
fig.show()

# %%
df_clean.head(61)

# %%
print(f"before outliers removed data:\n \n{df.describe()}\n")
print(f"\nshape of the dataset before removing outliers: {df.shape}\n")
print(f"outliers removed data:\n \n{df_clean.describe()}")
print(f"\nshape of the dataset after removing outliers: {df_clean.shape}")

# %%
# for mean pattern that comparing of mean of cleaned and uncleaned data:
a=(df_clean["sepal_width"].mean())
b=(df["sepal_width"].mean())
if a > b:
    print("mean increased after removing outliers")
elif a < b:
    print("mean decreased after removing outliers")
else:
    print("mean remained the same after removing outliers")

# %%
# for median pattern that comparing of mean of cleaned and uncleaned data:
a=(df_clean["sepal_width"].median())
b=(df["sepal_width"].median())
if a > b:
    print("median increased after removing outliers")
elif a < b:
    print("median decreased after removing outliers")
else:
    print("median remained the same after removing outliers")

# %%
a=(IQR)
print(a)

# %%
# for std pattern that comparing of mean of cleaned and uncleaned data:
a=(df_clean["sepal_width"].std())
b=(df["sepal_width"].std())
if a > b:
    print("std increased after removing outliers")
elif a < b:
    print("std decreased after removing outliers")
else:
    print("std remained the same after removing outliers")

# %%
# for std pattern that comparing of mean of cleaned and uncleaned data:
a=(df_clean["sepal_width"].25%)
b=(df["sepal_width"].25%)
if a > b:
    print("std increased after removing outliers")
elif a < b:
    print("std decreased after removing outliers")
else:
    print("std remained the same after removing outliers")


