import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1. LOAD DATA
# -------------------------

df = pd.read_csv("data/List of Unicorns in the World.csv", encoding="latin1")
print("Dataset Loaded Successfully\n")
print("Total Companies:", len(df))
print("\nColumns:")
print(df.columns)


# -------------------------
# 2. COUNTRY ANALYSIS
# -------------------------

print("\nTop 5 Countries:")
print(df["Country"].value_counts().head(5))

top_countries = df["Country"].value_counts().head(5)

plt.figure()
top_countries.plot(kind="bar")
plt.title("Top 5 Countries with Unicorn Companies")
plt.xlabel("Country")
plt.ylabel("Number of Companies")
plt.show()


# -------------------------
# 3. VALUATION CLEANING
# -------------------------

valuation_column = "Valuation ($B)"   
df[valuation_column] = df[valuation_column].str.replace("$", "", regex=False)
df[valuation_column] = df[valuation_column].str.replace("B", "", regex=False)
df[valuation_column] = df[valuation_column].astype(float)
print("\nAverage Valuation (in Billions):", df[valuation_column].mean())


# -------------------------
# 4. VALUATION DISTRIBUTION
# -------------------------

plt.figure()
df[valuation_column].hist()
plt.title("Distribution of Unicorn Valuations")
plt.xlabel("Valuation (Billion $)")
plt.ylabel("Frequency")
plt.show()
print(df.columns)

# -------------------------
# 5.Industry Analysis
# -------------------------

print("\nTop 10 Industries:")
print(df["Industry"].value_counts().head(10))

top_industries = df["Industry"].value_counts().head(5)

plt.figure()
top_industries.plot(kind="bar")
plt.title("Top 5 Industries with Unicorn Companies")
plt.xlabel("Industry")
plt.ylabel("Number of Companies")
plt.show()

# -------------------------
# 6.Year-wise Growth Analysis
# -------------------------

df["Year"] = pd.to_datetime(df["Date Joined"]).dt.year

year_counts = df["Year"].value_counts().sort_index()

plt.figure()
year_counts.plot(kind="line")
plt.title("Unicorn Companies Growth Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Companies")
plt.show()

# Convert Date Joined to datetime
df["Date Joined"] = pd.to_datetime(df["Date Joined"])

# Extract Year
df["Year"] = df["Date Joined"].dt.year

print("\nYear-wise Unicorn Count:")
print(df["Year"].value_counts().sort_index())

print("\nTop 10 Highest Valuation Companies:")

top_10 = df.sort_values(by="Valuation ($B)", ascending=False).head(10)

print(top_10[["Company", "Valuation ($B)", "Country", "Industry"]])

plt.figure()
plt.bar(top_10["Company"], top_10["Valuation ($B)"])
plt.xticks(rotation=90)
plt.title("Top 10 Highest Valuation Unicorn Companies")
plt.xlabel("Company")
plt.ylabel("Valuation (Billion $)")
plt.show()

print("\nCountry-wise Average Valuation:")

country_avg = df.groupby("Country")["Valuation ($B)"].mean().sort_values(ascending=False)
print(country_avg.head(10))

top_country_avg = country_avg.head(5)

plt.figure()
top_country_avg.plot(kind="bar")
plt.title("Top 5 Countries by Average Valuation")
plt.xlabel("Country")
plt.ylabel("Average Valuation (Billion $)")
plt.show()

plt.figure()
plt.boxplot(df["Valuation ($B)"])
plt.title("Boxplot of Unicorn Valuations")
plt.ylabel("Valuation (Billion $)")
plt.show()

# Calculate Q1 and Q3
Q1 = df["Valuation ($B)"].quantile(0.25)
Q3 = df["Valuation ($B)"].quantile(0.75)

IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df["Valuation ($B)"] < lower_bound) | 
              (df["Valuation ($B)"] > upper_bound)]

print("\nQ1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("\nNumber of Outliers:", len(outliers))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Encode categorical columns
df_encoded = df.copy()

df_encoded = pd.get_dummies(df_encoded, columns=["Country", "Industry"], drop_first=True)

# Features and Target
X = df_encoded.drop(["Valuation ($B)", "Company", "City", "Date Joined"], axis=1)
y = df_encoded["Valuation ($B)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nMachine Learning Model Performance")
print("R2 Score:", r2_score(y_test, y_pred))

# Extract founding year from Date Joined (approximation)
df["Join_Year"] = df["Year"]

# Assume company became unicorn same year joined
# Calculate age since joining
current_year = 2024
df["Unicorn_Age"] = current_year - df["Join_Year"]

print("\nAverage Unicorn Age:", df["Unicorn_Age"].mean())

plt.figure()
df["Unicorn_Age"].hist()
plt.title("Distribution of Unicorn Age")
plt.xlabel("Years Since Becoming Unicorn")
plt.ylabel("Frequency")
plt.show()

industry_avg = df.groupby("Industry")["Valuation ($B)"].mean().sort_values(ascending=False)

print("\nIndustry-wise Average Valuation:")
print(industry_avg.head(10))

plt.figure()
industry_avg.head(5).plot(kind="bar")
plt.title("Top 5 Industries by Average Valuation")
plt.xlabel("Industry")
plt.ylabel("Average Valuation (Billion $)")
plt.show()

city_counts = df["City"].value_counts().head(10)

print("\nTop 10 Cities with Unicorn Companies:")
print(city_counts)

plt.figure()
city_counts.plot(kind="bar")
plt.title("Top 10 Cities with Unicorn Companies")
plt.xlabel("City")
plt.ylabel("Number of Companies")
plt.show()

year_avg_val = df.groupby("Year")["Valuation ($B)"].mean()

plt.figure()
year_avg_val.plot(kind="line")
plt.title("Average Valuation Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Average Valuation (Billion $)")
plt.show()

import numpy as np

df["Log_Valuation"] = np.log(df["Valuation ($B)"])

plt.figure()
df["Log_Valuation"].hist()
plt.title("Log Transformed Valuation Distribution")
plt.xlabel("Log Valuation")
plt.ylabel("Frequency")
plt.show()

year_counts = df["Year"].value_counts().sort_index()

growth_rate = year_counts.pct_change() * 100

plt.figure()
growth_rate.plot(kind="line")
plt.title("Yearly Growth Rate of Unicorn Companies (%)")
plt.xlabel("Year")
plt.ylabel("Growth Rate (%)")
plt.show()

from sklearn.cluster import KMeans

X_cluster = df[["Valuation ($B)"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_cluster)

plt.figure()
plt.scatter(df["Valuation ($B)"], df["Cluster"])
plt.title("Unicorn Clusters based on Valuation")
plt.xlabel("Valuation (Billion $)")
plt.ylabel("Cluster")
plt.show()

diversification = df.groupby("Country")["Industry"].nunique().sort_values(ascending=False)

print("Country Diversification Index:")
print(diversification.head(10))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest R2:", r2_score(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

import numpy as np

def gini(array):
    array = np.sort(array)
    n = len(array)
    cumulative = np.cumsum(array)
    gini_index = (2 * np.sum((np.arange(1, n+1) * array))) / (n * np.sum(array)) - (n + 1) / n
    return gini_index

gini_value = gini(df["Valuation ($B)"].values)

print("Gini Coefficient (Valuation Inequality):", gini_value)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features = df_encoded.drop(["Valuation ($B)", "Company", "City", "Date Joined"], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

year_counts = df["Year"].value_counts().sort_index()

y = year_counts.values

model = ExponentialSmoothing(y, trend="add")
fit = model.fit()

forecast = fit.forecast(5)

last_year = year_counts.index.max()
future_years = np.arange(last_year+1, last_year+6)

print("\nNext 5 Years Unicorn Forecast:")
for year, value in zip(future_years, forecast):
    print(f"{year} -> {round(value)}")