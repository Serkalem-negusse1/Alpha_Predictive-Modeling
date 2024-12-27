import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
def load_data(file_path):
    return pd.read_csv(file_path)

# Summary Statistics
def data_summary(df):
    print("Descriptive Statistics:")
    print(df.describe())
    print("\nData Types:")
    print(df.dtypes)

# Missing Values
def check_missing(df):
    print("\nMissing Values:")
    print(df.isnull().sum())

# Univariate Analysis
def univariate_analysis(df, column):
    if df[column].dtype in ['float64', 'int64']:
        sns.histplot(df[column], kde=True)
    else:
        sns.countplot(data=df, x=column)
    plt.title(f"Distribution of {column}")
    plt.show()

# Bivariate Analysis
def bivariate_analysis(df, col1, col2):
    sns.scatterplot(data=df, x=col1, y=col2)
    plt.title(f"Relationship between {col1} and {col2}")
    plt.show()

if __name__ == "__main__":
    file_path = "E:/10Academy/Data03/MachineLearningRating_v3.txt"
    df = load_data(file_path)
    
    data_summary(df)
    check_missing(df)
    
    # Example Plots
    univariate_analysis(df, "TotalPremium")
    bivariate_analysis(df, "TotalPremium", "TotalClaims")
