# scripts/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class pmEDA:
    def __init__(self, file_path):
        """Initialize EDA with the dataset path."""
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load the dataset into a DataFrame."""
        self.df = pd.read_csv(self.file_path)
        print("Data loaded successfully.")
        return self.df

    def data_summary(self):
        """Print descriptive statistics and data types."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Descriptive Statistics:")
        print(self.df.describe())
        print("\nData Types:")
        print(self.df.dtypes)

    def check_missing(self):
        """Check for missing values in the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def univariate_analysis(self, column):
        """Perform univariate analysis on a given column."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.df[column].dtype in ['float64', 'int64']:
            sns.histplot(self.df[column], kde=True)
        else:
            sns.countplot(data=self.df, x=column)
        plt.title(f"Distribution of {column}")
        plt.show()

    def bivariate_analysis(self, col1, col2):
        """Perform bivariate analysis between two columns."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        sns.scatterplot(data=self.df, x=col1, y=col2)
        plt.title(f"Relationship between {col1} and {col2}")
        plt.show()


# Standalone execution when script is run
if __name__ == "__main__":
    file_path = "E:/10Academy/Data03/processed_data.csv"  # Adjust the path as needed
    eda = pmEDA(file_path=file_path)
    
    # Run EDA operations
    ##df = eda.load_data()
    ##eda.data_summary()
    ##eda.check_missing()
    ##eda.univariate_analysis("TotalPremium")  # Replace with a valid column
    ##eda.bivariate_analysis("TotalPremium", "TotalClaims")  # Replace with valid columns