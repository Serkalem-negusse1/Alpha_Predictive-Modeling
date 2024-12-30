#Scripts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the preprocessed data
data = pd.read_csv("... data/processed_data.csv")

# Perform basic EDA
def eda():
    print(data.describe())
    print(data.info())

    # Plot histogram of TotalPremium
    plt.hist(data['TotalPremium'], bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Total Premium')
    plt.xlabel('Total Premium')
    plt.ylabel('Frequency')
    plt.show()

    # Correlation matrix
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Logistic Regression Model
def logistic_regression():
    X = data[['Age', 'VehicleType', 'Province']]  # Example features
    y = data['TotalClaim']  # Target variable
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    eda()
    logistic_regression()
