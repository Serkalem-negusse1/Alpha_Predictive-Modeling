import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Import r2_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import numpy as np


class DataModeling:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        # Encode categorical variables
        self.data = pd.get_dummies(self.data, columns=['Province', 'Gender'], drop_first=True)

    def classification(self):
        # Features and target
        X = self.data[['Premium', 'Zipcode', 'Province_B', 'Province_C']]
        y = (self.data['Total_Claim'] > self.data['Total_Claim'].median()).astype(int)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Models to train
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        results = {}
        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred)
            }

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n{name} Confusion Matrix:")
            print(cm)

            # Plot Confusion Matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"{name} Confusion Matrix")
            plt.show()

        return results

    def regression(self):
        # Features and target
        X = self.data[['Premium', 'Zipcode', 'Province_B', 'Province_C']]
        y = self.data['Total_Claim']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Models to train
        models = {
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }

        results = {}
        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Store results for the model (convert np.float64 to float for better display)
            results[name] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2 Score": float(r2)  # Ensure R² is a regular float
            }

            # Debugging: Print the metrics for the current model
            print(f"{name} Metrics:")
            print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        return results


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("... data/insurance_data.csv")

    # Initialize and preprocess
    modeler = DataModeling(data)
    modeler.preprocess_data()

    # Classification results
    print("Classification Results:")
    classification_results = modeler.classification()

    # Regression results
    print("\nRegression Results with R²:")
    regression_results = modeler.regression()

    # Print regression results
    for model_name, metrics in regression_results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
