import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_file, output_file):
    # Load the dataset
    data = pd.read_csv(input_file, delimiter='|')
    
    # Fill missing values
    data.fillna(method='ffill', inplace=True)

    # Encode categorical variables
    categorical_columns = ['Province', 'Gender', 'VehicleType']
    label_encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))
        label_encoders[col] = encoder

    # Save the processed data
    data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data("E:/10Academy/Data03/MachineLearningRating_v3.txt", "E:/10Academy/Data03/processed_data.csv")
