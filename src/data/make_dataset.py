
"""
This module is responsible for creating and processing the dataset for crop
classification.

It includes functions for loading data, preprocessing, and saving the dataset
for further analysis or model training.
"""
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def process_data():
    """
    Load and preprocess the dataset.

    This function loads the dataset from a specified source, performs any
    necessary preprocessing steps, and prepares the data for analysis.

    Returns:
        DataFrame: A pandas DataFrame containing the processed dataset.
    """

    # Define the path to the raw data
    raw_data_path = os.path.join('data', 'raw', 'Crop_Recommendation.csv')

    # Check if the raw data file exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"The file {raw_data_path} does not exist. Please check the path."
        )

    # Read the raw data
    df = pd.read_csv(raw_data_path)
    logger.info(
        "Loaded data: %d rows, %d columns", df.shape[0], df.shape[1]
    )

    # Split the data into features (X) and target (y)
    X = df.drop('Crop', axis=1)  # Target variable is 'Crop'
    y = df['Crop']

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Combine features and target for train and test sets
    train_data = pd.concat([X_train.reset_index(drop=True),
                            y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test.reset_index(drop=True),
                           y_test.reset_index(drop=True)], axis=1)

    # Create processed data directory if it doesn't exist
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Save train and test data to CSV files
    train_data.to_csv(
        os.path.join(processed_dir, 'train.csv'),
        index=False
    )
    test_data.to_csv(
        os.path.join(processed_dir, 'test.csv'),
        index=False
    )
    logger.info(
        f"Train Set: {len(train_data)} | "
        f"Test Set: {len(test_data)}"
    )

    logger.info(
        "Data processing complete."
        "Train and test datasets saved to 'data/processed/' directory."
    )

    return train_data, test_data


if __name__ == "__main__":
    process_data()
