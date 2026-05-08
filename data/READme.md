# Crop Recommendation Dataset

## Overview

This dataset is designed to help in the recommendation of crops based on various environmental and soil conditions. It includes data on different parameters such as temperature, humidity, pH, and rainfall, which are crucial for determining the suitability of a crop for a particular region.

### Source:

- **Dataset URL**: [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

---

## Dataset Features

| Feature     | Description                                     | Type   | Units |
| ----------- | ----------------------------------------------- | ------ | ----- |
| Nitrogen    | Nitrogen content in the soil                    | Float  | kg/ha |
| Phosphorus  | Phosphorus content in the soil                  | Float  | kg/ha |
| Potassium   | Potassium content in the soil                   | Float  | kg/ha |
| Temperature | Temperature of the region                       | Float  | °C    |
| Humidity    | Humidity level in the region                    | Float  | %     |
| pH Value    | Acidity or alkalinity of the soil               | Float  | pH    |
| Rainfall    | Rainfall in the region                          | Float  | mm    |
| Crop Label  | Recommended crop based on soil and climate data | String | N/A   |

---

## Dataset Usage

### Example Use Cases:

- **Crop Recommendation Systems**: Suggesting the best crop to grow in a specific region based on soil nutrients and climate.
- **Soil Analysis**: Understanding the relationship between soil content and the crops that can grow best in different conditions.

### Data Preprocessing:

- **Missing Values**: Any missing data has been handled by removing the rows with missing values.
- **Normalization**: Feature scaling was applied to the numerical variables to normalize the data.
- **Label Encoding**: The target variable (crop) has been label-encoded for machine learning purposes.

### Train-Test Split:

- The dataset was split into 70% training data and 30% test data for evaluation and drift detection.

---

### File Format

- The file is in CSV (Comma-Separated Values) format.
- Each row represents a unique set of environmental conditions and the corresponding recommended crop.
- The first row contains the column headers.

### Usage

This file can be used for:

- Training machine learning models to predict the best crop for given environmental conditions.
- Conducting data analysis to understand the relationship between environmental factors and crop suitability.
- Supporting decision-making processes in agricultural planning and management.

### Notes

- Ensure that the data is preprocessed and cleaned before using it for any analysis or model training.
- The dataset may need to be updated periodically to reflect changes in environmental conditions and agricultural practices.
