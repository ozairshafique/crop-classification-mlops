# Crop Recommendation Dataset

## Overview

This dataset is designed to help in the recommendation of crops based on various environmental and soil conditions. It includes data on different parameters such as temperature, humidity, pH, and rainfall, which are crucial for determining the suitability of a crop for a particular region.

## License

Dataset sourced from Kaggle —
[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
Please refer to Kaggle's terms of use.

---

## Dataset Statistics

| Property           | Value      |
| ------------------ | ---------- |
| Total Samples      | 2200       |
| Training Samples   | 1760 (80%) |
| Test Samples       | 440 (20%)  |
| Number of Features | 7          |
| Number of Classes  | 22         |
| Missing Values     | None       |

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

## Feature Ranges

| Feature     | Min   | Max    | Mean   |
| ----------- | ----- | ------ | ------ |
| Nitrogen    | 0     | 140    | 50.55  |
| Phosphorus  | 5     | 145    | 53.36  |
| Potassium   | 5     | 205    | 48.14  |
| Temperature | 8.83  | 43.67  | 25.61  |
| Humidity    | 14.26 | 99.98  | 71.48  |
| pH Value    | 3.50  | 9.94   | 6.47   |
| Rainfall    | 20.21 | 298.56 | 103.46 |

---

## Crop Classes (22)

| #   | Crop        | #   | Crop       |
| --- | ----------- | --- | ---------- |
| 1   | Rice        | 12  | Mango      |
| 2   | Maize       | 13  | Grapes     |
| 3   | Chickpea    | 14  | Watermelon |
| 4   | KidneyBeans | 15  | Muskmelon  |
| 5   | PigeonPeas  | 16  | Apple      |
| 6   | MothBeans   | 17  | Orange     |
| 7   | MungBean    | 18  | Papaya     |
| 8   | Blackgram   | 19  | Coconut    |
| 9   | Lentil      | 20  | Cotton     |
| 10  | Pomegranate | 21  | Jute       |
| 11  | Banana      | 22  | Coffee     |

---

## Dataset Usage

### Example Use Cases:

- **Crop Recommendation Systems**: Suggesting the best crop to grow in a specific region based on soil nutrients and climate.
- **Soil Analysis**: Understanding the relationship between soil content and the crops that can grow best in different conditions.

### Data Preprocessing:

- **Missing Values**: No missing values present in the original dataset.
- **Normalization**: Features kept in raw scale — Random Forest does not require feature scaling.
- **Label Encoding**: The target variable (crop) has been label-encoded for machine learning purposes.

### Train-Test Split:

| Split    | Samples | Percentage |
| -------- | ------- | ---------- |
| Training | 1760    | 80%        |
| Testing  | 440     | 20%        |

---

### File Format

- The file is in CSV (Comma-Separated Values) format.
- Each row represents a unique set of environmental conditions and the corresponding recommended crop.
- The first row contains the column headers.

### Usage

This file can be used for:

- Training machine learning models to predict the best crop for given environmental conditions.
- Conducting data analysis to understand the relationship between environmental factors and crop suitability.
- Supporting decision-making processes in agricultural planning decisions and management.

### Notes

- Ensure that the data is preprocessed and cleaned before using it for any analysis or model training.
- The dataset may need to be updated periodically to reflect changes in environmental conditions and agricultural practices.
