# Heart Disease Classification

This repository contains a Jupyter notebook for a **Heart Disease Classification** project that uses a Decision Tree model to predict the presence or absence of heart disease based on patient data. The project includes data loading, exploratory data analysis (EDA), model training, and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [Usage Terms](#usage-terms)
- [Acknowledgments](#acknowledgments)

## Project Overview
The Heart Disease Classification project aims to predict whether a patient has heart disease (1) or not (0) using a Decision Tree classifier. The pipeline includes:
- Loading and exploring a dataset of patient attributes.
- Performing exploratory data analysis (EDA) to understand data distributions.
- Training a Decision Tree model using Scikit-learn.
- Evaluating the model with accuracy, precision, recall, and F1-score metrics.

## Features
- Predicts heart disease using a Decision Tree classifier.
- Includes EDA with data summaries and visualizations.
- Provides detailed model evaluation with classification metrics.
- Visualizes the Decision Tree structure.

## Dataset
The dataset is sourced from [this GitHub repository](https://github.com/PrathamSahani/heart-diseasses-prediction/blob/main/Heart/heart.csv) and contains 1025 instances with 14 attributes, including:
- **Age**: Patient's age in years.
- **Gender**: 0 (female), 1 (male).
- **ChestPain**: Type of chest pain (0–3).
- **RestingBP**: Resting blood pressure (mm Hg).
- **Cholesterol**: Serum cholesterol (mg/dL).
- **FastingBS**: Fasting blood sugar > 120 mg/dL (1 = true, 0 = false).
- **RestECG**: Resting ECG results (0–2).
- **MaxHR**: Maximum heart rate achieved.
- **ExerciseAngina**: Exercise-induced angina (1 = yes, 0 = no).
- **STDepression**: ST depression induced by exercise.
- **STSlope**: Slope of the peak exercise ST segment (0–2).
- **NumVessels**: Number of major vessels (0–3).
- **Thalassemia**: Thalassemia type (0–3).
- **HeartDisease**: Target variable (0 = no disease, 1 = disease).

The dataset (`heart.csv`) is not included in this repository due to size constraints. Download it from the source and place it in the project root directory.

## Technologies Used
- **Python 3.8+**
- **Pandas**: Data loading and manipulation.
- **NumPy**: Numerical operations.
- **Matplotlib/Seaborn**: Data visualization.
- **Scikit-learn**: Decision Tree model and evaluation metrics.
- **Jupyter Notebook**: Interactive development environment.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/heart-disease-classification.git
   cd heart-disease-classification
   ```
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
4. **Download the Dataset**:
   - Download `heart.csv` from [here](https://github.com/PrathamSahani/heart-diseasses-prediction/blob/main/Heart/heart.csv).
   - Place it in the project root directory.
5. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Open `Heart Disease Classification.ipynb` in the browser.

## Usage
1. **Run the Notebook**:
   - Open `Heart Disease Classification.ipynb` in Jupyter Notebook.
   - Execute the cells sequentially to:
     - Load and explore the dataset.
     - Perform EDA (e.g., view data shape, summary statistics).
     - Train the Decision Tree model.
     - Evaluate predictions and visualize the tree.
2. **Interpret Results**:
   - Check the model’s accuracy (~88.7%) and classification report for precision, recall, and F1-score.
   - Review the Decision Tree visualization for insights into the model’s decision-making process.

## Project Structure
```
heart-disease-classification/
├── heart.csv                # Dataset (download separately)
├── Heart Disease Classification.ipynb  # Main Jupyter notebook
├── requirements.txt         # Dependencies (optional, create if needed)
├── README.md                # This file
```

## Results
- **Model**: Decision Tree Classifier.
- **Accuracy**: 88.7% on the test set.
- **Classification Report**:
  - **Class 0 (No Disease)**: Precision = 0.85, Recall = 0.91, F1-score = 0.88.
  - **Class 1 (Disease)**: Precision = 0.92, Recall = 0.87, F1-score = 0.89.
- **Interpretation**:
  - The model correctly identifies 91% of no-disease cases and 87% of disease cases.
  - High precision for disease cases (92%) indicates reliable positive predictions.

## Contributing
1. Fork the repository.
2. Create a branch: `git checkout -b feature-branch`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push: `git push origin feature-branch`.
5. Open a Pull Request.

## Usage Terms
This project currently has no license. All rights are reserved, and you must contact the repository owner for permission to use, modify, or distribute the code.

## Acknowledgments
- Scikit-learn, Pandas, and Matplotlib communities for their open-source libraries.
- Jupyter Notebook for enabling interactive data analysis.
