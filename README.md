# Predicting Demand for Medical Supplies and Medicines

This project develops a predictive modeling pipeline to estimate future demand for essential medicines and medical supplies using global health, demographic, and behavioral data.

## Problem Statement

How can we forecast critical shortages and healthcare needs for medicines and supplies in low-resource countries, using data on disease prevalence, healthcare infrastructure, and population demographics?

## Project Workflow

1. **Data Collection & Cleaning**
   - Sources: WHO, World Bank, and other global health datasets.
   - Features: Disease prevalence (diabetes, obesity, hypertension), behavioral risk factors (smoking, inactivity), healthcare infrastructure, population demographics.
   - Standardizes country names and codes, handles missing values, merges datasets.

2. **Feature Engineering**
   - Constructs new variables (elderly ratio, health access index, demand per million).
   - Selects relevant features for each research question.

3. **Modeling**
   - Uses machine learning models (XGBoost, Random Forest, MLP) for regression and classification.
   - Splits data into training and test sets.
   - Handles class imbalance for shortage prediction.

4. **Evaluation**
   - Metrics: RMSE, MAE, R² (regression); accuracy, precision, recall, F1-score (classification).
   - Compares results to simple baselines and linear models.

5. **Interpretation & Visualization**
   - Analyzes feature importance.
   - Visualizes results and country-level predictions.

6. **Policy Insights**
   - Provides actionable insights for health system planning and early warning for medicine shortages.

## Main Research Questions

1. **Early Warning System:** Predict critical shortages of essential medicines in low-resource countries.
2. **Health Access Index:** Forecast and improve national health access index using demographic, behavioral, and economic data.
3. **Health Index Prediction:** Predict a country's overall health index based on behavioral and environmental health factors.
4. **Healthcare Needs Estimation:** Estimate national healthcare needs by modeling elderly ratio, healthcare access, economic indicators, and tobacco control.
5. **Infrastructure Access:** Predict health infrastructure access from age demographics and income.
6. **Diabetes Care Equity:** Assess equity of diabetes care across countries with different income levels and infrastructures.
7. **Health Access Index Prediction:** Predict a country's health access index using demographic, behavioral risk factors, and economic indicators.
8. **Youth Demographics & Tobacco:** Analyze the role of youth demographics in shaping tobacco-related health outcomes.

## Usage

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   Or install manually:
   ```sh
   pip install pycountry xgboost scikit-learn pandas matplotlib seaborn missingno
   ```

2. **Run the main script:**
   ```sh
   python copy_of_thesis.py
   ```

3. **Outputs:**
   - Cleaned datasets (`*_cleaned.csv`)
   - Master dataset (`master_cleaned.csv`)
   - Model evaluation metrics and visualizations

## File Structure

- [`copy_of_thesis.py`](copy_of_thesis.py): Main analysis and modeling pipeline.
- `README.md`: Project overview and instructions.
- `requirements.txt`: Python dependencies (optional).
- Data files: Downloaded and processed CSVs.

## Results

- High accuracy and recall for shortage prediction.
- Moderate-to-good fit for health access index forecasting (R² ≈ 0.64).
- Feature importance highlights economic and policy factors as key drivers.

## License

This project is for academic research purposes.

## Acknowledgements
Special thanks to [Prof. Dr. Iftikhar Ahmed ](https://www.linkedin.com/in/iftikhar-ahmed-279a02145/)  
[ University of Europe for Applied Sciences ] (https://www.linkedin.com/school/university-of-europe-for-applied-sciences/posts/?feedView=all), for guidance and supervision.

## References


## Additional Resources

- [Colab Notebook] (https://colab.research.google.com/drive/1R7iJZVAyovOUgnup7aFc1Mzy_WWzdJW1)
- [Overleaf Paper] (https://www.overleaf.com/project/686ec339fdbd3b759619da62)
- [Overleaf PDF] (https://github.com/Saeedeh8858/facial-emotion-recognition-ml/blob/main/Machine_Learning_B_Facial_Emotion_Recognition.pdf)

**Author:**  
Saeedeh Alamkar
https://github.com/Saeedeh8858
