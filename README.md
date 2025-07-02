# House Price Prediction Project

## Overview
This project demonstrates a complete machine learning pipeline for predicting house prices using multiple linear regression. The model analyzes various features of residential properties to predict their sale prices, providing valuable insights for real estate valuation.

## Project Structure
- `house_price_prediction.ipynb`: Jupyter notebook containing the complete analysis and model development
- `house_price_model.pkl`: Serialized trained model for making predictions
- `scaler.pkl`: StandardScaler object for feature normalization
- `selected_features.pkl`: List of features selected for the final model
- `test.csv`: Test dataset with house features
- `model_predictions.csv`: Model predictions on test data

## Features
- **Data Exploration & Visualization**: Comprehensive analysis of the housing dataset with visualizations to understand feature relationships
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling
- **Feature Engineering**: Creating new features and selecting the most relevant ones
- **Model Development**: Implementation of multiple linear regression with regularization techniques
- **Model Evaluation**: Assessment using metrics like RMSE, MAE, and R² score
- **Prediction**: Generating price predictions on new data

## Technologies Used
- Python 3
- Libraries:
  - pandas: Data manipulation and analysis
  - numpy: Numerical operations
  - scikit-learn: Machine learning algorithms and evaluation metrics
  - matplotlib & seaborn: Data visualization
  - pickle: Model serialization

## Dataset
The dataset contains various features of residential properties including:
- Structural attributes (square footage, number of rooms, etc.)
- Location information
- Quality and condition ratings
- Year built and remodeled
- Various amenities and special features

## Model Performance
The model demonstrates strong predictive capability with:
- Evaluation of residuals to assess prediction accuracy
- Analysis of feature importance to understand key price determinants
- Cross-validation to ensure model robustness

## How to Use

### Prerequisites
- Python 3.x
- Required libraries (install via pip):
```
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Project
1. Clone or download this repository
2. Open the Jupyter notebook:
```
jupyter notebook house_price_prediction.ipynb
```
3. To make predictions with the saved model:
```python
import pickle
import pandas as pd

# Load the model and preprocessing objects
model = pickle.load(open('house_price_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
selected_features = pickle.load(open('selected_features.pkl', 'rb'))

# Load and prepare new data
new_data = pd.read_csv('your_new_data.csv')
X_new = new_data[selected_features]
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_new_scaled)
```

## Future Improvements
- Experiment with advanced algorithms (Random Forest, Gradient Boosting, Neural Networks)
- Implement hyperparameter tuning for model optimization
- Add more feature engineering techniques
- Create a web interface for real-time predictions

## License
This project is open source and available for educational and personal use.

## Author
Mugesh

---

*This project was developed as part of a data science portfolio to demonstrate skills in machine learning and predictive modeling.*