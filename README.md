# 🏡 Energy Efficiency Prediction Model

This project builds a **Random Forest Regression** model to predict the **Energy Efficiency** of a building using synthetic architectural features such as wall area, roof area, glazing, and overall height. It includes data visualization, model training, and evaluation steps.

## 📊 Dataset

The dataset is **synthetically generated** for demonstration purposes and includes the following features:

- `WallArea`: Total wall surface area (in square meters)
- `RoofArea`: Total roof surface area (in square meters)
- `OverallHeight`: Overall height of the building (in meters)
- `GlazingArea`: Proportion of the wall that is glazed (between 0 and 1)
- `EnergyEfficiency`: Target variable indicating the efficiency score

## 📈 Features

- Data generation using NumPy
- Data visualization using Seaborn and Matplotlib
- Train/test split
- Model training with `RandomForestRegressor`
- Evaluation using Mean Squared Error (MSE)
- Visual comparison of predicted vs. actual values
