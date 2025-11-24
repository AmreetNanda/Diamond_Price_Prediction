# Diamond Price prediction App (Machine Learning + Flask + Docker)

The Diamond Price Prediction Web App is an interactive, user-friendly online tool that forecasts diamond prices based on a number of factors. This project shows how a machine learning model can be integrated with a web interface to give users who want to estimate the value of their diamonds a smooth experience.
It includes:

- A complete machine learning pipeline
- A Flask web UI for model training 
- A fully modular ML codebase
- Optional Docker container for deployment
---

## Features

- Input Form: Users can enter a diamond's carat, depth, table, dimensions (x, y, z), size, cut, color, and clarity, among other characteristics.
- Machine Learning Prediction: Based on the given attributes, the application makes predictions about the diamond's price using a trained machine learning model.
- User-Friendly Interface: Users may submit data and receive forecasts with ease because to the web application's appealing and simple interface.
- Dataset :A collection of diamond features and their associated prices makes up the dataset utilized in the Diamond Price Prediction Web App project. A machine learning model that can forecast a diamond's price based on its several attributes is trained using the dataset. The dataset is a useful tool for comprehending the connections between the characteristics of diamonds and their market values.

## Attributes in the Dataset:
- carat: Carat (ct.) refers to the unique unit of weight measurement used exclusively to weigh gemstones and diamonds.
- cut: Quality of Diamond Cut.
- color: Color of Diamond.
- clarity: Diamond clarity is a measure of the purity and rarity of the stone, graded by the visibility of these characteristics under 10-power magnification.
- depth: The depth of the diamond is its height (in millimeters) measured from the culet (bottom tip) to the table (flat, top surface).
- table: A diamond's table is the facet which can be seen when the stone is viewed face up.
- x: Diamond X dimension.
- y: Diamond Y dimension.
- z: Diamond Z dimension.

## Technologies Used:
- Front-End: HTML, CSS
- Back-End: Python (Flask framework)
- MachineLearning: Linear Regression, Lasso Regression, Ridge Regression, ElasticNet, Decision Tree Regressor

## Project Structure
```bash
Diamond_Price_Prediction/
│
├── app.py # Flask app
├── Diamond_Price_Prediction/ # Modular ML package
│ ├── EDA
│ ├── Template
│ │   ├── form.html
│ │   └── index.html
│ ├── src
│ │   ├── components
│ │   │   ├── __init__.py
│ │   │   ├── data_ingestion.py
│ │   │   ├── data_transformation.py
│ │   │   └── model_trainer.py
│ │   ├── pipeline
│ │   │   ├── __init__.py
│ │	  │   ├── predict_pipeline.py
│ │   │   └── training_pipeline.py
│ │   ├── Exception.py
│ │   ├── Logger.py
│ │   └── Utils.py
├── requirements.txt # Python dependencies
├── Dockerfile # Docker container
├── run.sh # Optional to run the script
└── setup.py # Optional to run the script
```
```
## Installation

## 🛠 Installation (without Docker)

### 1. Clone the repo
```bash
git clone https://github.com/AmreetNanda/Diamond_Price_Prediction.git
cd Diamond_Price_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Flask app
```bash
run app.py
```
Open in your browser:
👉 http://127.0.0.1:5000/
👉 Enter the attributes of the diamond in the input form.
👉 Click the "Predict Price" button.
👉 Receive the predicted price of the diamond.

## 🐳 Running with Docker (optional)
### Build the image
```bash
docker build -t Diamond_Price_Prediction .
```

### Run the container
```bash
docker run -p 8501:8501 diamond_price_prediction
```
Open: 👉 http://localhost:8501

## Screenshots

##### Home page
![App Screenshot](https://github.com/AmreetNanda/Diamond_Price_Prediction/blob/main/Diamond_Price_Prediction_1.png)

##### Form page
![App Screenshot](https://github.com/AmreetNanda/Diamond_Price_Prediction/blob/main/Diamond_Price_Prediction_2.png)

## Demo
https://github.com/user-attachments/assets/5ef8d5c0-cb8a-4602-b26d-3180f93be919

## License

[MIT](https://choosealicense.com/licenses/mit/)
