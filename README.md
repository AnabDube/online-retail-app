Online Retail Data Analysis App

An interactive Streamlit web application for automated Exploratory Data Analysis (EDA), data cleaning, visualization, and machine learning model building.

With this app, users can upload any CSV dataset and instantly:

Inspect basic dataset statistics and info

Identify and handle missing values

Generate automatic, relevant visualizations

Prepare the dataset for machine learning

Apply and compare different ML models

Download cleaned datasets and results

Features
1. Automated EDA

Displays .head(), .info(), and .describe() output

Lists missing values per column

Shows correlation heatmap for numerical features

2. Data Cleaning

Detects columns with missing values

Handles missing data (drop or impute)

Summarizes cleaning actions taken

3. Auto Visualizations

Histograms for numerical distributions

Bar plots for categorical variables

Boxplots/violin plots for category comparisons

Scatter plots for numerical relationships

Line plots for time-series data (if applicable)

4. Machine Learning Preparation

Encodes categorical variables

Scales numerical features

Prepares clean DataFrame ready for modeling

5. Machine Learning Models

Lets users select a target variable

Applies multiple ML models

Reports accuracy, precision, recall, and F1-score

How to Use
Option 1 — Online (Recommended)

Visit the live app:
Live App Link Here

Upload your CSV file

Navigate through the tabs for EDA, Cleaning, Visualizations, and ML

Download cleaned datasets and results

Option 2 — Run Locally
Prerequisites

Python 3.8+

pip

Installation
git clone https://github.com/yourusername/online-retail-app.git
cd online-retail-app
pip install -r requirements.txt

Run the App
streamlit run app.py

Dataset

You can use your own dataset or publicly available datasets such as the Online Retail Dataset.

Example Output

(Add screenshots or GIFs of your app running here for visual appeal.)

License

This project is open-source and available under the MIT License.
