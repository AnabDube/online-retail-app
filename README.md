# Automated EDA and Machine Learning App

A comprehensive Streamlit application that provides automated Exploratory Data Analysis (EDA), data cleaning, visualization, and machine learning capabilities in a user-friendly interface.

## Features

### 📊 Basic EDA
- Data preview and information
- Statistical summary
- Missing values analysis
- Correlation heatmap for numeric columns

### 🧹 Data Cleaning
- Automatic missing value detection
- Options to drop or impute missing values
- Download cleaned dataset

### 📈 Automatic Visualizations
- Bar plots for categorical variables
- Histograms and boxplots for numeric variables
- Comprehensive EDA report generation

### 🤖 Machine Learning Preparation
- Automatic label encoding for categorical variables
- Feature scaling using StandardScaler
- Data preparation for modeling

### 🧠 Machine Learning
- Automatic problem type detection (Classification/Regression)
- Multiple model options:
  - Classification: Logistic Regression, Decision Tree, Random Forest
  - Regression: Linear Regression, Decision Tree, Random Forest
- Performance metrics and visualizations
- Model download and comprehensive reports

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Upload your CSV file through the file uploader

3. Navigate through the tabs:
   - **Basic EDA**: Get initial insights about your data
   - **Data Cleaning**: Handle missing values and clean your dataset
   - **Auto Visualizations**: Generate automatic plots for all variables
   - **ML Preparation**: Prepare data for machine learning
   - **Machine Learning**: Train and evaluate models

4. Download results:
   - Cleaned datasets
   - EDA reports
   - Trained models
   - Model performance reports

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## File Structure

```
online-retail-app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Example Usage

1. Upload a CSV file containing your dataset
2. Review the basic statistics and missing values in the EDA tab
3. Clean your data by handling missing values appropriately
4. Explore automatic visualizations for each variable
5. Prepare your data for machine learning
6. Select a target variable and train machine learning models
7. Download the trained model and performance report

## Notes

- The app automatically detects whether you're working on a classification or regression problem
- For best results, ensure your data is properly formatted before uploading
- Large datasets may take longer to process, especially for visualizations and model training
- The app includes error handling for common data processing issues

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
