
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score


st.title("Automated EDA and Machine Learning App")

# Upload data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    st.success("File successfully uploaded!")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Basic EDA", "Data Cleaning", "Auto Visualizations", "ML Preparation", "Machine Learning"]
    )

    # TAB 1 - Basic EDA
    with tab1:
        st.header("Basic Exploratory Data Analysis")

        st.subheader("DataFrame Head")
        st.write(df.head())

        st.subheader("DataFrame Info")
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("DataFrame Description")
        st.write(df.describe())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for correlation heatmap.")

    # TAB 2 - Data Cleaning
    with tab2:
        st.header("Data Cleaning")
        missing_values = df.isnull().sum()
        missing_cols = missing_values[missing_values > 0].index.tolist()

        if not missing_cols:
            st.success("No missing values found!")
            df_cleaned = df.copy()
        else:
            st.warning(f"Found {len(missing_cols)} columns with missing values: {', '.join(missing_cols)}")
            cleaning_option = st.selectbox(
                "How to handle missing values?",
                ["Drop rows with missing values", "Impute missing values"]
            )

            if cleaning_option == "Drop rows with missing values":
                df_cleaned = df.dropna()
                st.write("Dropped rows with missing values.")
            else:
                df_cleaned = df.copy()
                for col in missing_cols:
                    if df_cleaned[col].dtype == 'object':
                        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                    else:
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                st.write("Imputed missing values with mode (categorical) and median (numerical).")

        st.subheader("Cleaned DataFrame Head")
        st.write(df_cleaned.head())

        st.subheader("Summary of Cleaning Actions")
        st.write(f"Original shape: {df.shape}")
        st.write(f"Cleaned shape: {df_cleaned.shape}")
        st.write(f"Number of dropped rows: {df.shape[0] - df_cleaned.shape[0]}")

        # Download button for cleaned dataset
        csv = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )

    # TAB 3 - Automatic Visualizations
    with tab3:
        st.header("Automatic Visualizations")
        if 'df_cleaned' not in locals():
            st.warning("Please clean the data in the 'Data Cleaning' tab first.")
        else:
            st.subheader("Visualizations")
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == 'object':
                    st.subheader(f"Bar plot for {col}")
                    fig, ax = plt.subplots()
                    sns.countplot(
                        y=col,
                        data=df_cleaned,
                        order=df_cleaned[col].value_counts().index[:10],
                        ax=ax
                    )
                    st.pyplot(fig)
                else:
                    st.subheader(f"Histogram for {col}")
                    fig, ax = plt.subplots()
                    sns.histplot(df_cleaned[col], kde=True, ax=ax)
                    st.pyplot(fig)

                    st.subheader(f"Boxplot for {col}")
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df_cleaned[col], ax=ax)
                    st.pyplot(fig)

    # TAB 4 - Machine Learning Preparation
    with tab4:
        st.header("Machine Learning Preparation")
        if 'df_cleaned' not in locals():
            st.warning("Please clean the data in the 'Data Cleaning' tab first.")
        else:
            df_ml = df_cleaned.copy()
            le = LabelEncoder()
            for col in df_ml.select_dtypes(include='object').columns:
                df_ml[col] = le.fit_transform(df_ml[col])

            scaler = StandardScaler()
            numeric_cols = df_ml.select_dtypes(include=np.number).columns
            df_ml[numeric_cols] = scaler.fit_transform(df_ml[numeric_cols])

            st.subheader("Prepared DataFrame for Machine Learning")
            st.write(df_ml.head())
            st.success("Data is now ready for modeling.")

    # TAB 5 - Apply Machine Learning
    with tab5:
        st.header("Apply Machine Learning Model")
        if 'df_ml' not in locals():
            st.warning("Please prepare the data in the 'ML Preparation' tab first.")
        else:
            target_variable = st.selectbox("Select the target variable", df_ml.columns)

            if target_variable:
                X = df_ml.drop(columns=[target_variable])
                y = df_ml[target_variable]

                if y.nunique() <= 10:
                    problem_type = "Classification"
                else:
                    problem_type = "Regression"

                st.write(f"Problem Type: {problem_type}")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if problem_type == "Classification":
                    models = {
                        "Logistic Regression": LogisticRegression(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Random Forest": RandomForestClassifier()
                    }
                else:
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest": RandomForestRegressor()
                    }

                for name, model in models.items():
                    st.subheader(f"Model: {name}")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.subheader("Model Evaluation")
                    if problem_type == "Classification":
                        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                        st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
                        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
                        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
                    else:
                        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
                        st.write(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")



