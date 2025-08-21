
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, confusion_matrix, classification_report
import pickle
import io
from datetime import datetime


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
            
            # Generate EDA Report
            eda_report = []
            eda_report.append("# Exploratory Data Analysis Report")
            eda_report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            eda_report.append(f"\n## Dataset Overview")
            eda_report.append(f"- Shape: {df_cleaned.shape}")
            eda_report.append(f"- Columns: {list(df_cleaned.columns)}")
            eda_report.append(f"- Data types: {df_cleaned.dtypes.to_dict()}")
            eda_report.append(f"\n## Missing Values")
            missing_summary = df_cleaned.isnull().sum()
            eda_report.append(f"{missing_summary.to_dict()}")
            eda_report.append(f"\n## Statistical Summary")
            eda_report.append(f"{df_cleaned.describe().to_string()}")
            
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == 'object':
                    st.subheader(f"Bar plot for {col}")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    try:
                        sns.countplot(
                            y=col,
                            data=df_cleaned,
                            order=df_cleaned[col].value_counts().index[:10],
                            ax=ax
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add to report
                        eda_report.append(f"\n## {col} (Categorical)")
                        eda_report.append(f"Value counts: {df_cleaned[col].value_counts().head(10).to_dict()}")
                        
                    except Exception as e:
                        st.error(f"Error creating bar plot for {col}: {str(e)}")
                    finally:
                        plt.close(fig)
                else:
                    st.subheader(f"Histogram for {col}")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    try:
                        sns.histplot(df_cleaned[col], kde=True, ax=ax)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add to report
                        eda_report.append(f"\n## {col} (Numerical)")
                        eda_report.append(f"Mean: {df_cleaned[col].mean():.4f}")
                        eda_report.append(f"Std: {df_cleaned[col].std():.4f}")
                        eda_report.append(f"Min: {df_cleaned[col].min():.4f}")
                        eda_report.append(f"Max: {df_cleaned[col].max():.4f}")
                        
                    except Exception as e:
                        st.error(f"Error creating histogram for {col}: {str(e)}")
                    finally:
                        plt.close(fig)

                    # Boxplot for numeric columns with enough data
                    if df_cleaned[col].dtype != 'object':
                        if df_cleaned[col].nunique(dropna=True) > 1:
                            st.subheader(f"Boxplot for {col}")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            try:
                                sns.boxplot(x=df_cleaned[col], ax=ax)
                                plt.tight_layout()
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error creating boxplot for {col}: {str(e)}")
                            finally:
                                plt.close(fig)
                        else:
                            st.info(f"Skipping boxplot for {col} — not enough unique values.")
            
            # Download EDA Report
            eda_report_text = '\n'.join(eda_report)
            st.download_button(
                label="Download EDA Report",
                data=eda_report_text.encode('utf-8'),
                file_name=f'eda_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                mime='text/plain'
            )

    # TAB 4 - Machine Learning Preparation
    with tab4:
        st.header("Machine Learning Preparation")
        if 'df_cleaned' not in locals():
            st.warning("Please clean the data in the 'Data Cleaning' tab first.")
        else:
            df_ml = df_cleaned.copy()
            le = LabelEncoder()
            for col in df_ml.select_dtypes(include='object').columns:
                df_ml[col] = le.fit_transform(df_ml[col].astype(str))

            # Select numeric columns again after encoding
            numeric_cols = df_ml.select_dtypes(include=[np.number]).columns

            # Handle any missing values in numeric columns
            df_ml[numeric_cols] = df_ml[numeric_cols].fillna(0)

            # Scale numeric features
            scaler = StandardScaler()
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
                try:
                    X = df_ml.drop(columns=[target_variable])
                    y = df_ml[target_variable]

                    # Check if we have enough data
                    if len(X) < 10:
                        st.error("Not enough data for machine learning. Need at least 10 samples.")
                        st.stop()

                    # Determine problem type
                    unique_values = y.nunique()
                    if unique_values <= 10:
                        problem_type = "Classification"
                        st.info(f"Detected Classification problem with {unique_values} unique classes")
                    else:
                        problem_type = "Regression"
                        st.info(f"Detected Regression problem with {unique_values} unique values")

                    # Model selection
                    if problem_type == "Classification":
                        available_models = {
                            "Logistic Regression": LogisticRegression(max_iter=1000),
                            "Decision Tree": DecisionTreeClassifier(random_state=42),
                            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
                        }
                    else:
                        available_models = {
                            "Linear Regression": LinearRegression(),
                            "Decision Tree": DecisionTreeRegressor(random_state=42),
                            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100)
                        }

                    # Model selection dropdown
                    selected_model_name = st.selectbox(
                        "Choose a model to train:",
                        list(available_models.keys())
                    )

                    # Train/test split
                    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
                    
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y if problem_type == "Classification" else None
                        )
                    except ValueError:
                        # If stratify fails, use without stratification
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )

                    if st.button("Train Model"):
                        with st.spinner(f"Training {selected_model_name}..."):
                            try:
                                # Get selected model
                                model = available_models[selected_model_name]
                                
                                # Train model
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                                st.success(f"✅ {selected_model_name} trained successfully!")

                                # Model Evaluation
                                st.subheader("📊 Model Performance")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if problem_type == "Classification":
                                        accuracy = accuracy_score(y_test, y_pred)
                                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                                        
                                        st.metric("Accuracy", f"{accuracy:.4f}")
                                        st.metric("Precision", f"{precision:.4f}")
                                        st.metric("Recall", f"{recall:.4f}")
                                        st.metric("F1 Score", f"{f1:.4f}")
                                        
                                        # Classification Report
                                        st.subheader("📋 Classification Report")
                                        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                                        st.text(classification_report(y_test, y_pred, zero_division=0))
                                        
                                    else:
                                        mse = mean_squared_error(y_test, y_pred)
                                        rmse = np.sqrt(mse)
                                        r2 = r2_score(y_test, y_pred)
                                        mae = np.mean(np.abs(y_test - y_pred))
                                        
                                        st.metric("Mean Squared Error", f"{mse:.4f}")
                                        st.metric("Root Mean Squared Error", f"{rmse:.4f}")
                                        st.metric("R² Score", f"{r2:.4f}")
                                        st.metric("Mean Absolute Error", f"{mae:.4f}")

                                with col2:
                                    if problem_type == "Classification":
                                        # Confusion Matrix
                                        st.subheader("🔥 Confusion Matrix")
                                        cm = confusion_matrix(y_test, y_pred)
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                        ax.set_xlabel('Predicted')
                                        ax.set_ylabel('Actual')
                                        ax.set_title('Confusion Matrix')
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    else:
                                        # Actual vs Predicted scatter plot
                                        st.subheader("📈 Actual vs Predicted")
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        ax.scatter(y_test, y_pred, alpha=0.7)
                                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                                        ax.set_xlabel('Actual')
                                        ax.set_ylabel('Predicted')
                                        ax.set_title('Actual vs Predicted Values')
                                        st.pyplot(fig)
                                        plt.close(fig)

                                # Feature Importance (for tree-based models)
                                if hasattr(model, 'feature_importances_'):
                                    st.subheader("🎯 Feature Importance")
                                    feature_importance = pd.DataFrame({
                                        'feature': X.columns,
                                        'importance': model.feature_importances_
                                    }).sort_values('importance', ascending=False)
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
                                    ax.set_title('Top 10 Feature Importances')
                                    st.pyplot(fig)
                                    plt.close(fig)

                                # Download trained model
                                st.subheader("💾 Download Trained Model")
                                model_buffer = io.BytesIO()
                                pickle.dump(model, model_buffer)
                                model_buffer.seek(0)
                                
                                st.download_button(
                                    label="Download Trained Model",
                                    data=model_buffer.getvalue(),
                                    file_name=f'{selected_model_name.lower().replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl',
                                    mime='application/octet-stream'
                                )

                                # Model summary report
                                report_lines = []
                                report_lines.append(f"# Machine Learning Model Report")
                                report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                report_lines.append(f"\n## Model Details")
                                report_lines.append(f"- Model: {selected_model_name}")
                                report_lines.append(f"- Problem Type: {problem_type}")
                                report_lines.append(f"- Target Variable: {target_variable}")
                                report_lines.append(f"- Training Set Size: {len(X_train)}")
                                report_lines.append(f"- Test Set Size: {len(X_test)}")
                                report_lines.append(f"- Features: {list(X.columns)}")
                                
                                if problem_type == "Classification":
                                    report_lines.append(f"\n## Performance Metrics")
                                    report_lines.append(f"- Accuracy: {accuracy:.4f}")
                                    report_lines.append(f"- Precision: {precision:.4f}")
                                    report_lines.append(f"- Recall: {recall:.4f}")
                                    report_lines.append(f"- F1 Score: {f1:.4f}")
                                else:
                                    report_lines.append(f"\n## Performance Metrics")
                                    report_lines.append(f"- Mean Squared Error: {mse:.4f}")
                                    report_lines.append(f"- Root Mean Squared Error: {rmse:.4f}")
                                    report_lines.append(f"- R² Score: {r2:.4f}")
                                    report_lines.append(f"- Mean Absolute Error: {mae:.4f}")

                                model_report = '\n'.join(report_lines)
                                st.download_button(
                                    label="Download Model Report",
                                    data=model_report.encode('utf-8'),
                                    file_name=f'model_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                                    mime='text/plain'
                                )

                            except Exception as e:
                                st.error(f"❌ Error training model: {str(e)}")
                                st.info("💡 Try selecting a different model or check your data preprocessing.")

                except Exception as e:
                    st.error(f"❌ Error in data preparation: {str(e)}")
                    st.info("💡 Please ensure your data is properly cleaned and prepared in the previous tabs.")
