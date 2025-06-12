import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .title-container {
            background-color: #1E88E5;
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
        .survival-true {
            background-color: #28a745;
            color: white;
        }
        .survival-false {
            background-color: #dc3545;
            color: white;
        }
        .stats-box {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Title section with custom styling
st.markdown("""
    <div class="title-container">
        <h1>🚢 Titanic Survival Prediction</h1>
        <p style="font-size: 1.2rem;">Enter passenger details to predict survival probability</p>
    </div>
""", unsafe_allow_html=True)

# Load the model and dataset
@st.cache_resource
def load_model():
    try:
        model = joblib.load('titanic_random_forest_model.joblib')
        return model
    except:
        try:
            model = joblib.load('titanic_random_forest_model.pkl')
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('Titanic-Dataset.csv')
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_data
def get_feature_importance(importance_scores):
    # Get feature names
    feature_names = ['Passenger Class', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
    
    # Create a DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    return importance_df

model = load_model()
df = load_dataset()

# Sidebar with dataset statistics
if df is not None:
    with st.sidebar:
        st.header("📊 Dataset Statistics")
        
        # Feature Importance
        if model is not None:
            st.subheader("🎯 Feature Importance")
            # Get feature importance scores from model first
            importance_scores = model.feature_importances_
            importance_df = get_feature_importance(importance_scores)
            
            # Create horizontal bar chart for feature importance
            fig = px.bar(importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title='Feature Importance in Survival Prediction')
            
            fig.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis of passenger class
            st.subheader("🎫 Passenger Class Analysis")
            
            # Calculate survival rate by class
            class_survival = df.groupby('Pclass')['Survived'].agg(['mean', 'count']).reset_index()
            class_survival['mean'] = class_survival['mean'] * 100
            class_survival['Pclass'] = class_survival['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
            
            # Create a bar chart with survival rates
            fig = px.bar(class_survival, 
                        x='Pclass', 
                        y='mean',
                        text=class_survival['mean'].round(1).astype(str) + '%',
                        title='Survival Rate by Passenger Class',
                        labels={'mean': 'Survival Rate (%)', 'Pclass': 'Passenger Class'})
            
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add text explanation
            st.markdown("""
            #### Key Findings:
            - 1st Class passengers had significantly higher survival rates
            - Class was one of the most important factors in survival
            - Lower class passengers faced more challenging survival odds
            """)
        
        # Original statistics
        st.subheader("📈 General Statistics")
        survival_rate = df['Survived'].mean() * 100
        st.metric("Overall Survival Rate", f"{survival_rate:.1f}%")
        
        # Gender distribution
        st.subheader("👥 Gender Distribution")
        gender_fig = px.pie(df, names='Sex', title='Passengers by Gender')
        st.plotly_chart(gender_fig, use_container_width=True)
        
        # Class distribution
        st.subheader("Passenger Class Distribution")
        class_fig = px.bar(df['Pclass'].value_counts(), title='Passengers by Class')
        class_fig.update_layout(xaxis_title="Class", yaxis_title="Count")
        st.plotly_chart(class_fig, use_container_width=True)

# Main content
if model:
    # Create two columns for input and visualization
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("### 📝 Passenger Information")
        # Create the input form
        with st.form("prediction_form"):
            pclass = st.selectbox(
                "Passenger Class",
                options=[1, 2, 3],
                help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class"
            )
            
            sex = st.selectbox(
                "Sex",
                options=["Female", "Male"],
                help="Select passenger's gender"
            )
            sex = 1 if sex == "Male" else 0
            
            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                help="Enter passenger's age"
            )
            
            sibsp = st.number_input(
                "Number of Siblings/Spouses",
                min_value=0,
                max_value=10,
                value=0,
                help="Number of siblings or spouses aboard"
            )
            
            parch = st.number_input(
                "Number of Parents/Children",
                min_value=0,
                max_value=10,
                value=0,
                help="Number of parents or children aboard"
            )
            
            fare = st.number_input(
                "Fare ($)",
                min_value=0.0,
                value=32.0,
                help="Passenger fare"
            )

            predict_button = st.form_submit_button("🔮 Predict Survival")

    with col2:
        if predict_button:
            try:
                # Prepare input data
                input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                survival_prob = probability[1] * 100
                
                # Display prediction result
                st.markdown("### 🎯 Prediction Result")
                
                # Create a gauge chart for survival probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = survival_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Survival Probability"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1E88E5"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                # Prediction result box
                result_html = f"""
                <div class="prediction-box {'survival-true' if prediction == 1 else 'survival-false'}">
                    <h2>{'✅ SURVIVED' if prediction == 1 else '❌ DID NOT SURVIVE'}</h2>
                    <h3>Probability: {survival_prob:.1f}%</h3>
                </div>
                """
                st.markdown(result_html, unsafe_allow_html=True)
                
                # Display input summary in a nice format
                st.markdown("### 📋 Input Summary")
                summary_df = pd.DataFrame({
                    'Feature': ['Class', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare'],
                    'Value': [
                        f"{pclass} Class",
                        'Male' if sex == 1 else 'Female',
                        f"{age:.1f} years",
                        sibsp,
                        parch,
                        f"${fare:.2f}"
                    ]
                })
                st.table(summary_df)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            # Show some visualizations when no prediction is made
            if df is not None:
                st.markdown("### 📈 Survival Analysis")
                
                # Age distribution plot
                fig = px.histogram(df, x="Age", color="Survived", 
                                 title="Age Distribution by Survival",
                                 labels={"Survived": "Survival Status"},
                                 barmode="overlay")
                st.plotly_chart(fig, use_container_width=True)
                
                # Survival rate by class
                survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
                fig = px.bar(survival_by_class, 
                           title="Survival Rate by Passenger Class",
                           labels={"value": "Survival Rate (%)", "Pclass": "Passenger Class"})
                st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Model could not be loaded. Please check if the model file exists.")
