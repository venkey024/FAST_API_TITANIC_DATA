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
            background-color: #F5F9FF;
            color: #1A237E;
        }
        [data-testid="stSidebar"] {
            background-color: #EFF6FF;
            color: #1A237E;
        }
        .title-container {
            background: linear-gradient(135deg, #48CAE4 0%, #023E8A 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .sidebar .element-container h1 {
            color: #1E88E5;
            font-size: 1.8rem;
            font-weight: 700;
        }
        .sidebar .element-container h2 {
            color: #2962FF;
            font-size: 1.4rem;
            font-weight: 600;
        }
        .sidebar .element-container h3 {
            color: #0D47A1;
            font-size: 1.2rem;
        }
        .metric-container {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #1A237E;
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
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            color: #1A237E;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #F5F9FF;
            color: #1A237E;
        }
        .element-container {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            color: #1A237E;
        }
        .stApp {
            background: linear-gradient(135deg, #F5F9FF 0%, #EFF6FF 100%);
            color: #1A237E;
        }
        /* Improve text contrast */
        .main, .stApp {
            color: #1A237E !important;
        }
        /* Ensure text is visible in all containers */
        .element-container, .stats-box, .metric-container {
            color: #1A237E !important;
        }
        /* Style for form labels and text inputs */
        .stTextInput label, .stTextInput input,
        .stNumberInput label, .stNumberInput input,
        .stSelectbox label, .stSelectbox > div[data-baseweb="select"] > div {
            color: #1A237E !important;
        }
        /* Ensure plot titles and labels are visible */
        .js-plotly-plot .plotly .gtitle,
        .js-plotly-plot .plotly .xtitle,
        .js-plotly-plot .plotly .ytitle {
            color: #1A237E !important;
        }
        /* Additional styles for better text visibility */
        .stMarkdown strong, .stMarkdown em {
            color: #1A237E !important;
        }
        /* Table text visibility */
        .dataframe th {
            color: #1A237E !important;
            background-color: #E3F2FD !important;
        }
        .dataframe td {
            color: #1A237E !important;
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
def get_feature_importance(_model, df):
    # Get feature names
    feature_names = ['Passenger Class', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
    
    # Get feature importance scores
    importance_scores = model.feature_importances_
    
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
            importance_df = get_feature_importance(model, df)            # Create a horizontal bar chart for feature importance
            fig = px.bar(importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title='Feature Importance in Survival Prediction',
                        color='Importance',
                        color_continuous_scale=['#E3F2FD', '#90CAF9', '#2196F3'],
                        template='plotly_white')
            
            fig.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False
            )
            fig.update_traces(marker_line_color='rgb(8,48,107)',
                            marker_line_width=1.5,
                            opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis of passenger class
            st.subheader("🎫 Passenger Class Analysis")
            
            # Calculate survival rate by class
            class_survival = df.groupby('Pclass')['Survived'].agg(['mean', 'count']).reset_index()
            class_survival['mean'] = class_survival['mean'] * 100
            class_survival['Pclass'] = class_survival['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})            # Create a bar chart with survival rates
            fig = px.bar(class_survival, 
                x='Pclass', 
                y='mean',
                text=class_survival['mean'].round(1).astype(str) + '%',
                title='Survival Rate by Passenger Class',
                labels={'mean': 'Survival Rate (%)', 'Pclass': 'Passenger Class'},
                color='mean',
                color_continuous_scale=['#E1F5FE', '#81D4FA', '#03A9F4'],
                template='plotly_white')
            
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add text explanation
            st.markdown("""
            #### Key Findings:
            - 1st Class passengers had significantly higher survival rates
            - Class was one of the most important factors in survival
            - Lower class passengers faced more challenging survival odds
            """)
        
            # After Passenger Class Analysis, add Parents/Children Analysis
            st.subheader("👨‍👩‍👧‍👦 Family Size Impact")
            
            # Calculate survival rate by parch
            parch_analysis = df.groupby('Parch')['Survived'].agg(['mean', 'count']).reset_index()
            parch_analysis['mean'] = parch_analysis['mean'] * 100
              # Create visualization for survival rate by parch
            fig = px.bar(parch_analysis, 
                x='Parch', 
                y='mean',
                text=parch_analysis['mean'].round(1).astype(str) + '%',
                title='Survival Rate by Number of Parents/Children',
                labels={'Parch': 'Number of Parents/Children', 'mean': 'Survival Rate (%)'},
                color='mean',
                color_continuous_scale=['#E0F7FA', '#80DEEA', '#00BCD4'])
            
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add key insights about family size
            st.markdown("""
            #### Family Size Impact:
            - 👶 Having 1-3 parents/children increased survival chances
            - 👨‍👩‍👧‍👦 Optimal family size: 2-3 members
            - ⚠️ Very large families (4+) had lower survival rates
            - 🔍 Possible reasons:
              * Families with children got priority
              * Better group coordination
              * Mutual support during crisis
            """)
              # Show the optimal family size for survival
            with st.container():
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                best_parch = parch_analysis.loc[parch_analysis['mean'].idxmax()]
                st.metric("✨ Optimal Family Size", 
                         f"{best_parch['Parch']} parents/children",
                         f"Survival Rate: {best_parch['mean']:.1f}%")

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
                                 barmode="overlay",
                 color_discrete_map={0: '#FFB6C1', 1: '#87CEEB'},
                                 template='plotly_white',
                                 opacity=0.8)
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    legend_title_text="Survival Status",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(255, 255, 255, 0.8)'
                    ),
                    xaxis=dict(gridcolor='lightgray'),
                    yaxis=dict(gridcolor='lightgray')
                )
                
                # Update legend labels
                fig.data[0].name = 'Did Not Survive'
                fig.data[1].name = 'Survived'
                st.plotly_chart(fig, use_container_width=True)
                  # Survival rate by class
                survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
                fig = px.bar(survival_by_class, 
                           title="Survival Rate by Passenger Class",
                           labels={"value": "Survival Rate (%)", "Pclass": "Passenger Class"},
                           color=survival_by_class.values,                           color_continuous_scale=['#E0F4FF', '#87CEEB', '#4682B4'],
                           template='plotly_white')
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(gridcolor='lightgray'),
                    yaxis=dict(gridcolor='lightgray')
                )
                
                fig.update_traces(
                    marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5,
                    opacity=0.8,
                    text=survival_by_class.round(1).astype(str) + '%',
                    textposition='outside'
                )
                st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Model could not be loaded. Please check if the model file exists.")
