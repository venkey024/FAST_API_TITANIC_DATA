from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import os
import numpy as np

# Try importing joblib as alternative
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

app = FastAPI()

# Load the Titanic model
model_loaded = False
titanic_model = None

# Try loading with joblib first (more reliable for scikit-learn models)
if JOBLIB_AVAILABLE:
    try:
        if os.path.exists('titanic_random_forest_model.joblib'):
            titanic_model = joblib.load('titanic_random_forest_model.joblib')
            model_loaded = True
            model_info = f"Model type: {type(titanic_model).__name__} (loaded with joblib)"
    except Exception as e:
        model_info = f"Joblib loading failed: {str(e)}"

# If joblib failed or not available, try pickle
if not model_loaded:
    try:
        with open('titanic_random_forest_model.pkl', 'rb') as file:
            # Try different pickle protocols
            try:
                titanic_model = pickle.load(file)
            except:
                # Reset file pointer and try with encoding
                file.seek(0)
                titanic_model = pickle.load(file, encoding='latin1')
        model_loaded = True
        model_info = f"Model type: {type(titanic_model).__name__} (loaded with pickle)"
    except FileNotFoundError:
        model_info = "Model files not found. Please ensure 'titanic_random_forest_model.pkl' or 'titanic_random_forest_model.joblib' is in the same directory."
    except Exception as e:
        model_info = f"Error loading model: {str(e)}. This might be due to version incompatibility. Try recreating the model with current scikit-learn version."

@app.get("/")
def read_root():
    return {"message": "Hello, World! This is your FastAPI app."}

@app.get("/titanic", response_class=HTMLResponse)
def welcome():
    model_status = "✅ Model Loaded Successfully" if model_loaded else "❌ Model Loading Failed"
    model_color = "#28a745" if model_loaded else "#dc3545"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titanic ML Model - FastAPI</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }}
            .welcome-container {{
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 600px;
                width: 90%;
            }}
            h1 {{
                color: #333;
                margin-bottom: 1rem;
                font-size: 2.5rem;
            }}
            p {{
                color: #666;
                font-size: 1.2rem;
                line-height: 1.6;
                margin-bottom: 1.5rem;
            }}
            .model-status {{
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                margin: 1.5rem 0;
                border-left: 4px solid {model_color};
            }}
            .status-text {{
                color: {model_color};
                font-weight: bold;
                font-size: 1.1rem;
            }}
            .model-info {{
                color: #666;
                font-size: 0.9rem;
                margin-top: 0.5rem;
            }}
            .api-link {{
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 0.8rem 1.5rem;
                text-decoration: none;
                border-radius: 5px;
                transition: background 0.3s ease;
                margin: 0.5rem;
            }}
            .api-link:hover {{
                background: #5a6fd8;
            }}
        </style>
    </head>
    <body>
        <div class="welcome-container">
            <h1>🚢 Titanic Survival Prediction</h1>
            <p>FastAPI application </p>
            
            
            <p>Input passenger details below to predict Titanic survival:</p>
            
            <form action="/prediction" method="post" style="text-align: left; max-width: 400px; margin: 0 auto;">
                <div style="margin-bottom: 1rem;">
                    <label style="display: block; margin-bottom: 0.5rem; font-weight: bold;">Passenger Class:</label>
                    <select name="pclass" required style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;">
                        <option value="1">1st Class</option>
                        <option value="2">2nd Class</option>
                        <option value="3">3rd Class</option>
                    </select>
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <label style="display: block; margin-bottom: 0.5rem; font-weight: bold;">Sex:</label>
                    <select name="sex" required style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;">
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <label style="display: block; margin-bottom: 0.5rem; font-weight: bold;">Age:</label>
                    <input type="number" name="age" min="0" max="100" step="0.1" required 
                           style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;" 
                           placeholder="Enter age">
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <label style="display: block; margin-bottom: 0.5rem; font-weight: bold;">Siblings/Spouses:</label>
                    <input type="number" name="sibsp" min="0" max="10" required 
                           style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;" 
                           placeholder="Number of siblings/spouses">
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <label style="display: block; margin-bottom: 0.5rem; font-weight: bold;">Parents/Children:</label>
                    <input type="number" name="parch" min="0" max="10" required 
                           style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;" 
                           placeholder="Number of parents/children">
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <label style="display: block; margin-bottom: 0.5rem; font-weight: bold;">$ Fare[Total-Amout]:</label>
                    <input type="number" name="fare" min="0" step="0.01" required 
                           style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px;" 
                           placeholder="Ticket fare">
                </div>
                
                <button type="submit" style="width: 100%; background: #28a745; color: white; padding: 0.8rem; border: none; border-radius: 5px; font-size: 1rem; cursor: pointer;">
                    Predict Survival
                </button>
            </form>
            
            <a href="/docs" class="api-link">View API Documentation</a>
            <a href="/" class="api-link">JSON Endpoint</a>
        </div>
    </body>
    </html>
    """
    return html_content

@app.post("/prediction", response_class=HTMLResponse)
def predict_survival(
    pclass: int = Form(...),
    sex: int = Form(...),
    age: float = Form(...),
    sibsp: int = Form(...),
    parch: int = Form(...),
    fare: float = Form(...),
):
    if not model_loaded:
        return """
        <html><body style="font-family: Arial; text-align: center; padding: 2rem;">
        <h2>❌ Model Not Available</h2>
        <p>The Titanic model could not be loaded.</p>
        <a href="/titanic">Go Back</a>
        </body></html>
        """
    
    try:
        # Prepare input data for prediction
        # Order: pclass, sex, age, sibsp, parch, fare (cabin removed)
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
        
        # Make prediction
        prediction = titanic_model.predict(input_data)[0]
        probability = titanic_model.predict_proba(input_data)[0]
        
        survival_status = "SURVIVED ✅" if prediction == 1 else "DID NOT SURVIVE ❌"
        survival_prob = probability[1] * 100  # Probability of survival
        death_prob = probability[0] * 100     # Probability of death
        
        result_color = "#28a745" if prediction == 1 else "#dc3545"
        
        html_response = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Titanic Survival Prediction</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 2rem;
                    min-height: 100vh;
                }}
                .result-container {{
                    background: white;
                    padding: 2rem;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    text-align: center;
                    max-width: 600px;
                    margin: 0 auto;
                }}
                .prediction-result {{
                    background: #f8f9fa;
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1.5rem 0;
                    border-left: 4px solid {result_color};
                }}
                .prediction-text {{
                    color: {result_color};
                    font-weight: bold;
                    font-size: 1.5rem;
                    margin-bottom: 1rem;
                }}
                .input-summary {{
                    background: #e9ecef;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                    text-align: left;
                }}
                .prob-bar {{
                    background: #ddd;
                    border-radius: 10px;
                    overflow: hidden;
                    height: 20px;
                    margin: 0.5rem 0;
                }}
                .prob-fill {{
                    height: 100%;
                    background: {result_color};
                    width: {survival_prob}%;
                    transition: width 0.3s ease;
                }}
                .back-link {{
                    display: inline-block;
                    background: #667eea;
                    color: white;
                    padding: 0.8rem 1.5rem;
                    text-decoration: none;
                    border-radius: 5px;
                    margin-top: 1rem;
                }}
            </style>
        </head>
        <body>
            <div class="result-container">
                <h1>🚢 Titanic Survival Prediction</h1>
                
                <div class="prediction-result">
                    <div class="prediction-text">{survival_status}</div>
                    <p><strong>Survival Probability:</strong> {survival_prob:.1f}%</p>
                    <div class="prob-bar">
                        <div class="prob-fill"></div>
                    </div>
                    <p><strong>Death Probability:</strong> {death_prob:.1f}%</p>
                </div>
                
                <div class="input-summary">
                    <h3>Input Parameters:</h3>
                    <p><strong>Class:</strong> {pclass} | <strong>Sex:</strong> {'Male' if sex == 1 else 'Female'} | <strong>Age:</strong> {age}</p>
                    <p><strong>Siblings/Spouses:</strong> {sibsp} | <strong>Parents/Children:</strong> {parch}</p>
                    <p><strong>Fare:</strong> ${fare:.2f}</p>
                </div>
                
                <a href="/titanic" class="back-link">Make Another Prediction</a>
            </div>
        </body>
        </html>
        """
        return html_response
        
    except Exception as e:
        return f"""
        <html><body style="font-family: Arial; text-align: center; padding: 2rem;">
        <h2>❌ Prediction Error</h2>
        <p>Error making prediction: {str(e)}</p>
        <a href="/titanic">Go Back</a>
        </body></html>
        """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)