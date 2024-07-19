# Standard library imports
import logging
import os
import time

# Related third party imports
from flask import (
    render_template, jsonify,
    redirect, url_for,
    flash, abort, 
    current_app, request
)
from pymongo import MongoClient
import pandas as pd
import joblib

# Local application/library specific imports
from . import app  # Import the app instance from the current package
from .config import Config
from .utils import (
    load_data, analyze_data,
    clean_data, visa_approval_model, 
    get_employers, get_employer_details
)

# Initialize the MongoDB client using the URI specified 
# in the configuration
client = MongoClient(Config.MONGO_URI)
# Access the 'visa' database within the MongoDB instance
db = client.visa 

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load and process data
start_time = time.time()
logging.info("Loading data...")
data = load_data("data/h-1b-data-2021-2023.csv")
logging.info(f"Data loaded in {time.time() - start_time} seconds")

# Clean data
start_time = time.time()
logging.info("Cleaning data...")
cleaned_data = clean_data(data)
logging.info(f"Data cleaned in {time.time() - start_time} seconds")

# Analyse data
start_time = time.time()
logging.info("Analyzing data...")
analyzed_data = analyze_data(cleaned_data)
logging.info(f"Data analyzed in {time.time() - start_time} seconds")

# Assign the cleaned_data DataFrame to the variable data_sample
data_sample = cleaned_data

# Create a directory named "models" if it does not already exist
os.makedirs("models", exist_ok=True)

# Check for cached model
model_path_one = "models/visa_approval_model.joblib"

try:
    # Load cached model
    logging.info("Loading cached models...")
    approval_model = joblib.load(model_path_one)
    logging.info("Models loaded from cache.")

except:
    # Initialize models and functions by calling visa_approval_model on the initial data
    start_time = time.time()
    logging.info("Initializing models...")
    approval_model = visa_approval_model(data_sample)
    logging.info(f"Models initialized in {time.time} seconds")
    joblib.dump(approval_model, model_path_one)
    logging.info("Models saved to cache.")


@app.route("/")
def index():
    try:
        # Attempt to render the index.html template
        return render_template("index.html")
    
    except Exception as e:
        # If an exception occurs during rendering, log the error 
        # using Flask's current_app logger
        current_app.logger.error(f"Error rendering index.html: {str(e)}")
        abort(500)  # Abort with Internal Server Error if rendering fails


@app.route("/h1b_visa")
def h1b_visa():
    try:
        # Attempt to render the h1b_visa.html template 
        return render_template("h1b_visa.html")

    except Exception as e:
        # Logging the error if template rendering fails
        current_app.logger.error(f"Error rendering h1b_visa.html: {str(e)}") 
        abort(500)  # Abort with Internal Server Error if rendering fails


@app.route("/employers")
def employers():
    try:
        # Attempt to retrieve employers list using cleaned data
        employers_list = get_employers(cleaned_data)  
        # Render employers.html template with employers list
        return render_template("employers.html", employers=employers_list)
    
    except Exception as e:
         # Log error if exception occurs
        current_app.logger.error(f"Error in /employers route: {str(e)}")
        # Abort with Internal Server Error if rendering or data retrieval fails
        abort(500)


@app.route("/employer/<employer_name>")
def employer(employer_name):
    try:
        # Attempt to retrieve details for the specified employer
        details = get_employer_details(cleaned_data, employer_name)
        # Render the 'employer_details.html' template with retrieved details
        return render_template(
            "employer_details.html", 
            details=details, 
            employer_name=employer_name
        )
    
    except ValueError as e:
        # Catch specific ValueError related to data retrieval or validation
        flash(f"Error: {str(e)}", "error")  # Flash an error message to the user
        app.logger.error(f"Error retrieving details for {employer_name}: {str(e)}") # Log the specific error
        return redirect(url_for("employers"))  # Redirect to the 'employers' route on error
    
    except Exception as e:
        # Catch any unexpected exceptions
        flash("An error occurred. Please try again later.", "error")  # Flash a generic error message
        app.logger.error(f"Unexpected error in employer route: {str(e)}")  # Log the unexpected error
        return redirect(url_for("employers"))  # Redirect to the 'employers' route on unexpected error


@app.route("/api/analysis", methods=["GET"])
def analysis():
    try:
        # Attempt to fetch analyzed data 
        data = analyzed_data
        # Return JSON response with the fetched data
        return jsonify(data)
    
    except Exception as e:
        # Log the error for debugging purposes
        current_app.logger.error(f"Error in /api/analysis endpoint: {str(e)}")
        # Abort with a 500 Internal Server Error if an exception occurs
        abort(500)


@app.route("/reports")
def reports():
    try:
        # Assign the analyzed data to the summary_data variable for further use
        summary_data = analyzed_data
        # Return rendered template with summary data
        return render_template("reports.html", summary=summary_data)
    
    except Exception as e:
        # Log the error for debugging purposes
        current_app.logger.error(f"Error in /reports endpoint: {str(e)}")
        # Abort with a 500 Internal Server Error if an exception occurs
        abort(500)


@app.route("/visa_approval_predictions")
def predictions():
    try:
        # Render the predictions.html template
        return render_template("predictions.html")
    
    except Exception as e:
        # If an exception occurs during rendering, log the error
        current_app.logger.error(f"Error rendering predictions.html: {str(e)}")
        abort(500)  # Abort with Internal Server Error if rendering fails


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Attempt to retrieve the input data from the JSON request body
        input_data = request.json
        # Load the pre-trained model from the joblib file
        with open("models/visa_approval_model.joblib", "rb") as f:
            model_dict = joblib.load(f)

        # Extract the scaler object from the loaded model dictionary
        scaler = model_dict["scaler"]
        # Extract the PCA object from the loaded model dictionary
        pca = model_dict["pca"]
        # Extract the approval model object from the loaded model dictionary
        approval_model = model_dict["approval_model"]
        # Extract the feature names from the loaded model dictionary
        feature_names = model_dict["feature_names"]

        # Create a DataFrame from the input data
        df = pd.DataFrame([input_data])
        # Convert categorical variables to dummy/indicator 
        # variables, dropping the first category to avoid multicollinearity
        df = pd.get_dummies(df, drop_first=True)

        # Identify any missing columns in the input data 
        # that are expected by the model
        missing_cols = set(feature_names) - set(df.columns)
        # If there are missing columns, create a DataFrame 
        # with those columns filled with zeros
        if missing_cols:
            missing_df = pd.DataFrame(0, index=df.index, columns=list(missing_cols))
            # Concatenate the input DataFrame with the DataFrame of missing columns
            df = pd.concat([df, missing_df], axis=1)
        
        # Ensure the DataFrame is in the same order as the 
        # feature names expected by the model
        df = df[feature_names]
        # Apply PCA transformation to the DataFrame
        transformed_data = pca.transform(df)
        # Standardize the transformed data using the pre-trained scaler
        standardized_data = scaler.transform(transformed_data)
        # Make a prediction using the pre-trained approval model
        prediction = approval_model.predict(standardized_data)
        # Interpret the prediction result
        result = "Visa Approved" if int(prediction[0]) == 1 else "Visa Not Approved"

        # Add the prediction result to the input data
        input_data["approval_prediction"] = result
        # Save the input data with prediction result to MongoDB
        db.predictions.insert_one(input_data)
        # Return the prediction result as a JSON response with a 200 OK status code
        return jsonify({"approval_prediction": result}), 200
    
    except Exception as e:
        # Return an error message as a JSON response 
        # with a 500 Internal Server Error status code
        return jsonify({"error": str(e)}), 500

