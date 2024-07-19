# Standard library imports.
import logging
from typing import Union

# Related third party imports.
import pandas as pd
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    make_scorer, 
    precision_score, 
    recall_score
)


def load_data(file_path: Union[str, bytes], chunksize: int = 10000) -> pd.DataFrame:
    """
    Load data from a CSV file in chunks and concatenate 
    them into a single DataFrame.
    
    Args:
        file_path (Union[str, bytes]): The path to the CSV file.
        chunksize (int, optional): The number of rows per chunk. 
        Default is 10000.
    
    Returns:
        pd.DataFrame: The concatenated DataFrame containing all data.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If there is a parsing error in the file.
    """
    # Log the beginning of the data reading process
    logging.info("Reading CSV file in chunks...")

    try:
        # Initialize an empty list to collect chunks of data
        chunks = []

        # Read the CSV file in chunks, specified by chunksize
        for chunk in pd.read_csv(file_path, dtype={"Fiscal Year": str}, chunksize=chunksize):
            # Append each chunk to the chunks list
            chunks.append(chunk)
            # Log the shape of the current chunk
            logging.info(f"Read chunk with shape: {chunk.shape}")

        # Concatenate all chunks into a single DataFrame
        data = pd.concat(chunks, ignore_index=True)
        # Log the shape of the concatenated DataFrame
        logging.info(f"Concatenated DataFrame shape: {data.shape}")
        # Return the concatenated DataFrame
        return data

    except FileNotFoundError as e:
        # Log an error message if the file is not found
        logging.error(f"File not found: {file_path}")
        # Raise the FileNotFoundError to be handled by the caller
        raise e
    except pd.errors.EmptyDataError as e:
        # Log an error message if the file is empty
        logging.error("No data: The file is empty.")
        # Raise the EmptyDataError to be handled by the caller
        raise e
    except pd.errors.ParserError as e:
        # Log an error message if there is a parsing error in the file
        logging.error("Parsing error: There is an error in the file.")
        # Raise the ParserError to be handled by the caller
        raise e
    except Exception as e:
        # Log any other unexpected errors
        logging.error(f"An unexpected error occurred: {str(e)}")
        # Raise the generic exception to be handled by the caller
        raise e
    

def clean_data(data):
    """
    Clean the input DataFrame `data`.

    Args:
        data (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.

    Raises:
        ValueError: If there are issues with data types or unexpected 
        conditions.
    """
    # Define a list of required column names
    required_columns = [
        "Tax ID", "Employer", 
        "State", "City", 
        "ZIP", "Fiscal Year", 
        "Initial Approval", 
        "Initial Denial", 
        "Continuing Approval", "Continuing Denial"
    ]

    # Create a list of missing columns by checking which required columns
    # are not present in the DataFrame's columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    # If there are any missing columns, raise a ValueError with a message
    # listing the missing columns
    if missing_columns:
        raise ValueError(f"Missing expected columns: {', '.join(missing_columns)}")

    try:
        logging.info("Handling missing values...")
        # Handle missing Tax ID values
        if data["Tax ID"].isnull().mean() < 0.05:
            data.dropna(subset=["Tax ID"], inplace=True)
        else:
            data["Tax ID"].fillna(0, inplace=True)

        # Drop rows with missing State, City, and ZIP
        data.dropna(subset=["State", "City", "ZIP"], inplace=True)
        # Drop duplicate rows
        data.drop_duplicates(inplace=True)
        # Convert ZIP code to string
        data["ZIP"] = data["ZIP"].astype(str)
        # Convert Fiscal Year to numeric with error handling
        data["Fiscal Year"] = pd.to_numeric(data["Fiscal Year"], errors="coerce")
        # Filter out rows with negative Initial Approval or Initial Denial
        data = data[(data["Initial Approval"] >= 0) & (data["Initial Denial"] >= 0)]
        # Filter out invalid Fiscal Years
        valid_fiscal_years = [2021, 2022, 2023]
        data = data[data["Fiscal Year"].isin(valid_fiscal_years)]
        # Standardize column names
        data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
        return data

    except KeyError as e:
        logging.error(f"KeyError occurred: {str(e)}")
        raise ValueError("Missing expected columns in the input data.") from e
    

def analyze_data(data):
    """
    Analyze the given DataFrame `data` to generate a summary 
    of key metrics and insights.

    Args:
        data (pd.DataFrame): The input DataFrame containing visa 
        application data.

    Returns:
        dict: A dictionary summarizing key metrics and insights.
    """

    def get_highest_counts(df, column, years):
        """
        Helper function to get the highest count and its value
        for a specific column in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            column (str): The column name to analyze.
            years (list): List of years to filter data.

        Returns:
            tuple: A tuple containing the highest count value 
            and its count.
        """
        # Filter the DataFrame to include only rows where 'fiscal_year'
        # is in the specified list of years
        filtered_df = df[df["fiscal_year"].isin(years)]
        # Get the counts of unique values in the specified column of 
        # the filtered DataFrame
        counts = filtered_df[column].value_counts()
        # Check if the counts Series is empty (i.e., no data for the specified years)
        if counts.empty:
            # If there are no counts, return None for both highest value and its count
            return None, None
        # Get the value with the highest count
        highest = counts.idxmax()
        # Return the value with the highest count and the count of that value
        return highest, counts[highest]

    summary = {}

    # Highest counts by overall application
    if "fiscal_year" in data.columns:
        summary["Highest Fiscal Year Application"] = data["fiscal_year"].value_counts().idxmax()
    if "state" in data.columns:
        summary["Highest State Application"] = data["state"].value_counts().idxmax()
    if "city" in data.columns:
        summary["Highest City Application"] = data["city"].value_counts().idxmax()
    if "employer" in data.columns:
        summary["Highest Employer Application"] = data["employer"].value_counts().idxmax()

    # Highest counts by initial approval
    # Check if the "initial_approval" column exists in the DataFrame
    if "initial_approval" in data.columns:
         # Calculate the employer with the highest sum of initial approvals 
         # and store it in the summary dictionary
        summary["Highest Employer Initial Approval"] = (
            data.groupby("employer")["initial_approval"]
            .sum().
            idxmax()
        )
        
        # Calculate the fiscal year with the highest sum of initial approvals 
        # and store it in the summary dictionary
        summary["Highest Fiscal Year Initial Approval"] = (
            data.groupby("fiscal_year")["initial_approval"]
            .sum()
            .idxmax()
        )
        
        # Calculate the state with the highest sum of initial approvals 
        # and store it in the summary dictionary
        summary["Highest State Initial Approval"] = (
            data.groupby("state")["initial_approval"]
            .sum()
            .idxmax()
        )
        
        # Calculate the city with the highest sum of initial approvals 
        # and store it in the summary dictionary
        summary["Highest City Initial Approval"] = (
            data.groupby("city")["initial_approval"]
            .sum()
            .idxmax()
        )

    # Highest counts by initial denial
    # Check if the "initial_denial" column exists in the DataFrame
    if "initial_denial" in data.columns:
        # Calculate the employer with the highest sum of initial denials 
        # and store it in the summary dictionary
        summary["Highest Employer Initial Denial"] = (
            data.groupby("employer")["initial_denial"]
            .sum()
            .idxmax()
        )
        
        # Calculate the fiscal year with the highest sum of initial denials 
        # and store it in the summary dictionary
        summary["Highest Fiscal Year Initial Denial"] = (
            data.groupby("fiscal_year")["initial_denial"]
            .sum()
            .idxmax()
        )
        
        # Calculate the state with the highest sum of initial denials 
        # and store it in the summary dictionary
        summary["Highest State Initial Denial"] = (
            data.groupby("state")["initial_denial"]
            .sum()
            .idxmax()
        )
        
        # Calculate the city with the highest sum of initial denials 
        # and store it in the summary dictionary
        summary["Highest City Initial Denial"] = (
            data.groupby("city")["initial_denial"]
            .sum()
            .idxmax()
        )

    # Top 20 cities and employers by application and approval
    # Check if the "city" column exists in the DataFrame
    if "city" in data.columns:
        # Get the top 20 cities by the number of applications and store it in a dictionary
        top_cities_application = data["city"].value_counts().nlargest(20).to_dict()
        # Add the dictionary of top 20 cities by applications to the summary dictionary
        summary["Top 20 Cities by Application"] = top_cities_application

        # Get the top 20 cities by the number of initial approvals
        top_cities_approval = (
            data[data["initial_approval"] == 1]["city"]  # Filter the data for rows with initial approval
            .value_counts()  # Count occurrences of each city in the filtered data
            .nlargest(20)  # Get the top 20 cities by count
            .to_dict()  # Convert the result to a dictionary
        )
        # Add the dictionary of top 20 cities by initial approvals to the summary dictionary
        summary["Top 20 Cities by Approval"] = top_cities_approval

    # Check if the "employer" column exists in the DataFrame
    if "employer" in data.columns:
        # Get the top 20 employers by the number of applications 
        # and store it in a dictionary
        top_employers_application = data["employer"].value_counts().nlargest(20).to_dict()
        # Add the dictionary of top 20 employers by applications to the summary dictionary
        summary["Top 20 Employers by Application"] = top_employers_application

        # Get the top 20 employers by the number of initial approvals
        top_employers_approval = (
            data[data["initial_approval"] == 1]["employer"]  # Filter the data for rows with initial approval
            .value_counts()  # Count occurrences of each employer in the filtered data
            .nlargest(20)  # Get the top 20 employers by count
            .to_dict()  # Convert the result to a dictionary
        )
         # Add the dictionary of top 20 employers by initial approvals 
         # to the summary dictionary
        summary["Top 20 Employers by Approval"] = top_employers_approval

    # Highest counts by application and approval for specific years
    # Define a list of years to analyze
    years = [2021, 2022, 2023]

    # Loop through each year in the list
    for year in years:
        # Check if the "employer" column exists in the DataFrame
        if "employer" in data.columns:
            # Get the employer with the highest count of applications for the given year
            highest_employer_app, _ = get_highest_counts(data, "employer", [year])
            # Store the highest employer application for the year in the summary dictionary
            # If no data is available, store 'No data available' instead
            summary[f"Highest Employer Application {year}"] = (
                highest_employer_app if highest_employer_app else "No data available"
            )

            # Get the employer with the highest count of initial approvals for the given year
            highest_employer_approval, _ = (
                get_highest_counts(data[data["initial_approval"] == 1], "employer", [year])
            )

            # Store the highest employer initial approval for the year in the summary dictionary
            # If no data is available, store 'No data available' instead
            summary[f"Highest Employer Initial Approval {year}"] = (
                highest_employer_approval if highest_employer_approval else "No data available"
            )

        # Check if the "state" column exists in the DataFrame
        if "state" in data.columns:
            # Get the state with the highest count of applications for the given year
            highest_state_app, _ = get_highest_counts(data, "state", [year])

            # Store the highest state application for the year in the summary dictionary
            # If no data is available, store 'No data available' instead
            summary[f"Highest State Application {year}"] = (
                highest_state_app if highest_state_app else "No data available"
            )

             # Get the state with the highest count of initial approvals for the given year
            highest_state_approval, _ = (
                get_highest_counts(data[data["initial_approval"] == 1], "state", [year])
            )

            # Store the highest state initial approval for the year in the summary dictionary
            # If no data is available, store 'No data available' instead
            summary[f"Highest State Initial Approval {year}"] = (
                highest_state_approval if highest_state_approval else "No data available"
            )

        # Check if the "city" column exists in the DataFrame
        if "city" in data.columns:
            # Get the city with the highest count of applications for the given year
            highest_city_app, _ = get_highest_counts(data, "city", [year])

            # Store the highest city application for the year in the summary dictionary
            # If no data is available, store 'No data available' instead
            summary[f"Highest City Application {year}"] = (
                highest_city_app if highest_city_app else "No data available"
            )

            # Get the city with the highest count of initial approvals for the given year
            highest_city_approval, _ = (
                get_highest_counts(data[data["initial_approval"] == 1], "city", [year])
            )

            # Store the highest city initial approval for the year in the summary dictionary
            # If no data is available, store 'No data available' instead
            summary[f"Highest City Initial Approval {year}"] = (
                highest_city_approval if highest_city_approval else "No data available"
            )

    # Rates of application and initial approval over fiscal years
    # Check if the "fiscal_year" column exists in the DataFrame
    if "fiscal_year" in data.columns:
        # Calculate the count of applications per fiscal year, sort by 
        # fiscal year, and convert to a dictionary
        summary["Rate of Application"] = data["fiscal_year"].value_counts().sort_index().to_dict()

        # Calculate the count of initial approvals per fiscal year
        # First, filter the data for rows with initial approval
        # Then, count occurrences of each fiscal year in the filtered data
        # Sort by fiscal year and convert the result to a dictionary
        summary["Rate of Initial Approval"] = (
            data[data["initial_approval"] == 1]["fiscal_year"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    # Return the summary dictionary
    return summary


def get_employers(data):
    """
    Extract unique employer names from the given DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing an 
        'employer' column.

    Returns:
        list: A list of lists, where each sublist contains a 
        unique employer name.
    """
    # Select the "employer" column from the DataFrame, drop duplicate values, 
    # and convert the result to a list of lists
    unique_employers = data[["employer"]].drop_duplicates().values.tolist()
    # Return the list of unique employer names
    return unique_employers


def get_employer_details(data, employer_name):
    """
    Retrieve detailed statistics for a specific employer from the 
    given DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing 
        visa application data.
        employer_name (str): The name of the employer for which
        to retrieve details.

    Returns:
        dict: A dictionary containing detailed statistics for the specified 
        employer, including:
            - total_applications (int): Total number of applications.
            - applications_2021 (int): Number of applications in 2021.
            - applications_2022 (int): Number of applications in 2022.
            - applications_2023 (int): Number of applications in 2023.
            - total_initial_approvals (int): Total number of initial approvals.
            - initial_approvals_2021 (int): Number of initial approvals in 2021.
            - initial_approvals_2022 (int): Number of initial approvals in 2022.
            - initial_approvals_2023 (int): Number of initial approvals in 2023.
            - total_initial_denials (int): Total number of initial denials.
            - state_highest_applications (str): State with the highest number of applications.
            - city_highest_applications (str): City with the highest number of applications.
            - state_highest_approvals (str): State with the highest number of initial approvals.
            - city_highest_approvals (str): City with the highest number of initial approvals.

    Raises:
        ValueError: If the data validation checks fail.
    """
    # Filter the DataFrame to include only the rows where the 
    # employer matches the given employer name
    employer_data = data[data["employer"] == employer_name]

    # Aggregate data
    # Count the total number of applications for the employer
    total_applications = employer_data.shape[0]
    # Count the number of applications for the employer in 2021
    applications_2021 = employer_data[employer_data["fiscal_year"] == 2021].shape[0]
    # Count the number of applications for the employer in 2022
    applications_2022 = employer_data[employer_data["fiscal_year"] == 2022].shape[0]
    # Count the number of applications for the employer in 2023
    applications_2023 = employer_data[employer_data["fiscal_year"] == 2023].shape[0]

    # Sum the total number of initial approvals for the employer
    initial_approvals = employer_data["initial_approval"].sum()

    # Sum the number of initial approvals for the employer in 2021
    initial_approvals_2021 = (
        employer_data[employer_data["fiscal_year"] == 2021]["initial_approval"]
        .sum()
    )
    # Sum the number of initial approvals for the employer in 2022
    initial_approvals_2022 = (
        employer_data[employer_data["fiscal_year"] == 2022]["initial_approval"]
        .sum()
    )
    # Sum the number of initial approvals for the employer in 2023
    initial_approvals_2023 = (
        employer_data[employer_data["fiscal_year"] == 2023]["initial_approval"]
        .sum()
    )

    # Sum the total number of initial denials for the employer
    initial_denials = employer_data["initial_denial"].sum()
    
    # Validation check for total initial approvals exceeding total applications
    if initial_approvals > total_applications:
        raise ValueError(
            f"H-1B Visa Details for {employer_name} cannot be retrieved due to insufficient data."
        )

    # Validation check for initial approvals in specific years exceeding applications in those years
    if initial_approvals_2021 > applications_2021:
        raise ValueError(
            f"H-1B Visa Details for {employer_name} cannot be retrieved due to insufficient data."
        )
    if initial_approvals_2022 > applications_2022:
        raise ValueError(
            f"H-1B Visa Details for {employer_name} cannot be retrieved due to insufficient data."
        )
    if initial_approvals_2023 > applications_2023:
        raise ValueError(
            f"H-1B Visa Details for {employer_name} cannot be retrieved due to insufficient data."
        )

    # Handling empty Series for mode
    # Get the mode (most frequent value) of the 'state' column in the employer's data
    state_highest_applications_series = employer_data["state"].mode()
    # If the mode series is not empty, get the first mode value; otherwise, set to 0
    state_highest_applications = (
        state_highest_applications_series[0] if not state_highest_applications_series
        .empty else 0
    )

    # Get the mode (most frequent value) of the 'city' column in the employer's data
    city_highest_applications_series = employer_data["city"].mode()
    # If the mode series is not empty, get the first mode value; otherwise, set to 0
    city_highest_applications = (
        city_highest_applications_series[0] if not city_highest_applications_series
        .empty else 0
    ) 

    # Get the mode (most frequent value) of the 'state' column in the employer's 
    # data where initial approval > 0
    state_highest_approvals_series = (
        employer_data[employer_data["initial_approval"] > 0]["state"].mode()
    )
    # If the mode series is not empty, get the first mode value; otherwise, set to 0
    state_highest_approvals = (
        state_highest_approvals_series[0] if not state_highest_approvals_series
        .empty else 0
    ) 

    # Get the mode (most frequent value) of the 'city' column in the employer's 
    # data where initial approval > 0
    city_highest_approvals_series = (
        employer_data[employer_data["initial_approval"] > 0]["city"].mode()
    )
    # If the mode series is not empty, get the first mode value; otherwise, set to 0
    city_highest_approvals = (
        city_highest_approvals_series[0] if not city_highest_approvals_series
        .empty else 0
    ) 

    # Return a dictionary with the calculated metrics and insights for the employer
    return {
        "total_applications": total_applications,
        "applications_2021": applications_2021,
        "applications_2022": applications_2022,
        "applications_2023": applications_2023,
        "total_initial_approvals": initial_approvals,
        "initial_approvals_2021": initial_approvals_2021,
        "initial_approvals_2022": initial_approvals_2022,
        "initial_approvals_2023": initial_approvals_2023,
        "total_initial_denials": initial_denials,
        "state_highest_applications": state_highest_applications,
        "city_highest_applications": city_highest_applications,
        "state_highest_approvals": state_highest_approvals,
        "city_highest_approvals": city_highest_approvals,
    }


def visa_approval_model(data, n_neighbors=1):
    """
    Trains a model to predict visa approval based on employer, city, state, and fiscal year.
    
    Parameters:
    - data (pd.DataFrame): The input data containing features and target variable.
    - n_neighbors (int): The number of neighbors to use for SMOTE. Default is 1.

    Returns:
    - dict: A dictionary containing the trained model, scaler, PCA transformer, and other details.
    """
    logging.info("Preparing data for modeling...")

    # Create the target variable for approval prediction
    data["approval"] = (
        ((data["initial_approval"] + data["continuing_approval"]) > 0)
        .astype(int)
    )

    # Aggregate data by city and state, summing up the counts of initial 
    # and continuing approvals and denials
    city_agg = data.groupby(["city", "state"]).agg({
        "initial_approval": "sum",  # Sum of initial approvals per city and state
        "initial_denial": "sum",  # Sum of initial denials per city and state
        "continuing_approval": "sum",  # Sum of continuing approvals per city and state
        "continuing_denial": "sum"  # Sum of continuing denials per city and state
    }).reset_index()  # Reset the index to turn groupby keys into columns

    # Calculate the approval rate for each city
    city_agg["city_approval_rate"] = (
        city_agg["initial_approval"] + city_agg["continuing_approval"]  # Total approvals (initial + continuing)
    ) / (
        # Total applications (initial approvals + initial denials 
        # + continuing approvals + continuing denials)
        city_agg["initial_approval"] 
        + city_agg["initial_denial"] 
        + city_agg["continuing_approval"] 
        + city_agg["continuing_denial"]
    )

    # Aggregate data by state, summing up the counts of initial 
    # and continuing approvals and denials
    state_agg = data.groupby(["state"]).agg({
        "initial_approval": "sum",  # Sum of initial approvals per state
        "initial_denial": "sum",  # Sum of initial denials per state
        "continuing_approval": "sum",  # Sum of continuing approvals per state
        "continuing_denial": "sum"  # Sum of continuing denials per state
    }).reset_index()  # Reset the index to turn groupby keys into columns

    # Calculate the approval rate for each state
    state_agg["state_approval_rate"] = (
        state_agg["initial_approval"] + state_agg["continuing_approval"]  # Total approvals (initial + continuing)
    ) / (
        # Total applications (initial approvals + initial denials 
        # + continuing approvals + continuing denials)
        state_agg["initial_approval"] 
        + state_agg["initial_denial"] 
        + state_agg["continuing_approval"] 
        + state_agg["continuing_denial"]
    )

    # Merge the aggregated data with the main dataframe
    data = data.merge(city_agg[["city", "state", "city_approval_rate"]], 
                      on=["city", "state"], 
                      how="left")
        
    data = data.merge(state_agg[["state", "state_approval_rate"]],
                      on="state", 
                      how="left")
        
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data["city"] = label_encoder.fit_transform(data["city"])
    data["state"] = label_encoder.fit_transform(data["state"])

    # Select features and target
    X = data[[
            "city", "state", 
            "fiscal_year", 
            "city_approval_rate", 
            "state_approval_rate"
        ]]
    y = data["approval"]

    # Store feature names before transforming the data
    feature_names = X.columns

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Set up RandomForestClassifier with class weights
    approval_model = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight={0: 20, 1: 1}
    )

    # Perform cross-validation and logging
    logging.info("Performing cross-validation...")
    scoring = {
        "accuracy": "accuracy", 
        "precision": make_scorer(precision_score), 
        "recall": make_scorer(recall_score)
    }
    
    # Perform cross-validation to evaluate the model
    cv_scores = cross_val_score(
        approval_model, X_train, y_train, cv=3, scoring="accuracy"
    )
    logging.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Define a parameter grid for hyperparameter tuning with GridSearchCV
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }

    # Initialize GridSearchCV with the parameter grid, cross-validation, and scoring metric
    grid_search = GridSearchCV(
        approval_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1
    )
    # Fit the GridSearchCV on the training data
    grid_search.fit(X_train, y_train)

    # Retrieve the best model from the grid search
    best_model = grid_search.best_estimator_
    logging.info(f"Best model parameters: {grid_search.best_params_}")
    # Fit the best model on the training data
    best_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    test_score = best_model.score(X_test, y_test)
    logging.info(f"Test set accuracy: {test_score:.4f}")

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)
    logging.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
    logging.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Return a dictionary containing the trained model and various evaluation metrics and data
    return {
        "approval_model": best_model,
        "scaler": scaler,
        "pca": pca,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "city_agg": city_agg,
        "state_agg": state_agg,
        "cross_val_accuracy": cv_scores.mean(),
        "test_accuracy": test_score,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
