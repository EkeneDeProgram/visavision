# Standard library imports
import os

# Related third party imports
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define a base configuration class
class Config:
    # Secret key for the Flask app, loaded from environment variables
    SECRET_KEY = os.getenv("SECRET_KEY")
    # MongoDB URI for database connection, loaded from environment variables
    MONGO_URI = os.getenv("MONGO_URI")

