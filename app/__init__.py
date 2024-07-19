# Standard library imports
import os

# Related third party imports
from flask import Flask
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Create an instance of the Flask class
app = Flask(__name__)
# Set the secret key for the Flask app from the environment variable
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")

# Import the routes from the app package
from app import routes
