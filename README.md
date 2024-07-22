# VisaVision.com

## Introduction
VisaVision.com is a web-based platform that provides comprehensive analysis and insights into H-1B visa application data from 2021 to 2023. The platform offers features such as employer analysis, visa insights, comprehensive reports, and approval predictions using AI-powered tools. This documentation details the project’s architecture, setup, features, and usage instructions.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Setup](#project-setup)
    1. [Prerequisites](#prerequisites)
    2. [Installation](#installation)
3. [Docker](#docker)
4. [Testing](#testing)
5. [Usage](#usage)
    1. [Running the Application](#running-the-application)
    2. [Accessing Features](#accessing-features)
6. [Features](#features)
    1. [Employers Analysis](#employers-analysis)
    2. [H-1B Visa](#h1b-visa)
    3. [Comprehensive Reports](#comprehensive-reports)
    4. [Approval Predictions](#approval-predictions)
7. [Architecture](#architecture)
    1. [Frontend](#frontend)
    2. [Backend](#backend)
    3. [Database](#database)
8. [Contributing](#contributing)


## Project Setup

### Prerequisites
- Python 3.8+
- MongoDB

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/visavision.com.git
   cd visavision.com

2. Create and activate a virtual environment:
    - On Windows:
        ```sh
        python -m venv venv
        venv\Scripts\activate

    - On macOS/Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate

3. Install dependencies using pip:
   ```sh
   pip install -r requirements.txt

4. Set up the environment variables:
    - Copy the .env.example file and rename it to .env.
    - Open the .env file and set the SECRET_KEY and MONGO_URI variable to your MongoDB connection URI.
    ```sh
    SECRET_KEY=your_secret_key
    MONGO_URI=mongodb://localhost:27017/your_db_name


## Docker
To run the VisaVision.com web application using Docker, follow these steps:
1. Build the Docker image:
   ```sh
   cd <project directory>
   docker-compose build

2. Run the Docker container:
   ```sh
   cd <project directory>
   docker-compose up


## Testing
To run test scripts for this project, execute:
  ```sh
  pytest
  ```


## Usage

### Running the Application

1. Run the development server:
   ```sh
   python run.py

2. Open your browser and navigate to http://127.0.0.1:5000/ to access the application.

### Accessing Features
- Employers Analysis: Navigate to the "Employers" tab to view detailed insights into top employers and their    application success rates.
- H-1B Visa Insights: Go to the "H-1B Visa" tab to get an overview of the H-1B visa program.
- Comprehensive Reports: Navigate to the "Reports" tab to Access in-depth reports of H-1B Visa from 2021 - 2023.
- Approval Predictions: Use the "Predictions" tab to predict the likelihood of visa approval based on various factors using our AI-powered tool.


## Features

### Employers Analysis
Detailed analysis of employers who sponsor H-1B visas, view detailed insights into top employers and their    application success rates.

### H-1B Visa
Comprehensive overview of the H-1B visa program

### Comprehensive Reports
In-depth reports providing detailed information on various aspects of the H-1B visa program, including application trends and approval rates.

### Approval Predictions
AI-driven predictions for H-1B visa approvals based on various factors such as employer and location.


## Architecture

### Frontend
The frontend is built using HTML, CSS, and JavaScript. It includes the following components:
- **HTML**: Used for structuring the web pages.
- **CSS**: Used for styling the web pages.
- **JavaScript**: Used for adding interactivity to the web pages.
- **D3.js**: Used for creating dynamic and interactive data visualizations.

### Backend
The backend is implemented using Flask, a lightweight web framework for Python. It handles the following tasks:
- **Routing**: Manages the URL routing for the web application.
- **API**: Provides endpoints for frontend interactions.
- **Business Logic**: Processes data and implements the core functionality of the application.

### Database
The database used for storing and managing the application's data is MongoDB. It offers the following features:
- **Document-Oriented Storage**: Stores data in JSON-like documents for flexibility.
- **Scalability**: Easily scales with the growth of the application.
- **Query Language**: Provides a rich query language for efficient data retrieval.


## Contributing
We welcome contributions. To contribute, please follow these steps:
- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -m 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Open a pull request.

