
🚗 AutoCare AI - AI-Powered Vehicle Health Monitor

AutoCare AI is a web application built with Streamlit and Python that leverages a machine learning model to predict potential vehicle failures. It provides a user-friendly interface for vehicle owners and fleet managers to monitor car vitals, receive rule-based alerts, and get predictive maintenance recommendations before a critical issue occurs.

The application features a complete user authentication system backed by a PostgreSQL database, allowing users to manage personal profiles and register multiple vehicles.

✨ Features

User Authentication: Secure sign-up and login system powered by a PostgreSQL database.

User & Car Management: Users can create profiles, add multiple cars, and manage their vehicle details (make, model, year, odometer).

AI-Powered Diagnostics: Predicts potential failures (e.g., engine, battery issues) using a pre-trained Scikit-Learn model.

Rule-Based Alerts: A hybrid system that provides instant warnings for critical thresholds like engine overheating, low battery voltage, and worn brake pads.

Interactive Dashboard: An intuitive dashboard to select a vehicle and input its current vital signs for diagnosis.

Detailed Analysis: Presents the diagnosis results with model confidence scores and failure probability charts.

Live Fleet Monitoring: A page that fetches all registered cars from the database and displays their simulated "live" health status in a grid of animated cards.

Persistent Contact Form: A "Contact Us" page where submissions are permanently saved to the database.

PDF Report Generation: Users can download a detailed PDF summary of their vehicle's health report.

🛠️ Tech Stack

Frontend: Streamlit

Backend & ML: Python, Pandas, NumPy, Scikit-Learn

Database: PostgreSQL (with psycopg2-binary library)

Data Visualization: Plotly

PDF Generation: ReportLab

⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

Prerequisites
Python 3.8 or higher

PostgreSQL database installed and running

1. Clone the Repository
Bash

git clone https://github.com/your-username/AutoCare-AI.git
cd AutoCare-AI

# Install the required libraries
pip install streamlit pandas numpy plotly psycopg2-binary scikit-learn reportlab

2. Set Up the PostgreSQL Database
Open your PostgreSQL interface (like psql or pgAdmin).

Create a new database for this project.

run this SQL in query tool -
-- Maintenance Records Table
CREATE TABLE maintenance_records (
    id SERIAL PRIMARY KEY,
    car_id INTEGER REFERENCES cars(id) ON DELETE CASCADE,
    issue VARCHAR(255) NOT NULL,
    diagnosis TEXT,
    cost DECIMAL,
    date DATE DEFAULT CURRENT_DATE
);

-- AI Predictions Table
CREATE TABLE ai_predictions (
    id SERIAL PRIMARY KEY,
    car_id INTEGER REFERENCES cars(id) ON DELETE CASCADE,
    prediction TEXT,
    confidence DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

Ensure you have a user and password with privileges to access this database.

3. Configure the Application

Open the main application file (e.g., app.py) in a code editor and update the database connection details with your credentials:

Python

# Find this section in the code
conn = psycopg2.connect(
    dbname="car",
    user="your_postgres_user",      # Change this
    password="your_postgres_password", # Change this
    host="localhost",
    port="5432"
)
4. Add Machine Learning Model Files

This project requires four pre-trained model files. Make sure these files are present in the root directory of your project:

model.pkl

scaler.pkl

encoder.pkl

training_columns.pkl

5. Run the Streamlit App
Once the setup is complete, run the following command in your terminal:

streamlit run app_main.py

The application should now be running and accessible in your web browser.

📋 File Structure
.
├── model trining.py        # Main Streamlit model file
├── app_main.py             # Main Streamlit application file
├── model.pkl               # Trained classification model
├── scaler.pkl              # Feature scaler for the model
├── encoder.pkl             # Label encoder for the model's target variable
├── requirement.txt         # reqired instalations
├── training_columns.pkl    # List of columns used during model training
├── upgraded_car_data.csv   # it contain data al all cars
└── README.md               # This file contains overviews of this project and step by step installation


🚀 How to Use

Sign Up: Create a new user account.

Log In: Log in with your new credentials.

Go to Profile: Navigate to the "Profile" page from the sidebar to add your car(s).

Go to Dashboard: Select a car and input its current vital signs.

Run Diagnosis: Click the "Run Diagnosis" button to see the results on the "Detailed Analysis" page.

Monitor Fleet: Visit the "Live Fleet Monitoring" page to see the status of all cars in the system.

