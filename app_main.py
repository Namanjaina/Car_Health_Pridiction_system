# -*- coding: utf-8 -*-
"""
Expanded Streamlit UI Mock:
- Signup / Login with real PostgreSQL DB
- User Profile (personal + car details from DB)
- Dashboard (mock UI, charts, alerts)
- Attractive styling + animations
Note: This is a UI-first mock for prototyping with a real database.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import hashlib
import time
import random
import psycopg2
import re
import pickle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
from datetime import datetime


# ----------------------- Database & Authentication Functions -----------------------
# Connect to PostgreSQL - MAKE SURE TO CONFIGURE YOUR DB CREDENTIALS
try:
    conn = psycopg2.connect(
        dbname="car",
        user="postgres",
        password="your_password_here",
        host="localhost",
        port="5432"
    )
    conn.autocommit = True
    cursor = conn.cursor()
except psycopg2.OperationalError as e:
    st.error(f"❌ Database connection failed. Please ensure PostgreSQL is running and credentials are correct. Error: {e}")
    st.stop()

def init_db():
    """Create tables if they don't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            email TEXT,
            mobile TEXT
        );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cars (
        car_id SERIAL PRIMARY KEY, 
        user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
        make TEXT NOT NULL,
        model TEXT NOT NULL,
        year INTEGER,
        odometer INTEGER
    );
""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS contact_submissions (
        submission_id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        message TEXT NOT NULL,
        submission_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """)


# Call init_db on startup
init_db()

def hash_password(pw):
    """Hashes a password using SHA256."""
    return hashlib.sha256(pw.encode()).hexdigest()

def create_user(username, pw, full, email, mobile):
    """Adds a new user to the database."""
    pw_hash = hash_password(pw)
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, email, mobile) VALUES (%s, %s, %s, %s, %s)",
            (username, pw_hash, full, email, mobile)
        )
        return True, "User created successfully"
    except psycopg2.IntegrityError:
        return False, "Username already exists"

def authenticate(username, pw):
    """Checks user credentials against the database."""
    cursor.execute(
        "SELECT password_hash, full_name FROM users WHERE username = %s",
        (username,)
    )
    result = cursor.fetchone()
    if result and result[0] == hash_password(pw):
        return True, result[1]  # Return True and full_name
    return False, None

def get_user_by_username(username):
    """Fetches a user's full details."""
    cursor.execute("SELECT full_name, email, mobile FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    if result:
        return {"full_name": result[0], "email": result[1], "mobile": result[2]}
    return None

def update_user_profile(username, full_name, email, mobile):
    """Updates a user's personal details."""
    cursor.execute(
        "UPDATE users SET full_name = %s, email = %s, mobile = %s WHERE username = %s",
        (full_name, email, mobile, username)
    )

def add_car(username, make, model, year, odometer):
    """Adds a new car linked to a user."""
    cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
    user_id = cursor.fetchone()[0]
    cursor.execute(
        "INSERT INTO cars (user_id, make, model, year, odometer) VALUES (%s, %s, %s, %s, %s)",
        (user_id, make, model, year, odometer)
    )

def get_cars_for_user(username):
    """Fetches all cars for a given user."""
    cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
    user_id = cursor.fetchone()[0]
    cursor.execute(
        "SELECT car_id, make, model, year, odometer FROM cars WHERE user_id = %s ORDER BY make, model",
        (user_id,)
    )
    cars = cursor.fetchall()
    return [{"car_id": c[0], "make": c[1], "model": c[2], "year": c[3], "odometer": c[4]} for c in cars]

# <<< NEW FUNCTION TO SAVE SUBMISSIONS TO DB >>>
def add_submission(name, email, message):
    """Adds a new contact form submission to the database."""
    try:
        cursor.execute(
            "INSERT INTO contact_submissions (name, email, message) VALUES (%s, %s, %s)",
            (name, email, message)
        )
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

# <<< NEW FUNCTION TO GET ALL SUBMISSIONS FROM DB >>>
def get_all_submissions():
    """Fetches all contact submissions from the database."""
    cursor.execute("SELECT name, email, message, submission_time FROM contact_submissions ORDER BY submission_time DESC")
    submissions = cursor.fetchall()
    df = pd.DataFrame(submissions, columns=['Name', 'Email', 'Message', 'Time'])
    return df

def update_car(car_id, make, model, year, odometer):
    """Updates car details by car_id."""
    cursor.execute(
        "UPDATE cars SET make = %s, model = %s, year = %s, odometer = %s WHERE car_id = %s",
        (make, model, year, odometer, car_id)
    )

def delete_car(car_id):
    """Deletes a car by car_id."""
    cursor.execute("DELETE FROM cars WHERE car_id = %s", (car_id,))

def current_user():
    return st.session_state.get('current_user')

def login_user(username, full_name):
    st.session_state['logged_in'] = True
    st.session_state['current_user'] = username
    st.session_state['full_name'] = full_name
    st.session_state["page"] = "Dashboard"
    st.rerun()

def logout_user():
    st.session_state.pop('current_user', None)
    st.session_state.pop('full_name', None)
    st.session_state["page"] = "Home"
    st.rerun()


# --- Load Pre-trained Model and Supporting Objects ---
@st.cache_resource
def load_model_assets():
    """Loads the saved model, scaler, encoder, and columns from .pkl files."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        with open('training_columns.pkl', 'rb') as f:
            columns = pickle.load(f)
        return model, scaler, encoder, columns
    except FileNotFoundError:
        st.error("Model files not found! Please make sure all .pkl files are present.")
        return None, None, None, None

model, scaler, encoder, training_columns = load_model_assets()


# ----------------------- Basic Config -----------------------
st.set_page_config(page_title="AutoCare AI ", layout="wide", page_icon="🚗")
st.title("AutoCare AI — UI Prototype")
st.markdown("Prototype: Sign up, login, add car details, view a polished dashboard.")

# ----------------------- Inject CSS (theme + animations) -----------------------
st.markdown(
    """
    <style>
    /* Page background with smooth fade */
    .stApp {
        background: linear-gradient(180deg,#0b1221 0%, #071022 100%);
        color: #e6eef8;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        animation: pageFade 0.7s ease-in-out;
    }
    @keyframes pageFade {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0px); }
    }
    /* Card style */
    .card {
        background: linear-gradient(180deg,#0f1724 0%, #0b1221 100%);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.45);
        transition: transform .25s ease, box-shadow .25s ease;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .card:hover {
        transform: translateY(-6px) rotateX(1deg) rotateY(-1deg) scale(1.01);
        box-shadow: 0 14px 35px rgba(0,0,0,0.65);
    }
    /* Glow animation for alert cards */
    @keyframes glowRed {
        0% { box-shadow: 0 0 8px rgba(255,0,0,0.2); }
        50% { box-shadow: 0 0 20px rgba(255,0,0,0.8); }
        100% { box-shadow: 0 0 8px rgba(255,0,0,0.2); }
    }
    .glow-alert {
        border: 1px solid rgba(255,77,77,0.5);
        animation: glowRed 1.5s ease-in-out infinite;
    }
    /* Remove default Streamlit header/footer */
    header, footer {
        visibility: ;
    }
    
    /* *** NEW: Custom styling for the primary 'Get Started' button *** */
    .stButton > button[kind="primary"] {
        background-color: #16a34a;
        color: white;
        padding: 12px 30px;
        border: none;
        border-radius: 8px;
        font-size: 18px;
        cursor: pointer;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #14853d; /* slightly darker green on hover */
        transform: scale(1.03);
    }
    .stButton > button[kind="primary"]:active {
        transform: scale(0.98);
    }
    /* Badges */
    .badge {
        display:inline-block;
        padding:6px 12px;
        border-radius:999px;
        font-weight:700;
        margin-right:8px;
        margin-top:6px;
        transition: transform 0.2s ease;
    }
    .badge:hover { transform: scale(1.05); }
    .badge-critical { background: rgba(255,77,77,0.12); color:#ff6b6b; border:1px solid rgba(255,77,77,0.18); }
    .badge-warning { background: rgba(255,165,0,0.10); color:#ffb84d; border:1px solid rgba(255,165,0,0.14); }
    .badge-normal { background: rgba(0,200,120,0.06); color:#7ee787; border:1px solid rgba(0,200,120,0.08); }
    .mini-foot { color: rgba(230,238,248,0.6); font-size:12px; margin-top:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------- Health Check Logic -----------------------
def check_custom_rules(data):
    """Return a list of rule-based failures triggered by input data."""
    alerts = []
    try:
        if data["odometer_km"] > 300000:
            alerts.append("Maintenance Due / High Mileage Warning")
        if data["engine_temp_c"] > 110:
            alerts.append("Engine Overheating")
        if data["battery_voltage_v"] < 12.0:
            alerts.append("Battery Failure")
        if data["oil_pressure_kpa"] < 150:
            alerts.append("Low Oil Pressure Warning")
        if data["brake_pad_wear_mm_front"] < 3 or data.get("brake_pad_wear_mm_rear", 3) < 3:
            alerts.append("Brake Pads Critically Worn")
        if data["suspension_health_pct"] < 40:
            alerts.append("Suspension Failure Risk")
        if data["tire_pressure_psi_fl"] < 20:
            alerts.append("Low Tire Pressure")
        if data["coolant_level_pct"] < 30:
            alerts.append("Coolant Critically Low")
        if data["brake_fluid_level_pct"] < 20:
            alerts.append("Brake Fluid Critically Low")
        if data["transmission_fluid_temp_c"] > 110:
            alerts.append("Transmission Overheating")
    except Exception:
        # If missing keys, ignore rule checks for missing fields
        pass
    return alerts

def predict_failure(input_data):
    """Takes a dictionary of sensor data and returns prediction details."""
    if not all([model, scaler, encoder, training_columns]):
        return "Model not loaded", 0, None

    # Build dataframe and engineered features
    input_df = pd.DataFrame([input_data])
    # safe feature engineering with try/except
    try:
        input_df["temp_pressure_ratio"] = input_df["engine_temp_c"] / (input_df["oil_pressure_kpa"] + 1e-6)
    except Exception:
        input_df["temp_pressure_ratio"] = 0.0
    try:
        input_df["total_brake_wear"] = input_df["brake_pad_wear_mm_front"] + input_df["brake_pad_wear_mm_rear"]
    except Exception:
        input_df["total_brake_wear"] = 0.0

    # Reindex to training columns safely
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # Scale + predict
    input_scaled = scaler.transform(input_df)
    prediction_encoded = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    predicted_failure = encoder.inverse_transform(prediction_encoded)[0]
    confidence = prediction_proba.max() * 100

    return predicted_failure, confidence, prediction_proba


# ----------------------- Sidebar Navigation Setup -----------------------
st.sidebar.markdown("## Navigation")

# Dynamic sidebar based on login state
if current_user():
    # --- LOGGED-IN VIEW ---
    st.sidebar.write(f"Signed in as **{current_user()}**")
    if st.sidebar.button("Logout"):
        logout_user()
    st.sidebar.markdown("---")

    # Define pages for logged-in users
    available_pages = ["Dashboard", "Profile", "Detailed Analysis", "Live Fleet Monitoring", "Contact Us"]
    
    # Ensure the current page is valid for a logged-in user
    # If not, default to Dashboard
    if st.session_state.get("page") not in available_pages:
        st.session_state.page = "Dashboard"
else:
    # --- LOGGED-OUT VIEW ---
    available_pages = ["Home", "Log In", "Sign Up"]
    
    # Ensure the current page is valid for a logged-out user
    # If not, default to Home
    if st.session_state.get("page") not in available_pages:
        st.session_state.page = "Home"

    if "submissions" not in st.session_state:
        st.session_state.submissions = [] 

# Create the selectbox with the dynamic list of pages
selected_page = st.sidebar.radio(
    "Go to",
    available_pages,
    index=available_pages.index(st.session_state.page)
)

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Prototype UI — Real DB and ML integration.")


# ----------------------- LIVE FLEET MONITORING -----------------------
if st.session_state.page == "Live Fleet Monitoring":
    st.title("📡 Live Fleet Monitoring")
    st.markdown("Monitor the **real-time health** of all vehicles in your fleet.")

    # Add extra CSS for pulsing/flickering animations
    st.markdown(
        """
        <style>
        @keyframes pulseRed {
            0% { color: #ff4d4d; text-shadow: 0 0 2px rgba(255,77,77,0.3); }
            50% { color: #ff0000; text-shadow: 0 0 10px rgba(255,77,77,0.8); }
            100% { color: #ff4d4d; text-shadow: 0 0 2px rgba(255,77,77,0.3); }
        }
        .engine-pulse { animation: pulseRed 1.5s infinite; font-weight: 700; }
        @keyframes flickerOrange {
            0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% { opacity: 1; }
            20%, 22%, 24%, 55% { opacity: 0.3; }
        }
        .battery-flicker { animation: flickerOrange 1.2s infinite; color: orange; font-weight: 700; }
        .badge { display:inline-block; padding:6px 12px; border-radius:999px; font-weight:700; }
        .badge-critical { background: rgba(255,77,77,0.12); color:#ff6b6b; border:1px solid rgba(255,77,77,0.18); }
        .badge-normal { background: rgba(0,200,120,0.06); color:#7ee787; border:1px solid rgba(0,200,120,0.08); }
        .glow-alert { border: 1px solid rgba(255,77,77,0.5); animation: glowRed 1.5s ease-in-out infinite; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Helper function to get all cars with owner info
    def get_all_cars_with_owners():
        cursor.execute("""
            SELECT cars.car_id, cars.make, cars.model, cars.year, cars.odometer,
                   users.full_name AS owner_name
            FROM cars
            JOIN users ON cars.user_id = users.user_id
            ORDER BY users.full_name, cars.make, cars.model
        """)
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame(columns=["car_id", "make", "model", "year", "odometer", "owner_name"])
        return pd.DataFrame(rows, columns=["car_id", "make", "model", "year", "odometer", "owner_name"])

    fleet_df = get_all_cars_with_owners()

    if fleet_df.empty:
        st.warning("No cars have been registered in the system yet. Please add cars in the Profile page.")
    else:
        # Add simulated live data to the real car data
        np.random.seed(int(time.time()))
        fleet_df['Engine Temp (°C)'] = np.random.uniform(80, 120, size=len(fleet_df)).round(1)
        fleet_df['Battery (V)'] = np.random.uniform(11.5, 14.5, size=len(fleet_df)).round(2)
        
        # Determine status based on simulated data for realism
        conditions = [
            (fleet_df['Engine Temp (°C)'] > 110),
            (fleet_df['Battery (V)'] < 12.0)
        ]
        choices = ['Alert', 'Alert']
        fleet_df['Status'] = np.select(conditions, choices, default='Normal')

        # Display overall metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("🚗 Total Vehicles in Fleet", len(fleet_df))
        with col_b:
            st.metric("🔥 Overheated Engines", len(fleet_df[fleet_df["Engine Temp (°C)"] > 110]))
        with col_c:
            st.metric("🔋 Low Battery", len(fleet_df[fleet_df["Battery (V)"] < 12.0]))

        st.markdown("### 🛠 Fleet Overview")
        
        # Display each car as a card
        cols = st.columns(4)
        for i, row in fleet_df.iterrows():
            with cols[i % 4]:
                status_class = "glow-alert" if row["Status"] == "Alert" else ""
                engine_class = "engine-pulse" if row["Engine Temp (°C)"] > 110 else ""
                battery_class = "battery-flicker" if row["Battery (V)"] < 12.0 else ""
                
                st.markdown(
                    f"""
                    <div class="card {status_class}">
                        <h4>{'🚨 ' if row['Status']=='Alert' else '🚗 '} {row['make']} {row['model']}</h4>
                        <p style="font-size: 0.9em; opacity: 0.8;">Owner: {row['owner_name']}</p>
                        <span class="badge {'badge-critical' if row['Status']=='Alert' else 'badge-normal'}">{row['Status']}</span>
                        <p><b>Odometer:</b> {row['odometer']} km</p>
                        <p><b>Engine Temp:</b> <span class="{engine_class}">{row['Engine Temp (°C)']} °C</span></p>
                        <p><b>Battery:</b> <span class="{battery_class}">{row['Battery (V)']} V</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ----------------------- HOME (FULL PROJECT LANDING PAGE) -----------------------
elif st.session_state.page == "Home":
    # 🖼 Hero Section with Banner + Headline
    st.markdown(
        """
        <div style="
            text-align:center;
            padding: 40px 20px;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f1724, #1a2435);
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
            margin-bottom: 25px;
        ">
            <h1 style="color:#7ee787; font-size:42px;">🚗 AutoCare AI</h1>
            <p style="font-size:20px; color:#d1d5db; max-width:700px; margin:auto;">
            Your <b>AI-powered vehicle health assistant</b> — monitor car vitals, predict failures,
            and get personalized maintenance recommendations before it's too late.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # MODIFIED: Use st.button for navigation
    cols_home_button = st.columns([3, 2, 3])
    with cols_home_button[1]:
        if st.button("🔑 Get Started", use_container_width=True, type="primary"):
            st.session_state.page = "Sign Up"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 🌟 Features Section
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.subheader("🌟 Why Choose AutoCare AI?")
    st.markdown(
        """
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap:15px;">
            <div style="background:#111827; padding:15px; border-radius:12px; text-align:center;">
                <h4>🧠 AI Diagnostics</h4>
                <p>Machine Learning models analyze car vitals and predict possible failures in advance.</p>
            </div>
            <div style="background:#111827; padding:15px; border-radius:12px; text-align:center;">
                <h4>🚦 Real-Time Alerts</h4>
                <p>Hybrid rule engine warns you about overheating, brake wear, battery issues, and more.</p>
            </div>
            <div style="background:#111827; padding:15px; border-radius:12px; text-align:center;">
                <h4>📊 Live Fleet Dashboard</h4>
                <p>Monitor multiple vehicles with live status, metrics, and aggregated health score.</p>
            </div>
            <div style="background:#111827; padding:15px; border-radius:12px; text-align:center;">
                <h4>📄 Detailed Reports</h4>
                <p>Download PDF reports with car details, diagnostics, maintenance tips & service centers.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ⚙️ How It Works Section
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.subheader("⚙️ How It Works")
    st.markdown(
        """
        <ol style="font-size:15px; line-height:1.8;">
        <li><b>Sign Up / Log In</b> – Create your account & add your car details.</li>
        <li><b>Enter Car Vitals</b> – Provide odometer, battery voltage, brake wear, etc.</li>
        <li><b>AI + Rule Analysis</b> – Get predicted failure risks & maintenance suggestions.</li>
        <li><b>View Dashboard</b> – Visual charts, alerts, and health confidence score.</li>
        <li><b>Download Report</b> – Generate a shareable PDF with all insights & recommendations.</li>
        </ol>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ✅ Call to Action Section
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3 style="text-align:center;">💡 Take Control of Your Car's Health Today</h3>
        <p style="text-align:center; font-size:15px;">
        Sign up now, connect your vehicle, and experience the power of predictive maintenance.
        </p>
        """,
        unsafe_allow_html=True
    )

# ----------------------- SIGN UP -----------------------
elif st.session_state.page == "Sign Up":
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.subheader("Create a new account")

    with st.form("signup_form"):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        mobile = st.text_input("Mobile Number", max_chars=18, placeholder="e.g., +91 9876543210")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm Password", type="password")

        submitted = st.form_submit_button("Create account")
        if submitted:
            # ---------------- VALIDATION CHECKS ----------------
            if not (username and password and full_name and email and mobile):
                st.error("Please fill all fields.")
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Enter a valid email address.")
            elif not re.match(r"^\+?[0-9\s-]{10,18}$", mobile):
                st.error("Enter a valid mobile number.")
            elif password != password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(username, password, full_name, email, mobile)
                if ok:
                    st.success(f"✅ Account created. Please proceed to the Log In page.")
                    # Automatically switch to Login page
                    st.session_state.page = "Log In"
                    st.rerun()
                else:
                    st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------- LOG IN PAGE -----------------------
elif st.session_state.page == "Log In":
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.subheader("🔑 Log in to your account")

    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if not (username and password):
                st.error("❌ Please enter both username and password.")
            else:
                authenticated, full_name = authenticate(username, password)
                if authenticated:
                    st.toast(f"✅ Welcome back, {full_name}!", icon="🎉")
                    login_user(username, full_name)
                else:
                    st.error("❌ Invalid username or password. Please try again.")
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------- Profile (car management) -----------------------
elif st.session_state.page == "Profile":
    if not current_user():
        st.warning("Please log in first to edit your profile.")
    else:
        username = current_user()
        user_details = get_user_by_username(username)

        # Personal info card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("👤 Profile Details")
        st.markdown(f"**Full Name:** {user_details['full_name']}")
        st.markdown(f"**Email:** {user_details['email']}")
        st.markdown(f"**Mobile:** {user_details.get('mobile','Not provided')}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Edit personal details
        with st.expander("✏️ Edit Personal Details"):
            with st.form("edit_profile_form"):
                new_name = st.text_input("Full Name", value=user_details['full_name'])
                new_email = st.text_input("Email", value=user_details['email'])
                new_mobile = st.text_input("Mobile Number", value=user_details.get('mobile', ""), max_chars=18)
                save_changes = st.form_submit_button("Save Changes")
                if save_changes:
                    if not (new_name and new_email and new_mobile):
                        st.error("All fields are required.")
                    elif not re.match(r"[^@]+@[^@]+\.[^@]+", new_email):
                        st.error("Enter a valid email address.")
                    elif not re.match(r"^\+?[0-9\s-]{10,18}$", new_mobile):
                        st.error("Enter a valid mobile number.")
                    else:
                        update_user_profile(username, new_name, new_email, new_mobile)
                        st.success("✅ Profile updated successfully.")
                        st.rerun()

        # Car management
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🚗 Your Cars")
        user_cars = get_cars_for_user(username)

        with st.form("add_car_form", clear_on_submit=True):
            st.markdown("#### ➕ Add a Car")
            make = st.text_input("🚘 Car Make", placeholder="e.g., Toyota")
            model = st.text_input("📑 Model", placeholder="e.g., Corolla")
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input("📅 Year", min_value=1980, max_value=2035, value=2015)
            with col2:
                odometer = st.number_input("🛣 Odometer (km)", min_value=0, step=500, value=50000)

            submitted = st.form_submit_button("➕ Add Car", use_container_width=True)
            if submitted:
                if not (make and model):
                    st.error("❌ Please enter both make and model.")
                else:
                    add_car(username, make, model, year, odometer)
                    st.success(f"✅ **{make} {model} ({year})** added successfully!")
                    st.rerun()

        # Show existing cars with edit/delete
        if not user_cars:
            st.info("No cars added yet. Use the form above to add one.")
        else:
            for c in user_cars:
                with st.expander(f"🚘 {c['make']} {c['model']} ({c['year']}) — {c['odometer']} km", expanded=False):
                    with st.form(f"edit_car_form_{c['car_id']}"):
                        new_make = st.text_input("Make", value=c['make'], key=f"make_{c['car_id']}")
                        new_model = st.text_input("Model", value=c['model'], key=f"model_{c['car_id']}")
                        new_year = st.number_input("Year", min_value=1980, max_value=2035, value=c['year'], key=f"year_{c['car_id']}")
                        new_odo = st.number_input("Odometer (km)", min_value=0, value=c['odometer'], step=500, key=f"odo_{c['car_id']}")

                        col1, col2 = st.columns([1, 0.4])
                        with col1:
                            save_car = st.form_submit_button("💾 Save Changes")
                        with col2:
                            delete_car_btn = st.form_submit_button("🗑 Delete", type="secondary")

                        if save_car:
                            update_car(c['car_id'], new_make, new_model, new_year, new_odo)
                            st.success("✅ Car details updated.")
                            st.rerun()
                        if delete_car_btn:
                            delete_car(c['car_id'])
                            st.warning(f"🗑 {c['make']} {c['model']} removed.")
                            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------- DASHBOARD (ENHANCED UI + PDF DOWNLOAD) -----------------------
elif st.session_state.page == "Dashboard":
    if not current_user():
        st.warning("Please log in to view dashboard.")
    else:
        username = current_user()
        full_name = st.session_state.get('full_name', username)
        
        # 🏁 Dashboard Hero Section
        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding: 25px;
                border-radius: 16px;
                background: linear-gradient(135deg, #0f1724, #1a2435);
                box-shadow: 0 8px 30px rgba(0,0,0,0.4);
                margin-bottom: 20px;
            ">
                <h1 style="color:#7ee787; font-size:32px;">📊 Vehicle Health Dashboard</h1>
                <p style="font-size:17px; color:#d1d5db; max-width:750px; margin:auto;">
                Hello <b>{full_name}</b> 👋 — Here's a quick snapshot of your car's current health, AI predictions, and recommendations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        cars = get_cars_for_user(username)
        if not cars:
            st.warning("🚗 You have no cars added. Please go to **Profile** and add one first.")
            if st.button("➕ Go to Profile Page", use_container_width=True):
                st.session_state["page"] = "Profile"
                st.rerun()
        else:
            st.success(f"✅ You have {len(cars)} car(s) registered.")
            left, right = st.columns([1, 1.4])

            # Left column: Car Selection & Vitals
            with left:
                st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
                st.subheader("🚗 Select Your Vehicle")
                options = [f"{c['make']} {c['model']} ({c['year']})" for c in cars]
                selected = st.selectbox("Choose car", options)
                sel_idx = options.index(selected)
                selected_car = cars[sel_idx]

                st.markdown("### 🛠 Quick Vitals")
                odometer_km = st.number_input("📏 Odometer (km)", value=selected_car.get('odometer', 50000))
                engine_temp_c = st.slider("🌡 Engine Temp (°C)", 60, 140, 95)
                battery_voltage_v = st.slider("🔋 Battery Voltage (V)", 10.5, 15.0, 13.8, 0.01)
                oil_pressure_kpa = st.slider("🛢 Oil Pressure (kPa)", 80, 600, 320)
                brake_wear_front = st.slider("🛑 Front Brake Wear (mm)", 0.5, 20.0, 7.5)

                if st.button("▶️ Run Diagnosis", key="run_mock", use_container_width=True):
                    st.session_state["diagnosis_input"] = {
                        "car": selected_car,
                        "odometer_km": odometer_km,
                        "engine_temp_c": engine_temp_c,
                        "battery_voltage_v": battery_voltage_v,
                        "oil_pressure_kpa": oil_pressure_kpa,
                        "brake_pad_wear_mm_front": brake_wear_front,
                        "brake_pad_wear_mm_rear": brake_wear_front,
                        "suspension_health_pct": 85,
                        "coolant_level_pct": 95,
                        "brake_fluid_level_pct": 95,
                        "fuel_level_pct": 70,
                        "transmission_fluid_temp_c": 85,
                        "tire_pressure_psi_fl": 32
                    }
                    st.session_state["page"] = "Detailed Analysis"
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
                st.markdown(
                    "<h3 style='text-align:center;'>🧠 AI Diagnosis & Final Verdict</h3>",
                    unsafe_allow_html=True
                )
                st.info(
                    "ℹ️ Run Diagnosis to generate AI predictions and detailed health report. "
                    "This will take you to the **Detailed Analysis** page."
                )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='mini-foot'>Use this page to select your vehicle and run diagnosis. Detailed results will open on the next page.</div>",
            unsafe_allow_html=True
        )

        # <<< CHANGE 2: ADDED THIS ELIF BLOCK FOR THE NEW PAGE >>>
elif st.session_state.page == "Contact Us":
     st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
     st.subheader("📞 Contact Us")
     st.markdown("For any support or inquiries, please reach out to us through the following channels:")
    
     st.markdown("---")
    
     st.markdown("📧 **Email**")
     st.markdown("support@autocareai.com")
    
     st.markdown("📱 **Phone**")
     st.markdown("+91-6261075631 (Mon-Fri, 9 AM - 6 PM IST)")
    
     st.markdown("📍 **Address**")
     st.markdown("Mahunaka, Indore, Madhya Pradesh, India")
    
     st.markdown("---")

     with st.form("contact_form",clear_on_submit=True):
        st.markdown("#### Send us a message")
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("Send Message")
        
     if submitted:
            # Regular expression for validating an Email
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            if not (name and email and message):
                st.error("Please fill out all the fields.")
            # <<< CHANGE IS HERE: Added elif to validate email format >>>
            elif not re.match(email_regex, email):
                st.error("Please enter a valid email address.")
            else:
                if add_submission(name, email, message):
                 st.success("Thank you for your message! We will get back to you shortly.")
                
                st.session_state.submissions.append({
                    "Name": name,
                    "Email": email,
                    "Message": message,
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })


     st.markdown("</div>", unsafe_allow_html=True)

     # <<< CHANGE 3: DISPLAY THE SUBMISSIONS TABLE IF IT'S NOT EMPTY >>>
     submissions_df = get_all_submissions()
     if not submissions_df.empty:
        st.markdown("---")
        st.subheader("Recent Submissions")
        st.dataframe(submissions_df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------- DETAILED ANALYSIS PAGE -----------------------
elif st.session_state.page == "Detailed Analysis":
    if "diagnosis_input" not in st.session_state:
        st.warning("Please run a diagnosis from the Dashboard first.")
        if st.button("⬅ Back to Dashboard"):
            st.session_state["page"] = "Dashboard"
            st.rerun()
    else:
        data = st.session_state["diagnosis_input"]
        selected_car = data["car"]
        username = current_user()
        user = get_user_by_username(username) if username else {"full_name": "Guest", "email": "N/A"}

        st.markdown(
            """
            <div style="
                text-align:center;
                padding: 25px;
                border-radius: 16px;
                background: linear-gradient(135deg, #0f1724, #1a2435);
                box-shadow: 0 8px 30px rgba(0,0,0,0.4);
                margin-bottom: 20px;
            ">
                <h1 style="color:#7ee787;">🔍 Detailed Analysis & Report</h1>
                <p style="color:#d1d5db;">Complete breakdown of your car's health, predicted issues, and maintenance schedule.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        odometer_km = data["odometer_km"]
        engine_temp_c = data["engine_temp_c"]
        battery_voltage_v = data["battery_voltage_v"]
        oil_pressure_kpa = data["oil_pressure_kpa"]
        brake_wear_front = data["brake_pad_wear_mm_front"]

        rule_alerts = check_custom_rules(data)
        predicted_failure, confidence, prediction_proba = predict_failure(data)
        all_alerts = set(rule_alerts)
        if predicted_failure not in ["None", "Normal"] and confidence >= 50:
            all_alerts.add(predicted_failure)

        if all_alerts:
            for alert in all_alerts:
                st.error(f"🚨 {alert}")
            st.metric(label="Prediction Confidence", value=f"{confidence:.1f}%")
            if len(all_alerts) > 1:
                st.warning(f"⚠ Multiple issues detected ({len(all_alerts)} alerts).")
        else:
            st.success("✅ Vehicle Health: NORMAL")
            st.metric(label="Confidence in Normal Status", value=f"{confidence:.1f}%")
            st.info("👍 No issues detected. Keep up the good maintenance!")

        st.subheader("📊 Model Confidence")
        conf_val = float(confidence) if isinstance(confidence, (int, float, np.floating, np.integer)) else 0.0
        fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf_val,
                gauge={'axis': {'range': [0,100]},
                       'bar': {'color': "#66e07f"},
                       'steps':[{'range':[0,50],'color':'#ff7b7b'},{'range':[50,80],'color':'#ffcf6b'},{'range':[80,100],'color':'#9be89b'}]},
                title={'text':"Model Confidence (%)"}
        ))
        fig_g.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        st.subheader("📈 Predicted Failure Probabilities")
        if prediction_proba is not None and encoder is not None:
            try:
                prob_df = pd.DataFrame(prediction_proba[0], index=encoder.classes_, columns=['Probability']).sort_values(by='Probability', ascending=False)
                bar = go.Figure(go.Bar(
                    x=prob_df.index,
                    y=prob_df['Probability'],
                    marker=dict(color=['#7ee787','#ff6b6b','#ffb84d','#ffd86b','#9bd0ff'][:len(prob_df)])
                ))
                bar.update_layout(yaxis=dict(range=[0,1]), height=300)
                st.plotly_chart(bar, use_container_width=True)
            except Exception:
                st.info("⚠ Probability breakdown unavailable (encoder missing).")
        else:
            st.info("⚠ Model probabilities not available.")
                
        st.subheader("🧾 Maintenance Recommendations")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Recommendations</div>", unsafe_allow_html=True)
        if all_alerts:
            st.markdown("- **Schedule service** for critical issues immediately.")
            st.markdown("- **Check oil & coolant levels**, and inspect brake pads.")
            st.markdown("- If battery low, consider alternator check and battery health test.")
        else:
            st.markdown("- All systems normal. Next routine check in 3 months or 5,000 km.")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("📄 Download Detailed Car Report", use_container_width=True):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("<b>AutoCare AI - Vehicle Health Report</b>", styles["Title"]))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("<b>👤 User Details</b>", styles["Heading2"]))
            elements.append(Paragraph(f"Name: {user['full_name']}", styles["Normal"]))
            elements.append(Paragraph(f"Email: {user['email']}", styles["Normal"]))
            elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles["Normal"]))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("<b>🚘 Car Details</b>", styles["Heading2"]))
            car_data = [["Make", "Model", "Year", "Odometer (km)"],
                        [selected_car['make'], selected_car['model'], selected_car['year'], selected_car['odometer']]]
            car_table = Table(car_data, hAlign='LEFT')
            car_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            elements.append(car_table)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("<b>📊 Diagnostic Readings</b>", styles["Heading2"]))
            vitals_data = [
                ["Parameter", "Value"],
                ["Odometer", f"{odometer_km} km"],
                ["Engine Temperature", f"{engine_temp_c} °C"],
                ["Battery Voltage", f"{battery_voltage_v:.2f} V"],
                ["Oil Pressure", f"{oil_pressure_kpa} kPa"],
                ["Front Brake Wear", f"{brake_wear_front} mm"],
            ]
            vitals_table = Table(vitals_data, hAlign='LEFT', colWidths=[180, 120])
            vitals_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            elements.append(vitals_table)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("<b>🛠 Diagnosis Summary</b>", styles["Heading2"]))
            if all_alerts:
                for a in sorted(all_alerts):
                    elements.append(Paragraph(f"⚠ {a}", styles["Normal"]))
            else:
                elements.append(Paragraph("✅ No critical issues detected. Car health is normal.", styles["Normal"]))
            elements.append(Spacer(1, 12))

            if prediction_proba is not None and encoder is not None:
                elements.append(Paragraph("<b>📈 Prediction Probabilities</b>", styles["Heading2"]))
                prob_df = pd.DataFrame(prediction_proba[0], index=encoder.classes_, columns=['Probability'])
                table_data = [["Class", "Probability"]] + [[cls, f"{p*100:.1f}%"] for cls, p in prob_df['Probability'].items()]
                prob_table = Table(table_data, hAlign='LEFT', colWidths=[220, 100])
                prob_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
                ]))
                elements.append(prob_table)
                elements.append(Spacer(1, 12))

            elements.append(Paragraph("<b>🧾 Recommended Maintenance</b>", styles["Heading2"]))
            if all_alerts:
                elements.append(Paragraph("• Schedule a service visit within 48 hours.", styles["Normal"]))
                elements.append(Paragraph("• Inspect brakes, coolant, and battery system immediately.", styles["Normal"]))
            elements.append(Paragraph("• Oil & coolant top-up every 5,000 km.", styles["Normal"]))
            elements.append(Paragraph("• Brake inspection every 10,000 km.", styles["Normal"]))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("<b>🏢 Nearby Service Centers</b>", styles["Heading2"]))
            service_data = [
                ["Service Center", "City", "Contact"],
                ["Maruti Suzuki Service Arena", "Delhi", "+91-9876543210"],
                ["Tata Motors Service Hub", "Mumbai", "+91-9988776655"],
                ["Hyundai Authorised Service", "Bangalore", "+91-9123456789"],
                ["Mahindra First Choice", "Chennai", "+91-9000112233"],
            ]
            service_table = Table(service_data, hAlign='LEFT', colWidths=[170, 120, 120])
            service_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            elements.append(service_table)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph(f"<i>Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M')}</i>", styles["Normal"]))
            elements.append(Paragraph("<i>This report is auto-generated by AutoCare AI. For official diagnostics, contact your nearest service center.</i>", styles["Italic"]))

            doc.build(elements)
            buffer.seek(0)

            st.download_button(
                label="⬇️ Download PDF Report",
                data=buffer,
                file_name=f"{selected_car['make']}_{selected_car['model']}_HealthReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

        if st.button("⬅ Back to Dashboard", use_container_width=True):
            st.session_state["page"] = "Dashboard"
            st.rerun()


        st.markdown("<div class='mini-foot'>Prototype UI — login, add cars, mock diagnosis, and polished visuals. Integrate ML & DB next.</div>", unsafe_allow_html=True)
