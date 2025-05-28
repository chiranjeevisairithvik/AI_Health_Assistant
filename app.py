from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import requests
from flask import current_app
from model import db, UserHealthData
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Import model functions 
from model import (
    load_anemia_model, predict_anemia_status,
    load_cholesterol_model, predict_cholesterol_status,
    load_blood_pressure_model, predict_bp_status,
    load_diabetes_model, predict_diabetes
)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    mobile = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# HealthRecord Modell
class HealthRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)  # Can be null if user not logged in
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    hemoglobin = db.Column(db.Float, nullable=False)
    systolic = db.Column(db.Float, nullable=False)
    diastolic = db.Column(db.Float, nullable=False)
    total_chol = db.Column(db.Float, nullable=False)
    ldl = db.Column(db.Float, nullable=False)
    hdl = db.Column(db.Float, nullable=False)
    hba1c = db.Column(db.Float, nullable=False)
    anemia = db.Column(db.String(50), nullable=False)
    cholesterol = db.Column(db.String(50), nullable=False)
    blood_pressure = db.Column(db.String(50), nullable=False)
    diabetes = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Ensure database is initialized
with app.app_context():
    db.create_all()

faiss_index = faiss.read_index("medical_faiss.index")

# Load extracted medical data for context retrieval
with open("medical_data.json", "r") as file:
    medical_data = json.load(file)

# Load LLaMA model
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama runs on port 11434

def query_llama_ollama(prompt):
    payload = {
        "model": "llama3.1",  # Use your installed model name
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "Error: No response from LLaMA")
    else:
        return "Error: Failed to connect to LLaMA server."

# Initialize sentence transformer for query embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Prompt template for RAG-based AI
prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="Medical Context: {context}\nUser Question: {query}\nAI Response:"
)

llm_chain = lambda query: query_llama_ollama(query)

def fetch_user_health_data(user_id):
    with app.app_context():  # Ensures the function runs within the Flask app context
        user_data = UserHealthData.query.filter_by(user_id=user_id).first()

        if user_data:
            return {
                "hemoglobin": user_data.hemoglobin,
                "blood_pressure": f"{user_data.systolic}/{user_data.diastolic}",
                "cholesterol": user_data.total_cholesterol,
                "hba1c": user_data.hba1c
            }
        else:
            return {"error": "No health data found"}
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

# Database simulation for medicine suppliers
medicine_suppliers = [
    {"email": "supplier1@example.com", "phone": "1234567890", "medicines": {"Paracetamol": 10, "Aspirin": 5}},
    {"email": "supplier2@example.com", "phone": "9876543210", "medicines": {"Ibuprofen": 20, "Amoxicillin": 15}}
]

@app.route('/api/nearby-hospitals', methods=['GET'])
def nearby_hospitals():
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if not lat or not lon:
        return jsonify({"error": "Missing location data"}), 400

    query = f"""
    [out:json];
    node[amenity=hospital](around:7000,{lat},{lon});
    out center;
    """

    response = requests.get(OVERPASS_API_URL, params={'data': query})

    if response.status_code == 200:
        data = response.json()
        hospitals = [
            {
                "name": node["tags"].get("name", "Unnamed Hospital"),
                "lat": node["lat"],
                "lon": node["lon"],
                "address": node["tags"].get("addr:full", "No address available")
            }
            for node in data.get("elements", [])
        ]
        return jsonify(hospitals)

    return jsonify({"error": "Failed to fetch hospitals"}), 500

@app.route('/api/place-order', methods=['POST'])
def place_order():
    medicine_name = request.form.get('medicine_name')
    quantity = int(request.form.get('quantity', 0))

    if not medicine_name or quantity <= 0:
        return jsonify({"message": "Invalid medicine order."}), 400

    for supplier in medicine_suppliers:
        if medicine_name in supplier["medicines"] and supplier["medicines"][medicine_name] >= quantity:
            return jsonify({
                "message": "Order accepted! Contact the supplier.",
                "supplier": {"email": supplier["email"], "phone": supplier["phone"]}
            })
    
    return jsonify({"message": "No supplier has the requested stock."})

# Homepage Route
@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('home.html')
    flash('Please log in to access your dashboard.', 'warning')
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']
        retype_password = request.form['retype_password']

        if password != retype_password:
            flash('Passwords do not match', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
        elif User.query.filter_by(mobile=mobile).first():
            flash('Mobile number already registered', 'danger')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(name=name, email=email, mobile=mobile, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            session['user_id'] = new_user.id  
            flash('Registration successful!', 'success')
            return redirect(url_for('home'))

    return render_template('register.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/manual-entry', methods=['GET', 'POST'])
def manual_entry():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            gender = request.form['gender']
            hemoglobin = float(request.form['hemoglobin'])
            systolic = float(request.form['systolic'])
            diastolic = float(request.form['diastolic'])
            total_chol = float(request.form['total_chol'])
            ldl = float(request.form['ldl'])
            hdl = float(request.form['hdl'])
            hba1c = float(request.form['hba1c'])

            # Predictions
            anemia_model, anemia_le_gender, anemia_le_status, anemia_scaler = load_anemia_model()
            anemia_result = predict_anemia_status(anemia_model, anemia_le_gender, anemia_le_status, anemia_scaler, age, gender, hemoglobin)

            chol_model = load_cholesterol_model()
            chol_result = predict_cholesterol_status(chol_model, age, gender, total_chol, ldl, hdl)

            bp_model, bp_le_gender, bp_le_status, bp_scaler, bp_features = load_blood_pressure_model()
            bp_result = predict_bp_status(bp_model, bp_le_gender, bp_le_status, bp_scaler, bp_features, age, gender, systolic, diastolic)

            diabetes_model, diabetes_le = load_diabetes_model()
            diabetes_result = predict_diabetes(diabetes_model, diabetes_le, age, gender, hba1c)

            # Store results in DB
            new_record = HealthRecord(
                user_id=session.get('user_id'),
                age=age,
                gender=gender,
                hemoglobin=hemoglobin,
                systolic=systolic,
                diastolic=diastolic,
                total_chol=total_chol,
                ldl=ldl,
                hdl=hdl,
                hba1c=hba1c,
                anemia=anemia_result,
                cholesterol=chol_result,
                blood_pressure=bp_result,
                diabetes=diabetes_result
            )
            db.session.add(new_record)
            db.session.commit()

            # Return predictions as JSON
            results = {
                "anemia": anemia_result,
                "cholesterol": chol_result,
                "blood_pressure": bp_result,
                "diabetes": diabetes_result
            }

            return jsonify({"results": results})

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("manual-entry.html")

@app.route('/diet-plan')
def diet_plan():
    return render_template('diet_plan.html')
@app.route('/analytics')
def analytics():
    # Retrieve health records, sorting by age first, then by timestamp if ages are the same
    records = HealthRecord.query.order_by(HealthRecord.age, HealthRecord.timestamp).all()

    # Convert records to JSON format
    hemoglobin = [{"age": r.age, "time": r.timestamp, "value": r.hemoglobin} for r in records]
    blood_pressure = [{"age": r.age, "time": r.timestamp, "value": (r.systolic + r.diastolic) / 2} for r in records]
    cholesterol = [{"age": r.age, "time": r.timestamp, "value": r.total_chol} for r in records]
    hba1c = [{"age": r.age, "time": r.timestamp, "value": r.hba1c} for r in records]

    if request.headers.get('Accept') == 'application/json':
        return jsonify({
            "hemoglobin": hemoglobin,
            "blood_pressure": blood_pressure,
            "cholesterol": cholesterol,
            "hba1c": hba1c
        })

    return render_template('analytics.html', hemoglobin=hemoglobin, blood_pressure=blood_pressure, cholesterol=cholesterol, hba1c=hba1c)


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        return render_template("chatbot.html")  # Show chatbot UI if accessed via browser

    data = request.get_json()
    user_message = data.get("message", "")

    # 🔹 Step 1: Retrieve User Data (Example User ID - Update as Needed)
    user_id = session.get('user_id', 1)  # Fetch user_id from session or default to 1
    user_data = fetch_user_health_data(user_id)  # Fetch from database

    # 🔹 Step 2: Format the User Data for Context
    health_context = f"""
    User Health Data:
    - Hemoglobin: {user_data.get('hemoglobin', 'N/A')} g/dL
    - Blood Pressure: {user_data.get('blood_pressure', 'N/A')} mmHg
    - Cholesterol: {user_data.get('cholesterol', 'N/A')} mg/dL
    - HbA1c: {user_data.get('hba1c', 'N/A')} %
    """
    # 🔹 Step 3: Use RAG - FAISS Retrieval + Health Data Context
    query_vector = embedder.encode([user_message])
    D, I = faiss_index.search(np.array(query_vector), k=3)

    retrieved_context = " ".join([medical_data[i]["content"] for i in I[0]])
    rag_prompt = f"""
        Medical Report Context:
        {retrieved_context}

        User Health Data:
        {health_context}

        User Question: {user_message}
        """

    ai_response = query_llama_ollama(rag_prompt)

    return jsonify({"response": ai_response})


@app.route('/telemedicine', methods=['GET', 'POST'])
def telemedicine():
    if request.method == 'POST':
        medicine_name = request.form['medicine_name']
        quantity = request.form['quantity']
        delivery_address = request.form['delivery_address']
        flash(f'Medicine order for {quantity} {medicine_name} placed successfully!', 'success')
        return redirect(url_for('telemedicine'))
    return render_template('telemedicine.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        flash('File uploaded successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
