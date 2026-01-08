import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import simpy
import random
import statistics
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Hospital ER Operations Dashboard",
    layout="wide"
)

# --- 1. Load Model & Assets (Cached) ---
# This function now loads your pre-trained files, it does NOT train
@st.cache_resource
def load_assets():
    st.write("Cache miss: Loading AI model, scalers, and data...")
    try:
        model = load_model('hospital_model.h5')
        scaler_x = joblib.load('scaler_x.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        last_window = np.load('last_window.npy')
        return model, scaler_x, scaler_y, last_window
    except FileNotFoundError:
        st.error("ERROR: Model or scaler files not found.")
        st.error("Please make sure 'hospital_model.h5', 'scaler_x.pkl', 'scaler_y.pkl', and 'last_window.npy' are in the same folder as the script.")
        return None, None, None, None

# --- 2. New Prediction Function ---
# This function generates *new* predictions "on-the-fly"
def generate_future_predictions(model, scaler_x, scaler_y, start_window, hours_to_predict=168):
    
    # We need the input feature names for creating the 'next_hour_features'
    input_features = ['hour', 'day_of_week', 'is_weekend', 'non_urgent_arrivals', 'urgent_arrivals', 'critical_arrivals']
    
    # Get the number of features
    n_features = start_window.shape[1]
    
    # Start with the last known data
    current_batch = start_window.reshape((1, LOOK_BACK, n_features))
    
    predictions_list = []
    
    # We need to get the last known hour and day to increment them
    # Un-scale the last row of the start_window to get the real values
    last_known_row_scaled = start_window[-1]
    last_known_row_unscaled = scaler_x.inverse_transform(last_known_row_scaled.reshape(1, -1))[0]
    
    # Create a dictionary for easy access
    last_known_features = dict(zip(input_features, last_known_row_unscaled))
    
    current_hour = last_known_features['hour']
    current_day = last_known_features['day_of_week']

    for i in range(hours_to_predict):
        # 1. Predict the next hour
        next_pred_scaled = model.predict(current_batch, verbose=0)[0]
        
        # 2. Store the un-scaled prediction
        next_pred_unscaled = scaler_y.inverse_transform(next_pred_scaled.reshape(1, -1))[0]
        predictions_list.append(next_pred_unscaled)
        
        # 3. Create the input features for the *next* prediction
        # Increment time
        current_hour = (current_hour + 1) % 24
        if current_hour == 0:
            current_day = (current_day + 1) % 7
        is_weekend = 1 if current_day in [5, 6] else 0
        
        # Get predicted arrivals
        non_urgent, urgent, critical = next_pred_unscaled
        
        # Build the new feature row
        next_hour_features = [
            current_hour,
            current_day,
            is_weekend,
            non_urgent,
            urgent,
            critical
        ]
        
        # 4. Scale and update the batch
        next_hour_features_scaled = scaler_x.transform([next_hour_features])[0]
        
        # Reshape for LSTM
        next_hour_features_scaled = next_hour_features_scaled.reshape((1, 1, n_features))
        
        # Append new prediction and drop first value
        current_batch = np.append(current_batch[:, 1:, :], next_hour_features_scaled, axis=1)

    # Convert predictions to a clean DataFrame
    prediction_df = pd.DataFrame(np.maximum(0, predictions_list).astype(int), columns=['Non-Urgent', 'Urgent', 'Critical'])
    return prediction_df

# --- 3. Simulation Logic (Unchanged) ---
# (Using global for simplicity in Streamlit, though passing state is cleaner)
wait_times_global = {}

class EmergencyRoomAcuity:
    def __init__(self, env, num_nurses, num_doctors, num_beds):
        self.env = env
        self.triage_nurse = simpy.PriorityResource(env, capacity=num_nurses)
        self.doctor = simpy.PriorityResource(env, capacity=num_doctors)
        self.bed = simpy.Resource(env, capacity=num_beds)
        self.triage_time = 5
        self.consult_urgent = 15
        self.consult_non_urgent = 10
        self.critical_stabilization = 40

    def patient_visit(self, patient_name, acuity):
        arrival_time = self.env.now
        priority_level = 1 if acuity == 'critical' else (2 if acuity == 'urgent' else 3)
        
        with self.triage_nurse.request(priority=priority_level) as req:
            yield req
            triage_start = self.env.now
            wait = triage_start - arrival_time
            wait_times_global[acuity].append(wait)
            yield self.env.timeout(self.triage_time)
        
        if acuity == 'critical':
            with self.doctor.request(priority=1) as doc_req, self.bed.request() as bed_req:
                yield doc_req & bed_req
                yield self.env.timeout(self.critical_stabilization)
        elif acuity == 'urgent':
            with self.doctor.request(priority=2) as doc_req:
                yield doc_req
                yield self.env.timeout(self.consult_urgent)
        else: 
            with self.doctor.request(priority=3) as doc_req:
                yield doc_req
                yield self.env.timeout(self.consult_non_urgent)

def run_simulation(env, critical_list, urgent_list, non_urgent_list, num_nurses, num_doctors, num_beds):
    er = EmergencyRoomAcuity(env, num_nurses, num_doctors, num_beds)
    for hour in range(len(critical_list)):
        patients_to_generate = []
        patients_to_generate.extend([('critical', f"{hour}-c{i}") for i in range(critical_list[hour])])
        patients_to_generate.extend([('urgent', f"{hour}-u{i}") for i in range(urgent_list[hour])])
        patients_to_generate.extend([('non_urgent', f"{hour}-n{i}") for i in range(non_urgent_list[hour])])
        random.shuffle(patients_to_generate)
        
        total_patients_this_hour = len(patients_to_generate)
        if total_patients_this_hour == 0:
            yield env.timeout(60); continue
            
        time_between_arrivals = 60.0 / total_patients_this_hour
        for acuity, name in patients_to_generate:
            env.process(er.patient_visit(name, acuity))
            yield env.timeout(time_between_arrivals)

# --- 4. The Streamlit App UI ---

st.title("🏥 Hospital Emergency Room: AI Operations Dashboard")

LOOK_BACK = 24 # This must match the LOOK_BACK you used for training
model, scaler_x, scaler_y, last_window = load_assets()

if model is not None:
    st.sidebar.header("🔧 Simulation Controls")
    sim_nurses = st.sidebar.slider("Number of Triage Nurses", 1, 10, 4)
    sim_doctors = st.sidebar.slider("Number of Doctors", 1, 10, 3)
    sim_beds = st.sidebar.slider("Number of Beds", 1, 20, 5)
    
    st.sidebar.header("🗓️ Forecast Period")
    sim_hours = st.sidebar.slider("Hours to Simulate", 24, 500, 168) # 168 hours = 1 week

    st.header(f"📈 AI Patient Load Forecast (Next {sim_hours} Hours)")
    
    # Generate predictions on-the-fly
    with st.spinner(f"Generating AI forecast for {sim_hours} hours..."):
        prediction_df = generate_future_predictions(model, scaler_x, scaler_y, last_window, sim_hours)
    st.line_chart(prediction_df)
    
    st.header("📊 'What-If' Simulation Results")

    if st.button(f"Run Simulation with {sim_nurses} Nurses, {sim_doctors} Doctors, {sim_beds} Beds"):
        
        wait_times_global = {'critical': [], 'urgent': [], 'non_urgent': []}
        
        predicted_critical = prediction_df['Critical'].tolist()
        predicted_urgent = prediction_df['Urgent'].tolist()
        predicted_non_urgent = prediction_df['Non-Urgent'].tolist()
        
        with st.spinner(f"Running simulation for {len(predicted_critical)} hours..."):
            random.seed(42)
            ai_sim_env = simpy.Environment()
            ai_sim_env.process(run_simulation(
                ai_sim_env, 
                predicted_critical, 
                predicted_urgent, 
                predicted_non_urgent,
                sim_nurses, sim_doctors, sim_beds
            ))
            ai_sim_env.run()
        
        st.success("Simulation complete!")

        col1, col2, col3 = st.columns(3)
        results = {}
        for acuity in ['critical', 'urgent', 'non_urgent']:
            waits = wait_times_global[acuity]
            results[acuity] = {
                'Avg Wait (min)': statistics.mean(waits) if waits else 0,
                'Max Wait (min)': max(waits) if waits else 0,
                'Patient Count': len(waits)
            }
        
        col1.metric("Avg. **Critical** Wait", f"{results['critical']['Avg Wait (min)']:.1f} min")
        col2.metric("Avg. **Urgent** Wait", f"{results['urgent']['Avg Wait (min)']:.1f} min")
        col3.metric("Avg. **Non-Urgent** Wait", f"{results['non_urgent']['Avg Wait (min)']:.1f} min")
        
        st.subheader("Operational Recommendation")
        avg_critical_wait = results['critical']['Avg Wait (min)']
        avg_urgent_wait = results['urgent']['Avg Wait (min)']

        if avg_critical_wait > 5:
            st.error(f"CRITICAL ALERT: Avg. wait for CRITICAL patients is {avg_critical_wait:.1f} min! Add staff immediately.")
        elif avg_urgent_wait > 15:
            st.warning(f"WARNING: Avg. wait for URGENT patients is {avg_urgent_wait:.1f} min. Consider adding 1 Nurse.")
        else:
            st.success("SUCCESS: Staffing levels appear adequate for the predicted load.")
            
        st.subheader("Full Results by Acuity")
        st.dataframe(pd.DataFrame(results).T)