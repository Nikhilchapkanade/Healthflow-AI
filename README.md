# Healthflow-AI

The "Digital Twin" Approach to Hospital Staffing
Managing an Emergency Room is a balancing act. Staff too heavily, and you waste budget; staff too lightly, and patients wait—sometimes dangerously long. This project replaces guesswork with a Digital Twin: a system that predicts the future using AI and then "rehearses" it in a virtual simulation to help managers make safer, smarter decisions.
+1

🧠 The Theory: A Two-Engine System
This dashboard isn't just displaying data; it is generating new information using a "Predict-then-Simulate" pipeline. Here is the human-readable theory behind the two engines running under the hood:

1. The Prediction Engine (The "Meteorologist")
Technology: Long Short-Term Memory (LSTM) Neural Network

Think of a standard algorithm like a goldfish—it has no memory of what happened five minutes ago. But hospital traffic is a story, not a random series of events. A busy Friday afternoon often leads to a busy Friday evening.


The "Memory" Concept: We utilize an LSTM, a special type of AI that has a "memory". It looks at the last 24 hours of data—patient arrivals, the day of the week, and the hour—and "remembers" the patterns.
+2


The Forecast: Just as a weather app predicts rain based on current pressure, our AI takes the current hospital state and forecasts exactly how many Non-Urgent, Urgent, and Critical patients will walk through the door over the next week.
+1

2. The Simulation Engine (The "Video Game")
Technology: Discrete Event Simulation (DES) with SimPy

Knowing how many patients are coming is only half the battle. You also need to know if your current team can handle them.

Why Averages Fail: You can't just use simple math (e.g., "10 patients divided by 2 doctors"). In the real world, patients arrive in bursts, treatments take unpredictable amounts of time, and bottlenecks happen.


The Virtual Rehearsal: We use Discrete Event Simulation to build a virtual ER. In this computer model, virtual patients actually walk in, get triaged by virtual nurses, wait for beds, and see doctors. By fast-forwarding this "video game" thousands of times, we can see exactly where lines will form and test "What-If" scenarios before they happen in real life.
+1

📂 Inside the Toolkit: What the Files Actually Do
Here is the breakdown of the files in this repository, explained as components of a machine:

hospital_model.h5 (The Pre-Trained Brain)
This file is the crystallized knowledge of the AI.


Role: It contains the complex mathematical structure (layers of neurons) that has already "learned" the patterns of patient arrivals. It doesn't need to be trained again; it is ready to predict the moment you wake it up.
+2

hospital_dashboard.py (The Conductor)
This is the interface where the human meets the machine.

Role: It orchestrates the entire show. It loads the "Brain" (model), sets up the "Virtual World" (simulation), and presents the results in an interactive dashboard where you can slide bars to add or remove staff and see the immediate impact.
+1

scaler_x.pkl & scaler_y.pkl (The Translators)
AI models speak a different language than us—they prefer numbers between 0 and 1.

Role: These files act as translators.


scaler_x: Shrinks real-world data (like "Hour 23") down so the AI can understand it.
+2


scaler_y: Takes the AI's tiny output numbers and translates them back into real human terms (like "5 Critical Patients").
+1

last_window.npy (The Starting Line)
You can't predict the future without knowing the immediate past.


Role: This file contains the snapshot of the very last 24 hours of recorded hospital activity. The AI uses this as its "jumping-off point" to start predicting the very next hour.

🎯 The Goal
The ultimate purpose of this tool is Risk-Free Experimentation.

In the real world, you can't just fire two doctors on a Saturday night to "see what happens." In this simulation, you can. It allows hospital administrators to find the breaking point of their system safely, ensuring that when real patients arrive, the resources are there to meet them.
