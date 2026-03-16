# CAN-BUS Intrusion Detection System (IDS)

A Machine Learning based Intrusion Detection System designed to detect malicious activity in CAN (Controller Area Network) vehicle communication systems.

This project analyzes CAN traffic patterns and identifies abnormal behavior that may indicate attacks such as denial-of-service or message spoofing. The system combines real-time monitoring, machine learning based anomaly detection, and an interactive dashboard for visualization and alerts.

---

## Project Overview

Modern vehicles rely heavily on CAN-Bus networks for communication between ECUs (Electronic Control Units). While efficient, CAN networks lack built-in authentication and encryption, making them vulnerable to cyber attacks.

This project implements an IDS capable of detecting abnormal CAN traffic using machine learning models trained on real-world attack datasets.

The system provides:

• Real-time CAN packet monitoring
• Machine learning based intrusion detection
• Attack visualization through a web dashboard
• Training and model management interface
• Automated alerting mechanisms

---

## System Architecture

```
CAN Traffic / Dataset
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Detection
        │
        ▼
Detection Pipeline
        │
        ▼
Alert & Visualization System
        │
        ▼
Web Dashboard (Frontend)
```

The architecture separates the detection engine, training pipeline, and monitoring interface to allow modular development and experimentation.

---

## Features

### Real-Time Detection

Processes incoming CAN packets and analyzes them using trained machine learning models.

### Replay Engine

Allows historical CAN datasets to be replayed through the detection pipeline for testing and evaluation.

### Model Training Interface

Supports training new models and evaluating detection performance.

### Web-Based Monitoring Dashboard

Displays traffic activity, alerts, and detection metrics in real time.

### Alert System

Triggers notifications when suspicious traffic patterns are detected.

---

## Tech Stack

Backend:

* Python
* Flask
* WebSockets
* Machine Learning (Scikit-learn / related tools)

Frontend:

* React / Vite
* TypeScript
* Tailwind CSS

Tools & Environment:

* Linux / Virtual environments
* Git
* Network analysis tooling

---

## Repository Structure

```
backend/
    IDS pipeline
    training modules
    dataset loaders
    real-time detection components

frontend/
    monitoring dashboard
    visualization components
    control panels

scripts/
    dataset preparation
    evaluation utilities
    testing tools

config/
    dataset and model configuration files
```

---

## Running the Project

Clone the repository:

```
git clone https://github.com/Deekshith-GR/CAN-BUS-Intrusion-Detection-System.git
cd CAN-BUS-Intrusion-Detection-System
```

Install backend dependencies:

```
pip install -r backend/requirements.txt
```

Start the backend server:

```
python server.py
```

Start the frontend:

```
cd frontend
npm install
npm run dev
```

---

## Datasets

Due to size limitations, datasets are not included in this repository.

The system is designed to work with CAN intrusion detection datasets commonly used in research such as:

* Attack-free CAN traffic
* Denial-of-Service attack data
* Impersonation attack data
* Fuzzy attack data

Dataset paths can be configured through the project configuration files.

---

## Learning Goals

This project was built to explore:

• Vehicle network security
• Machine learning based intrusion detection
• Real-time security monitoring systems
• Security visualization dashboards

---

## Future Improvements

Potential areas for expansion include:

* Improved anomaly detection models
* Streaming pipeline optimization
* More advanced visualization and analytics
* Integration with additional vehicle network protocols

---

## Author

Deekshith GR
Computer Science Student – Cybersecurity Focus

GitHub: https://github.com/Deekshith-GR
