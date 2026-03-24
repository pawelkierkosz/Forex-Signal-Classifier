# Forex Trading Signal Classifier with Neural Networks

This repository contains our engineering thesis project: a **Forex trading signal classification system** built for the **EUR/USD** currency pair. The project combines **machine learning**, **feature engineering based on the ZigZag indicator**, and a full **web application stack** for signal visualization and model interaction.

The main goal of the project was to design and implement a system that supports investment decisions on the Forex market using neural networks. As part of the research, we developed and tested **MLP** and **LSTM** models, evaluated them with a **walk-forward** methodology, and built a practical application with a **FastAPI backend** and a **React frontend**.

---

## Overview

Financial time series are noisy, non-stationary, and difficult to model directly from raw OHLC candles. In this project, we focused on improving predictive performance through **carefully designed input features**, instead of relying only on raw price values.

The core idea of the system is to:
- process historical and live Forex candle data,
- generate **ZigZag-based structural market features**,
- enrich them with **technical indicators**,
- train machine learning models for prediction,
- expose the trained models through a backend API,
- visualize market data and model signals in a web interface.

The project was completed as part of an engineering thesis by a team of four students.

---

## Main Goals

- build a machine learning pipeline for **Forex signal classification**,
- compare neural and non-neural approaches,
- verify whether **feature engineering** improves performance over raw OHLC input,
- evaluate models in a realistic **walk-forward** setup,
- provide a usable web application for signal visualization in quasi-real time.

---

## Machine Learning Part

The project includes two main model families:

### 1. MLP (Multi-Layer Perceptron)
The MLP is the main classification model in the project.  
Its task is to predict the direction of the next important market move.

It uses:
- engineered input features,
- ZigZag-derived market structure,
- recent closing prices,
- technical indicators,
- normalization relative to the next candle open used as execution-oriented scaling.

### 2. LSTM
The LSTM part was developed for sequence-based financial analysis and long-range temporal context.  
It is kept as a separate module in the project and was used as part of the thesis experiments.

---

## Why Feature Engineering Matters Here

One of the most important parts of this project is the **feature engineering pipeline**.

Instead of feeding only raw OHLC candles into the model, we transform the market data into a richer representation that tries to capture the **structure of price movement**.

### Feature groups used in the project

#### ZigZag-based features
These features are built from detected ZigZag pivots and are meant to reduce market noise.

They include:
- prices of recent ZigZag pivots,
- local turning-point structure,
- geometric market context.

#### Recent closing prices
A fixed number of recent closing prices is added to preserve short-term local price context.

#### Technical indicators
The pipeline also computes classical technical analysis indicators such as:
- SMA,
- EMA,
- Bollinger Bands,
- MACD,
- RSI,
- Stochastic Oscillator,
- ATR,
- CCI,
- Momentum.

### Why this is important
Raw OHLC data alone often does not expose market structure clearly enough.  
The engineered features aim to:
- reduce noise,
- highlight turning points,
- describe momentum and volatility,
- give the classifier a more meaningful representation of the market state.

In our thesis experiments, this was one of the key factors behind improved results.

---

## Labeling Strategy

For the MLP classification task, labels are generated using the **next confirmed ZigZag pivot** after the current candle.

In simplified terms:
- the model receives features describing the market at time `t`,
- the target label is based on whether the next significant ZigZag move is upward or downward.

This means the model is not trained to predict random candle-to-candle noise, but rather the **next important confirmed move** in the market structure.

---

## Evaluation Methodology

Because financial time series are **non-stationary**, a classic random train/test split would be misleading.

Instead, the project uses **walk-forward evaluation**:
- train on earlier data,
- validate on the next segment,
- test on the following unseen segment,
- move the window forward and repeat.

This setup better reflects how a trading-oriented system would behave in changing market conditions.

Hyperparameter tuning is performed with **Optuna**.

---

## Benchmarking

The project also includes a benchmark based on **Random Forest**.

This allows comparison between:
- neural approaches,
- classical machine learning,
- engineered features,
- raw OHLC representations.

The thesis results showed that the **representation of the input data** had a very strong effect on final performance.

---

## Project Structure

```text
backend/
└── app/
    ├── model_files/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── data_pipeline.py
    │   └── train_model.py
    ├── model_files_lstm/
    │   ├── data_pipeline.py
    │   ├── lstm_config.py
    │   ├── lstm_model.py
    │   ├── lstm_section.py
    │   └── train_lstm.py
    ├── __init__.py
    ├── main.py
    ├── schemas.py
    ├── service.py
    └── requirements.txt

frontend/
└── src/
    ├── App.jsx
    ├── main.jsx
    └── styles.css
````

---

## Tech Stack

### Backend

* Python
* FastAPI
* PyTorch
* Pandas
* NumPy
* Optuna
* Uvicorn
* MetaTrader5
* Pydantic
* python-multipart

### Frontend

* React
* Vite
* JavaScript
* CSS

---

## Installation

## Backend

Create and activate a virtual environment:

```bash
cd backend
python -m venv venv
```

### Windows

```bash
.\venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If some packages are still missing in your local environment, install them manually, for example:

```bash
pip install torch
pip install MetaTrader5
pip install python-multipart
```

---

## Configuration

Before running the backend, update your local file paths in the configuration files.

Most importantly, check:

* `backend/app/model_files/config.py`
* `backend/app/model_files_lstm/lstm_config.py`

If your project uses additional local ingestion scripts or file watchers, update those paths as well.

---

## Training

### Train the MLP model

Run the MLP training script before starting the application:

```bash
python app/model_files/train_model.py
```

### Train the LSTM model

If you want to use the LSTM part:

```bash
python app/model_files_lstm/train_lstm.py
```

---

## Run the Backend API

Start the FastAPI server with Uvicorn:

```bash
uvicorn app.main:app --reload
```

By default, the backend should be available at:

```text
http://127.0.0.1:8000
```

---

## Frontend

Go to the frontend directory and start the development server:

```bash
cd frontend
npm install
npm run dev
```

The frontend should be available at:

```text
http://127.0.0.1:5173
```

---

## What the Application Does

The web application is designed to:

* visualize market candles,
* present ZigZag-derived structure,
* show model outputs and trading signals,
* communicate with the backend API,
* support near-real-time interaction with the trained models.

---

## Notes

* This project was created as part of an **engineering thesis**.
* The implementation reflects both the **research part** and the **application part** of the work.
* Some scripts may use **local file paths** and require adjustment before first run.
* The repository is best understood as a combination of:

  * experiment code,
  * model training pipeline,
  * data processing utilities,
  * web application.

---

## Authors

This project was developed as an engineering thesis by:

* **Paweł Kierkosz**
* **Paweł Kolec**
* **Adam Nowacki**
* **Bartłomiej Rudowicz**

### Main responsibilities in the thesis project

* **Paweł Kierkosz** – MLP model, feature engineering, classification experiments
* **Bartłomiej Rudowicz** – LSTM model and sequence-based analysis
* **Paweł Kolec** – React frontend and data visualization
* **Adam Nowacki** – backend architecture, FastAPI API, integration layer

---

## Repository Purpose

This repository is both:

* a record of our engineering thesis work,
* and a practical machine learning + web application project focused on financial time series.

The most important contribution of the project is the combination of:

* **domain-inspired feature engineering**,
* **machine learning for market signal prediction**,
* and a **working software system** that exposes these results in an accessible form.

```
