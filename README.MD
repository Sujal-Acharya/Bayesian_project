
# 💰 Bayesian Bandit for Dynamic Pricing in Investment Banking

This project demonstrates a **Bayesian Multi-Armed Bandit (Thompson Sampling)** implementation tailored for **dynamic pricing of investment banking services** such as M&A advisory, equity offerings, debt issuance, and restructuring.

It simulates realistic client interactions and continuously learns which price points work best for different client segments based on deal-specific context.

---

## 🚀 Live Demo

👉 **[Click here to try the working prototype](https://huggingface.co/spaces/SujalAcharya/project)**  
Deployed using [Gradio](https://www.gradio.app/) on Hugging Face Spaces.

---

## 🧠 Concept Overview

### 🎯 Objective:
Implement a multi-armed bandit algorithm to dynamically adjust service pricing in investment banking based on:
- Market demand
- Client behavior
- Deal size
- Contextual factors (segment, industry, region)

### 💡 Why Multi-Armed Bandits?
Unlike static models, **bandit algorithms** dynamically explore multiple pricing strategies while maximizing expected revenue using **Bayesian inference**.

---

## 🧱 Core Components

### 📌 Price Points:
- `$5,000`, `$7,500`, `$10,000`, `$12,500`, `$15,000`

### 🧑‍💼 Client Segments:
- `Retail`, `HNI (High Net Worth Individuals)`, `Corporate`

### 📂 Deal Types:
- `M&A Advisory`, `Debt Issuance`, `Equity Offering`, `Restructuring`

### 🌍 Regions & Industries:
- Sampled across `5 global regions` and `6 industry verticals`

---

## 🧮 Algorithm: Thompson Sampling

- Each price point is modeled as a "slot machine arm"
- For each context (client segment), the model maintains a Beta distribution over success rates
- The model:
  1. **Samples from distributions** to pick a price
  2. **Observes success/failure** (based on simulated acceptance)
  3. **Updates distributions** accordingly

---

## 📊 Dataset

- Used Hugging Face's [`banking77`](https://huggingface.co/datasets/banking77) as a **proxy for investment banking client records**
- Augmented with synthetic:
  - Client segments
  - Deal sizes (log-normal)
  - Industry, region, and historical interaction counts

---

## 📈 Features

- ✅ Context-aware price recommendations
- ✅ Real-time simulation of pricing interactions
- ✅ Interactive Gradio UI with dropdowns for segment, deal type, etc.
- ✅ Revenue & acceptance tracking
- ✅ Exportable model state in JSON
- ✅ Visualizations: acceptance rates, price distributions, revenue share

---

## 📦 Files

- `dynamic_pricing_bandit_app.py`: Main Gradio application script
- `requirements.txt`: Python dependencies
- `README.md`: Project overview

---

## ⚙️ Setup Instructions (Local)

```bash
git clone <your-repo>
cd bandit_pricing_gradio
pip install -r requirements.txt
python dynamic_pricing_bandit_app.py
