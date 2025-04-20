# dynamic_pricing_bandit_app.py
from datasets import load_dataset
import pandas as pd
import numpy as np
import gradio as gr
import json

PRICE_POINTS = [5000, 7500, 10000, 12500, 15000]
SEGMENTS = ["Retail", "HNI", "Corporate"]
DEAL_TYPES = ["M&A Advisory", "Debt Issuance", "Equity Offering", "Restructuring"]
REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
INDUSTRIES = ["Technology", "Healthcare", "Financial Services", "Energy", "Consumer Goods", "Industrial"]

class ThompsonBandit:
    def __init__(self, n_arms):
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def select_arm(self):
        return np.argmax(np.random.beta(self.successes, self.failures))

    def update(self, arm, reward):
        if reward:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

# Load HF dataset
dataset = load_dataset("banking77", split="train[:500]")
df = pd.DataFrame(dataset)
df["segment"] = np.random.choice(SEGMENTS, len(df))
df["deal_type"] = np.random.choice(DEAL_TYPES, len(df))
df["deal_size"] = np.random.lognormal(mean=16, sigma=1.0, size=len(df)).astype(int)
df["region"] = np.random.choice(REGIONS, len(df))
df["industry"] = np.random.choice(INDUSTRIES, len(df))

bandit = ThompsonBandit(len(PRICE_POINTS))

def recommend_price(segment, deal_type, deal_size_str, region, industry):
    try:
        deal_size = float(deal_size_str.replace("$", "").replace(",", ""))
        arm = bandit.select_arm()
        price = PRICE_POINTS[arm]
        acceptance_prob = max(0.1, 1 - (price / PRICE_POINTS[-1]) * 0.8)
        accepted = np.random.binomial(1, acceptance_prob)
        bandit.update(arm, accepted)
        return f"Recommended Price: ${price:,}\nClient would {'accept' if accepted else 'decline'} this price."
    except Exception as e:
        return str(e)

with gr.Blocks() as app:
    gr.Markdown("# Dynamic Pricing Bandit App")
    with gr.Row():
        with gr.Column():
            segment_input = gr.Dropdown(choices=SEGMENTS, label="Client Segment")
            deal_type_input = gr.Dropdown(choices=DEAL_TYPES, label="Deal Type")
            deal_size_input = gr.Textbox(label="Deal Size (USD)", value="$50000000")
            region_input = gr.Dropdown(choices=REGIONS, label="Region")
            industry_input = gr.Dropdown(choices=INDUSTRIES, label="Industry")
            btn = gr.Button("Get Recommendation")
        with gr.Column():
            result = gr.Markdown()

    btn.click(fn=recommend_price,
              inputs=[segment_input, deal_type_input, deal_size_input, region_input, industry_input],
              outputs=result)

if __name__ == "__main__":
    app.launch()
