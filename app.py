import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import os 

# ----------------------------
# Model definition (MUST match training)
# ----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)




# ----------------------------
# Load model (CPU ONLY)
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "autoencoder.pth")
checkpoint = torch.load("autoencoder.pth", map_location="cpu", weights_only=False)


input_dim = checkpoint["input_dim"]
threshold = checkpoint["threshold"]

ae = Autoencoder(input_dim)
ae.load_state_dict(checkpoint["model_state_dict"], strict=False)
ae.eval()


# ----------------------------
# Prediction function
# ----------------------------
def predict_transaction(
    step, tx_type, amount,
    old_org, new_org, old_dest, new_dest
):
    x = np.array([
        step, tx_type, amount,
        old_org, new_org, old_dest, new_dest
    ], dtype=np.float32).reshape(1, -1)

    x = torch.tensor(x)

    with torch.no_grad():
        recon = ae(x)
        error = torch.mean((x - recon) ** 2).item()

    anomaly = "YES 🚨" if error > threshold else "NO ✅"

    return (
        f"Reconstruction error: {error:.6f}\n"
        f"Threshold: {threshold:.6f}\n"
        f"Anomaly detected: {anomaly}"
    )

# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚨 PaySim Fraud Detection
        **Unsupervised Autoencoder-based anomaly detection**
        """
    )

    with gr.Row():
        with gr.Column():
            step = gr.Number(label="Step ⏱️", precision=0)
            tx_type = gr.Dropdown(
                choices=[
                    ("CASH_IN", 0),
                    ("CASH_OUT", 1),
                    ("DEBIT", 2),
                    ("PAYMENT", 3),
                    ("TRANSFER", 4)
                ],
                label="Transaction Type 💳"
            )
            amount = gr.Number(label="Amount")

        with gr.Column():
            old_org = gr.Number(label="Old Balance (Origin)")
            new_org = gr.Number(label="New Balance (Origin)")
            old_dest = gr.Number(label="Old Balance (Destination)")
            new_dest = gr.Number(label="New Balance (Destination)")

    btn = gr.Button("🔍 Detect Fraud", variant="primary")
    output = gr.Textbox(lines=4, label="Result")

    btn.click(
        predict_transaction,
        [step, tx_type, amount, old_org, new_org, old_dest, new_dest],
        output
    )

demo.launch(theme=gr.themes.Soft(), share=True)

