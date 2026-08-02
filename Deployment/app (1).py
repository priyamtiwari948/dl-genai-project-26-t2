"""
Smart MCQ Solver - Hugging Face Space (Gradio App)
====================================================
Deploys the from-scratch BiLSTM+attention model as an interactive demo.

Files needed in this Space repo:
    app.py                (this file)
    requirements.txt
    scratch_model_best.pt (your trained checkpoint - upload separately)
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import spaces

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["A", "B", "C", "D", "E"]
MAX_PROMPT_LEN = 64
MAX_OPT_LEN = 16

TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


def tokenize(text: str):
    text = str(text).lower()
    return TOKEN_RE.findall(text)


def encode(text, stoi, max_len):
    ids = [stoi.get(tok, 1) for tok in tokenize(text)][:max_len]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids


# ----------------------------------------------------------------------
# Model definition (must match training code exactly)
# ----------------------------------------------------------------------
class ScratchMCQModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_W = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        match_dim = hidden_dim * 2 * 4
        self.scorer = nn.Sequential(
            nn.Linear(match_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_seq(self, x):
        emb = self.embedding(x)
        out, _ = self.encoder(emb)
        return out

    def masked_mean(self, seq, ids):
        mask = (ids != 0).unsqueeze(-1).float()
        summed = (seq * mask).sum(1)
        count = mask.sum(1).clamp(min=1)
        return summed / count

    def forward(self, prompt_ids, option_ids):
        B, num_opts, Lo = option_ids.shape
        prompt_enc = self.encode_seq(prompt_ids)
        prompt_vec = self.masked_mean(prompt_enc, prompt_ids)

        scores = []
        for i in range(num_opts):
            opt_ids_i = option_ids[:, i, :]
            opt_enc = self.encode_seq(opt_ids_i)
            opt_vec = self.masked_mean(opt_enc, opt_ids_i)

            attn_scores = torch.bmm(self.attn_W(prompt_enc), opt_vec.unsqueeze(-1)).squeeze(-1)
            pad_mask = (prompt_ids == 0)
            attn_scores = attn_scores.masked_fill(pad_mask, -1e9)
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
            prompt_ctx = (attn_weights * prompt_enc).sum(1)

            diff = torch.abs(opt_vec - prompt_ctx)
            prod = opt_vec * prompt_ctx
            match_vec = torch.cat([opt_vec, prompt_ctx, diff, prod], dim=-1)
            score_i = self.scorer(match_vec).squeeze(-1)
            scores.append(score_i)

        return torch.stack(scores, dim=1)


# ----------------------------------------------------------------------
# Load checkpoint (contains both model weights and vocab)
# ----------------------------------------------------------------------
CHECKPOINT_PATH = "scratch_model_best.pt"
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
stoi = checkpoint["vocab_stoi"]

model = ScratchMCQModel(vocab_size=len(stoi)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()


# ----------------------------------------------------------------------
# Inference function used by the Gradio UI
# ----------------------------------------------------------------------
@spaces.GPU
def predict_mcq(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    if not prompt.strip() or not all([opt_a, opt_b, opt_c, opt_d, opt_e]):
        return "Please fill in the prompt and all 5 options."

    prompt_ids = torch.tensor([encode(prompt, stoi, MAX_PROMPT_LEN)]).to(DEVICE)
    options_text = [opt_a, opt_b, opt_c, opt_d, opt_e]
    opt_ids = torch.tensor(
        [[encode(t, stoi, MAX_OPT_LEN) for t in options_text]]
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(prompt_ids, opt_ids)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    ranked = sorted(zip(LABELS, options_text, probs), key=lambda x: -x[2])
    top3_labels = " ".join(r[0] for r in ranked[:3])

    result_lines = [f"**Top-3 ranked prediction: `{top3_labels}`**", ""]
    for label, text, prob in ranked:
        result_lines.append(f"- **{label}**: {text}  —  confidence: {prob:.3f}")

    return "\n".join(result_lines)


# ----------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------
demo = gr.Interface(
    fn=predict_mcq,
    inputs=[
        gr.Textbox(label="Question / Prompt", lines=3, placeholder="Enter the MCQ question..."),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
        gr.Textbox(label="Option E"),
    ],
    outputs=gr.Markdown(label="Prediction"),
    title="Smart MCQ Solver — From-Scratch BiLSTM + Attention Model",
    description=(
        "A model built entirely from scratch (no pretrained weights) using an "
        "embedding layer, BiLSTM encoder, and custom attention to rank the top-3 "
        "most likely correct answers among 5 options. Part of the Deep Learning "
        "& Generative AI project (BS Data Science, IIT Madras)."
    ),
    examples=[
        [
            "Pick the best possible answer: What is a planetary structure? among the listed options.",
            "A framework of planets that are all located in the same solar framework.",
            "A structure of planets that are all the same size and shape.",
            "Any set of gravitationally bound non-stellar objects in or out of orbit around a star or star mechanism.",
            "A mechanism of planets that are all located in the same galaxy.",
            "A framework of planets that are all made of gas.",
        ],
    ],
)

if __name__ == "__main__":
    demo.launch()
