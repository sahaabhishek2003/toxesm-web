# ============================================================
# IMPORTS
# ============================================================
from flask import Flask, render_template, request,send_file, session
import joblib
import torch
import esm
import os
from predict_module import run_prediction
from datetime import datetime

# ============================================================
# INIT APP
# ============================================================
app = Flask(__name__)
app.secret_key = "toxesm_secret_key"


# ============================================================
# LOAD MODELS
# ============================================================
print("Loading ML models...")

models = {
    "xgb": joblib.load("XGB_based_toxicity_model.pkl"),
    "lgbm": joblib.load("LGBM_based_toxicity_model.pkl"),
    "et": joblib.load("ET_based_toxicity_model.pkl")
}

model_names = {
    "xgb": "XGBoost",
    "lgbm": "LGBM",
    "et": "Extra Trees"
}


# ============================================================
# LOAD ESM MODEL
# ============================================================
print("Loading ESM model...")

esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

esm_model = esm_model.to(device)
esm_model.eval()


# ============================================================
# FASTA PARSER
# ============================================================
def parse_fasta(text):
    sequences, headers = [], []
    current_seq, current_header = "", None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith(">"):
            if current_seq:
                sequences.append(current_seq)
                headers.append(current_header if current_header else f">seq{len(headers)+1}")
                current_seq = ""
            current_header = line
        else:
            current_seq += line

    if current_seq:
        sequences.append(current_seq)
        headers.append(current_header if current_header else f">seq{len(headers)+1}")

    return sequences, headers


# ============================================================
# HOME
# ============================================================
@app.route("/")
def home():
    return render_template("home.html")


# ============================================================
# PREDICT
# ============================================================
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # Inputs
        sequence = request.form.get("sequence")
        file = request.files.get("file")

        model_choice = request.form.get("model_choice", "xgb")

        if model_choice == "ensemble":
            selected_model = None
            model_name = "Conservative Ensemble"
        else:
            selected_model = models.get(model_choice, models["xgb"])
            model_name = model_names.get(model_choice, "XGBoost")

        # Feature selection
        selected_features = request.form.getlist("features")
        if not selected_features:
            selected_features = ["length"]

        sequences, headers = [], []

        # Text input
        if sequence and sequence.strip():
            seqs, hdrs = parse_fasta(sequence)
            sequences.extend(seqs)
            headers.extend(hdrs)

        # File input
        if file and file.filename:
            try:
                content = file.read().decode("utf-8")
            except:
                return "Invalid file format. Please upload a valid FASTA file."

            seqs, hdrs = parse_fasta(content)
            sequences.extend(seqs)
            headers.extend(hdrs)

        if not sequences:
            return "No input provided!"

        # Delegate to prediction module
        result = run_prediction(
            sequences=sequences,
            headers=headers,
            model=selected_model,
            model_name=model_name,
            esm_model=esm_model,
            batch_converter=batch_converter,
            device=device,
            selected_features=selected_features,
            all_models=models,
            mode=model_choice
        )

        # UNPACK RESULT
        table_html = result["table"]
        summary = result["summary"]
        note = result["note"]

        # INIT HISTORY
        if "history" not in session:
            session["history"] = []

        # STORE RESULT
        session["history"].append({
            "id": len(session["history"]) + 1,
            "table": table_html,
            "summary": summary,
            "note": note,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
        })

        # KEEP LAST 50 ONLY
        session["history"] = session["history"][-50:]

        return render_template(
            "result.html",
            table=table_html,
            summary=summary,
            note=note
        )

    return render_template("predict.html")

# ============================================================
# ALGORITHM PAGE
# ============================================================
@app.route("/algorithm")
def algorithm():
    return render_template("algorithm.html")

#------------------------------------------------------------
# Performance
#------------------------------------------------------------
@app.route("/performance")
def performance():
    return render_template("performance.html")

# ============================================================
# RESIDUE ANALYSIS PAGE
# ============================================================
@app.route("/residue_analysis", methods=["GET", "POST"])
def residue_analysis():

    if request.method == "POST":
        raw_input = request.form.get("sequence")

        if not raw_input or not raw_input.strip():
            return render_template("residue_analysis.html", error="Please enter a sequence.")

        # -------- FASTA PARSING --------
        lines = raw_input.strip().split("\n")

        sequence = ""
        for line in lines:
            if line.startswith(">"):
                continue
            sequence += line.strip()

        # Clean sequence
        VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = "".join([aa for aa in sequence.upper() if aa in VALID_AA])

        if not sequence or not sequence.strip():
            return render_template("residue_analysis.html", error="Please enter a sequence.")

        # Clean sequence
        VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = "".join([aa for aa in sequence.upper() if aa in VALID_AA])

        if len(sequence) < 2 or len(sequence) > 50:
            return render_template(
                "residue_analysis.html",
                error="Sequence must be between 2 and 50 amino acids."
            )

        try:
            # IMPORT FUNCTION HERE (IMPORTANT)
            from predict_module import compute_residue_importance

            importance = compute_residue_importance(
                sequence,
                models["xgb"],  # default model
                esm_model,
                batch_converter,
                device
            )

            return render_template(
                "residue_analysis.html",
                sequence=sequence,
                importance=importance
            )

        except Exception as e:
            print("Residue analysis error:", e)
            return render_template("residue_analysis.html", error=str(e))

    return render_template("residue_analysis.html")

# ============================================================
# DOWNLOAD PAGE
# ============================================================
@app.route("/download")
def download():
    data_folder = "static/data"

    datasets = [
        {
            "name": "Training Dataset",
            "file": "train_dataset.csv",
            "description": "Curated dataset used for model training."
        },
        {
            "name": "Test Dataset",
            "file": "test_dataset.csv",
            "description": "Independent dataset used for evaluation."
        },
        {
            "name": "Independent Validation Dataset",
            "file": "Independent_Dataset_for_model_testing.xlsx",
            "description": "Dataset used for external validation and comparison with existing tools."
        }
    ]

    for d in datasets:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "static", "data", d["file"])

        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            d["size"] = f"{size_kb:.1f} KB"
        else:
            d["size"] = "Not available"

    return render_template("download.html", datasets=datasets)


# ============================================================
# ABOUT PAGE
# ============================================================
@app.route("/about")
def about():
    return render_template("about.html")

# ============================================================
# HISTORY PAGE
# ============================================================
@app.route("/history")
def history():
    history_data = session.get("history", [])
    return render_template("history.html", history=history_data)


# ============================================================
# VIEW HISTORY RESULT
# ============================================================
@app.route("/history/<int:job_id>")
def view_history(job_id):
    history_data = session.get("history", [])

    for h in history_data:
        if h["id"] == job_id:
            return render_template(
                "result.html",
                table=h["table"],
                summary=h["summary"],
                note=h["note"]

            )

    return "Result not found"

# ============================================================
# CONTACT PAGE
# ============================================================
@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/download_result")
def download_result():
    return send_file("static/output.csv", as_attachment=True)




@app.route("/download_file/<filename>")
def download_file(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "static", "data", filename)

    if not os.path.exists(path):
        print("DEBUG PATH:", path)  # optional debug
        return "File not found", 404

    return send_file(path, as_attachment=True)

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)