# ============================================================
# IMPORTS
# ============================================================
from flask import render_template
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import torch
import os


# ============================================================
# CONSTANTS
# ============================================================
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ============================================================
# RESIDUE IMPORTANCE FUNCTION
# ============================================================
def compute_residue_importance(sequence, model, esm_model, batch_converter, device):

    def get_score(seq):
        data = [("protein", seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[33])

        embedding = results["representations"][33]
        emb = embedding[0, 1:-1].mean(0).cpu().numpy().reshape(1, -1)

        return model.predict_proba(emb)[0][1]

    original_score = get_score(sequence)

    importance = []

    for i in range(len(sequence)):
        mutated = sequence[:i] + sequence[i+1:]

        try:
            new_score = get_score(mutated)
            diff = original_score - new_score
        except:
            diff = 0

        importance.append(diff)

    import numpy as np

    importance = np.array(importance)

    # Remove negative contributions
    importance[importance < 0] = 0

    # Remove noise
    importance[importance < 0.01] = 0

    # Normalize
    max_val = importance.max()
    if max_val > 0:
        importance = importance / max_val

    # ===========================

    return importance.tolist()

# ============================================================
# MAIN FUNCTION
# ============================================================
def run_prediction(sequences, headers, model, model_name,
                   esm_model, batch_converter, device, selected_features,
                   all_models=None, mode="xgb"):
    results = []

    # ====================================================
    # CLEAN SEQUENCES + PREPARE BATCH
    # ====================================================
    valid_batch = []
    valid_indices = []
    cleaned_sequences = []
    invalid_flags = []

    for i, seq in enumerate(sequences):
        original_seq = seq.upper()

        clean_seq = "".join([aa for aa in original_seq if aa in VALID_AA])

        has_invalid_aa = len(clean_seq) != len(original_seq)
        cleaned_sequences.append(clean_seq)
        invalid_flags.append(has_invalid_aa)

        if 2 <= len(clean_seq) <= 50:
            valid_batch.append(("seq", clean_seq))
            valid_indices.append(i)

    # ====================================================
    # BATCH ESM INFERENCE
    # ====================================================
    embeddings = {}

    if valid_batch:
        _, _, tokens = batch_converter(valid_batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            results_esm = esm_model(tokens, repr_layers=[33])

        # Get representations from layer 33
        token_embeddings = results_esm["representations"][33]

        for idx, (i, (_, seq)) in enumerate(zip(valid_indices, valid_batch)):
            # Mean pooling over the sequence length (excluding BOS/EOS tokens)
            emb = token_embeddings[idx, 1:len(seq) + 1].mean(0)
            embeddings[i] = emb.cpu().numpy()

    # ====================================================
    # PROCESS EACH SEQUENCE
    # ====================================================
    for i, (seq, header) in enumerate(zip(sequences, headers)):
        clean_seq = cleaned_sequences[i]
        has_invalid_aa = invalid_flags[i]

        try:
            if len(clean_seq) == 0:
                row = {"Header": header, "Sequence": clean_seq, "Note": "Empty or invalid sequence"}
                if has_invalid_aa:
                    row["Note"] = "Contains non-standard amino acids"
                    results.append(row)
                    continue
                results.append(row)
                continue

            analysed_seq = ProteinAnalysis(clean_seq) if len(clean_seq) > 0 else None

            def safe_calc(func):
                try:
                    return func() if analysed_seq else "NA"
                except:
                    return "NA"

            # Calculate Biophysical Properties
            length = len(clean_seq)
            mw = safe_calc(analysed_seq.molecular_weight)
            aromaticity = safe_calc(analysed_seq.aromaticity)
            instability = safe_calc(analysed_seq.instability_index)
            gravy = safe_calc(analysed_seq.gravy)
            charge = safe_calc(lambda: analysed_seq.charge_at_pH(7.0))
            pi = safe_calc(analysed_seq.isoelectric_point)
            extinction = safe_calc(lambda: analysed_seq.molar_extinction_coefficient()[0])

            # Build Row
            row = {"Header": header, "Sequence": clean_seq}

            # Map selected features
            feat_map = {
                "length": ("Length", length),
                "mw": ("Molecular Weight", round(mw, 2) if isinstance(mw, float) else "NA"),
                "pi": ("pI", round(pi, 2) if isinstance(pi, float) else "NA"),
                "charge": ("Net Charge", round(charge, 2) if isinstance(charge, float) else "NA"),
                "gravy": ("GRAVY", round(gravy, 3) if isinstance(gravy, float) else "NA"),
                "instability": ("Instability Index", round(instability, 2) if isinstance(instability, float) else "NA"),
                "aromaticity": ("Aromaticity", round(aromaticity, 3) if isinstance(aromaticity, float) else "NA"),
                "extinction": ("Extinction Coefficient", extinction)
            }

            for key, (label, val) in feat_map.items():
                if key in selected_features:
                    row[label] = val

            # ----------------------------
            # LENGTH & EMBEDDING FILTERS
            # ----------------------------

            # FIX: ensure invalid AA never reaches prediction
            if has_invalid_aa:
                row[f"{model_name} Prediction"] = "*"
                row[f"{model_name} Prediction (Raw)"] = "*"
                row[f"{model_name} Score"] = "*"
                results.append(row)
                continue

            # Handle single AA
            if length == 1:
                row[f"{model_name} Prediction"] = "*"
                row[f"{model_name} Prediction (Raw)"] = "*"
                row[f"{model_name} Score"] = "*"
                results.append(row)
                continue

            # Handle long sequences
            if length > 50:
                row[f"{model_name} Prediction"] = "*"
                row[f"{model_name} Prediction (Raw)"] = "*"
                row[f"{model_name} Score"] = "*"
                results.append(row)
                continue

            # Handle missing embeddings
            if i not in embeddings:
                row[f"{model_name} Prediction"] = "*"
                row[f"{model_name} Prediction (Raw)"] = "*"
                row[f"{model_name} Score"] = "*"
                results.append(row)
                continue
            # ----------------------------
            # PREDICTION LOGIC
            # ----------------------------
            embedding_df = pd.DataFrame([embeddings[i]])

            if mode == "ensemble" and all_models:
                scores = {}
                labels = []
                for key, m in all_models.items():
                    p = m.predict_proba(embedding_df)[0][1]
                    scores[key] = round(p, 6)
                    labels.append("Toxic" if p >= 0.5 else "Non-Toxic")

                final_label = "Toxic" if "Toxic" in labels else "Non-Toxic"
                row["XGBoost Score"] = scores.get("xgb", "NA")
                row["LGBM Score"] = scores.get("lgbm", "NA")
                row["Extra Trees Score"] = scores.get("et", "NA")

                color = "#d9534f" if final_label == "Toxic" else "#28a745"
                # HTML for UI
                row["Final Prediction"] = f'<span style="color:{color}; font-weight:600;">{final_label}</span>'

                # RAW for CSV
                row["Final Prediction (Raw)"] = final_label

            else:
                proba = model.predict_proba(embedding_df)[0]
                score = proba[1]
                label_text = "Toxic" if score >= 0.5 else "Non-Toxic"
                color = "#d9534f" if score >= 0.5 else "#28a745"

                # HTML for UI
                row[f"{model_name} Prediction"] = f'<span style="color:{color}; font-weight:600;">{label_text}</span>'

                # RAW for CSV
                row[f"{model_name} Prediction (Raw)"] = label_text

                row[f"{model_name} Score"] = round(score, 6)

            results.append(row)

        except Exception as e:
            results.append({"Header": header, "Sequence": clean_seq, "Error": str(e)})

    # ====================================================
    # OUTPUT GENERATION
    # ====================================================
    df = pd.DataFrame(results)
    df.insert(0, "Seq ID", [f"seq{i}" for i in range(1, len(df) + 1)])

    os.makedirs("static", exist_ok=True)

    # ----------------------------
    # CLEAN CSV EXPORT (NO HTML)
    # ----------------------------
    df_export = df.copy()

    for col in df_export.columns:
        if "(Raw)" in col:
            clean_col = col.replace(" (Raw)", "")
            df_export[clean_col] = df_export[col]

    # Remove RAW helper columns
    df_export = df_export[[c for c in df_export.columns if "(Raw)" not in c]]

    df_export.to_csv("static/output.csv", index=False)

    note = "Conservative Ensemble Mode: Sequence is classified as toxic if ANY model predicts toxicity." if mode == "ensemble" else \
        "* Sequences longer than 50 amino acids cannot be reliably predicted by ToxESM."

    df_display = df[[c for c in df.columns if "(Raw)" not in c]]
    table_html = df_display.to_html(index=False, classes="result-table", border=0, escape=False)

    # COUNT SUMMARY
    toxic_count = 0
    non_toxic_count = 0
    invalid_count = 0

    if mode == "ensemble":
        pred_col = "Final Prediction (Raw)"
    else:
        pred_col = f"{model_name} Prediction (Raw)"

    for val in df[pred_col]:
        if val == "Toxic":
            toxic_count += 1
        elif val == "Non-Toxic":
            non_toxic_count += 1
        else:
            invalid_count += 1

    total = len(df)

    summary = {
        "toxic": toxic_count,
        "non_toxic": non_toxic_count,
        "invalid": invalid_count,
        "toxic_pct": (toxic_count / total) * 100 if total else 0,
        "non_toxic_pct": (non_toxic_count / total) * 100 if total else 0,
        "invalid_pct": (invalid_count / total) * 100 if total else 0
    }

    return {
        "table": table_html,
        "summary": summary,
        "note": note
    }