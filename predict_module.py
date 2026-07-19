# ============================================================
# IMPORTS
# ============================================================
from flask import render_template
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import os
import requests
import time
import numpy as np

# ============================================================
# CONSTANTS & HUGGING FACE API CONFIG
# ============================================================
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/esm2_t33_650M_UR50D"
HF_TOKEN = os.environ.get("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ============================================================
# API HELPER FOR EMBEDDINGS (ONE-BY-ONE PROCESSED)
# ============================================================
def get_esm_embeddings_api(sequences):
    """
    Queries Hugging Face's API for each sequence individually.
    Bypasses bulk payload size rejections on the free inference tier.
    """
    embeddings_dict = {}
    if not sequences:
        return embeddings_dict

    if not HF_TOKEN:
        print("CRITICAL: HF_TOKEN environment variable is missing on Render settings.")
        return embeddings_dict

    for idx, seq in enumerate(sequences):
        # Send one single string to the API at a time
        payload = {"inputs": seq, "options": {"wait_for_model": True}}
        try:
            response = requests.post(HF_API_URL, json=payload, headers=HEADERS, timeout=60)

            # If the model is cold/waking up, pause and try one more time
            if response.status_code == 503:
                time.sleep(12)
                response = requests.post(HF_API_URL, json=payload, headers=HEADERS, timeout=60)

            if response.status_code == 200:
                output = response.json()
                raw_arr = np.array(output)

                # Single sequence queries return shape: [1, sequence_length, 1280]
                if len(raw_arr.shape) == 3:
                    # Mean-pool along the sequence length, dropping BOS/EOS tokens
                    mean_emb = raw_arr[0, 1:-1, :].mean(axis=0)
                elif len(raw_arr.shape) == 2:
                    mean_emb = raw_arr[1:-1, :].mean(axis=0)
                else:
                    mean_emb = raw_arr.flatten()

                # Validate feature dimension length requirements
                if mean_emb.shape[0] == 1280:
                    embeddings_dict[idx] = mean_emb
                else:
                    print(f"Dimensions mismatch for sequence {idx}. Found shape: {mean_emb.shape}")
            else:
                print(f"HF API Error for sequence {idx}: Status {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Exception during HF API call for sequence {idx}: {e}")

    return embeddings_dict


# ============================================================
# RESIDUE IMPORTANCE FUNCTION
# ============================================================
def compute_residue_importance(sequence, model, esm_model=None, batch_converter=None, device=None):
    mutated_sequences = [sequence]
    for i in range(len(sequence)):
        mutated = sequence[:i] + sequence[i + 1:]
        mutated_sequences.append(mutated)

    embeddings_map = get_esm_embeddings_api(mutated_sequences)

    if 0 not in embeddings_map:
        return [0] * len(sequence)

    original_emb = embeddings_map[0].reshape(1, -1)
    original_score = model.predict_proba(original_emb)[0][1]

    importance = []
    for i in range(len(sequence)):
        mut_idx = i + 1
        if mut_idx in embeddings_map:
            mut_emb = embeddings_map[mut_idx].reshape(1, -1)
            try:
                new_score = model.predict_proba(mut_emb)[0][1]
                diff = original_score - new_score
            except:
                diff = 0
        else:
            diff = 0
        importance.append(diff)

    importance = np.array(importance)
    importance[importance < 0] = 0
    importance[importance < 0.01] = 0

    max_val = importance.max()
    if max_val > 0:
        importance = importance / max_val

    return importance.tolist()


# ============================================================
# MAIN FUNCTION
# ============================================================
def run_prediction(sequences, headers, model, model_name,
                   esm_model, batch_converter, device, selected_features,
                   all_models=None, mode="xgb"):
    results = []

    cleaned_sequences = []
    invalid_flags = []
    api_batch = []
    api_to_original_idx = []

    for i, seq in enumerate(sequences):
        original_seq = seq.upper()
        clean_seq = "".join([aa for aa in original_seq if aa in VALID_AA])
        has_invalid_aa = len(clean_seq) != len(original_seq)

        cleaned_sequences.append(clean_seq)
        invalid_flags.append(has_invalid_aa)

        if 2 <= len(clean_seq) <= 50 and not has_invalid_aa:
            api_batch.append(clean_seq)
            api_to_original_idx.append(i)

    embeddings = {}
    if api_batch:
        api_results = get_esm_embeddings_api(api_batch)
        for api_idx, original_idx in enumerate(api_to_original_idx):
            if api_idx in api_results:
                embeddings[original_idx] = api_results[api_idx]

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

            analysed_seq = ProteinAnalysis(clean_seq)
            length = len(clean_seq)

            def safe_calc(func):
                try:
                    return func()
                except:
                    return "NA"

            mw = safe_calc(analysed_seq.molecular_weight)
            aromaticity = safe_calc(analysed_seq.aromaticity)
            instability = safe_calc(analysed_seq.instability_index)
            gravy = safe_calc(analysed_seq.gravy)
            charge = safe_calc(lambda: analysed_seq.charge_at_pH(7.0))
            pi = safe_calc(analysed_seq.isoelectric_point)
            extinction = safe_calc(lambda: analysed_seq.molar_extinction_coefficient()[0])

            row = {"Header": header, "Sequence": clean_seq}
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

            if has_invalid_aa or length == 1 or length > 50 or i not in embeddings:
                fill_val = "Error (API fail)" if (i not in embeddings and 2 <= length <= 50 and not has_invalid_aa) else "*"
                if mode == "ensemble":
                    row["XGBoost Score"] = fill_val
                    row["LGBM Score"] = fill_val
                    row["Extra Trees Score"] = fill_val
                    row["Final Prediction"] = fill_val
                    row["Final Prediction (Raw)"] = fill_val
                else:
                    row[f"{model_name} Prediction"] = fill_val
                    row[f"{model_name} Prediction (Raw)"] = fill_val
                    row[f"{model_name} Score"] = fill_val
                results.append(row)
                continue

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
                row["Final Prediction"] = f'<span style="color:{color}; font-weight:600;">{final_label}</span>'
                row["Final Prediction (Raw)"] = final_label
            else:
                proba = model.predict_proba(embedding_df)[0]
                score = proba[1]
                label_text = "Toxic" if score >= 0.5 else "Non-Toxic"
                color = "#d9534f" if score >= 0.5 else "#28a745"

                row[f"{model_name} Prediction"] = f'<span style="color:{color}; font-weight:600;">{label_text}</span>'
                row[f"{model_name} Prediction (Raw)"] = label_text
                row[f"{model_name} Score"] = round(score, 6)

            results.append(row)
        except Exception as e:
            results.append({"Header": header, "Sequence": clean_seq, "Error": str(e)})

    df = pd.DataFrame(results)
    df.insert(0, "Seq ID", [f"seq{idx}" for idx in range(1, len(df) + 1)])
    os.makedirs("static", exist_ok=True)

    df_export = df.copy()
    for col in df_export.columns:
        if "(Raw)" in col:
            clean_col = col.replace(" (Raw)", "")
            df_export[clean_col] = df_export[col]
    df_export = df_export[[c for c in df_export.columns if "(Raw)" not in c]]
    df_export.to_csv("static/output.csv", index=False)

    note = "Conservative Ensemble Mode: Sequence is classified as toxic if ANY model predicts toxicity." if mode == "ensemble" else \
        "* Sequences longer than 50 amino acids cannot be reliably predicted by ToxESM."

    df_display = df[[c for c in df.columns if "(Raw)" not in c]]
    table_html = df_display.to_html(index=False, classes="result-table", border=0, escape=False)

    toxic_count = 0
    non_toxic_count = 0
    invalid_count = 0

    pred_col = "Final Prediction (Raw)" if mode == "ensemble" else f"{model_name} Prediction (Raw)"
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

    return {"table": table_html, "summary": summary, "note": note}