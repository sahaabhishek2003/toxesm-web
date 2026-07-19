import esm
import torch
import pandas as pd
from tqdm import tqdm
import re

# file paths
train_path = r"C:\Users\sahaa\Downloads\Study Materials\Bioinformatics Notes\Project\Data\ESM-2\train_dataset.csv"
test_path = r"C:\Users\sahaa\Downloads\Study Materials\Bioinformatics Notes\Project\Data\ESM-2\test_dataset.csv"

output_train = r"C:\Users\sahaa\Downloads\Study Materials\Bioinformatics Notes\Project\Data\ESM-2\train_embeddings.csv"
output_test = r"C:\Users\sahaa\Downloads\Study Materials\Bioinformatics Notes\Project\Data\ESM-2\test_embeddings.csv"


print("Loading ESM-2 model...")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

model.eval()

valid_aa = re.compile("^[ACDEFGHIKLMNPQRSTVWY]+$")


def generate_embeddings(input_file, output_file):

    df = pd.read_csv(input_file)

    embeddings = []
    labels = []
    ids = []

    skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):

        seq = str(row["sequence"]).upper()
        label = row["label"]

        # skip invalid sequences
        if not valid_aa.match(seq):
            skipped += 1
            continue

        data = [("seq", seq)]

        batch_labels, batch_strs, tokens = batch_converter(data)

        with torch.no_grad():
            results = model(tokens, repr_layers=[33])

        token_embeddings = results["representations"][33]

        seq_embedding = token_embeddings[0, 1:len(seq)+1].mean(0)

        embeddings.append(seq_embedding.numpy())
        labels.append(label)
        ids.append(i)

    emb_df = pd.DataFrame(embeddings)

    emb_df.insert(0, "ID", ids)
    emb_df["label"] = labels

    emb_df.to_csv(output_file, index=False)

    print(f"\nSaved embeddings → {output_file}")
    print(f"Skipped invalid sequences: {skipped}")


print("Generating train embeddings...")
generate_embeddings(train_path, output_train)

print("Generating test embeddings...")
generate_embeddings(test_path, output_test)

print("All embeddings generated successfully.")