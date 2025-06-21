from datasets import load_dataset
import os

# Load MedQuAD from Hugging Face
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

# Prepare output
output_dir = "docs/"
os.makedirs(output_dir, exist_ok=True)

# Group all Q&A into a single text file
with open(f"{output_dir}/medquad.txt", "w", encoding="utf-8") as f:
    for e in dataset:
        q = e["Question"].strip()
        a = e["Answer"].strip()
        if q and a:
            f.write(f"Q: {q}\nA: {a}\n\n")

print("✅ Saved all Q&A to data/docs/medquad.txt")
