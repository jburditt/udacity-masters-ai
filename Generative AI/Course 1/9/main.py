import numpy as np
from evaluate import load
from sentence_transformers import SentenceTransformer
import random

# Set a seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#region EXACT MATCH

# Let's compare predicted fruit names with the correct labels.
preds = ["Apple", "banana ", " Orange"]
labels = ["apple", "banana", "grape"]

def normalize(s: str) -> str:
    """Normalize a string by lowercasing and stripping whitespace."""
    return s.lower().strip()

def exact_match(pred: str, label: str) -> int:
    """Return 1 if normalized strings are identical, else 0."""
    return int(normalize(pred) == normalize(label))

# Calculate EM score for each pair
em_scores = [exact_match(p, l) for p, l in zip(preds, labels)]

# The final score is the average of individual scores
em_accuracy = sum(em_scores) / len(em_scores)

print(f"Individual Scores: {em_scores}")
print(f"Average Exact Match Accuracy: {em_accuracy:.2f}")

#endregion

#region ROUGE

# Define a prediction and a reference text
pred = "the quick brown fox"
label = "the fox is quick and brown"

# Load the ROUGE metric from the 'evaluate' library
rouge = load("rouge")

# Compute the scores
results = rouge.compute(predictions=[pred], references=[label])

print(f"ROUGE-1 Score: {results['rouge1']:.4f}")
print(f"ROUGE-L Score: {results['rougeL']:.4f}")

#endregion