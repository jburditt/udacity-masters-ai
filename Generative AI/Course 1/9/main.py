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

#region SEMANTIC SIMILARITY
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define prediction and label sentences
labels = ["A dog is a loyal pet", "Cats are independent animals", "The sky is blue"]
preds = [
    "Dogs make great companions",
    "A cat is a solitary creature",
    "The ocean is vast",
]

# 3. Generate embeddings for each list
pred_embeddings = model.encode(preds)
label_embeddings = model.encode(labels)

# 4. Calculate cosine similarity for each pair
for i in range(len(preds)):
    similarity = np.dot(pred_embeddings[i], label_embeddings[i]) / (
        np.linalg.norm(pred_embeddings[i]) * np.linalg.norm(label_embeddings[i])
    )
    print(
        f"Pair {i + 1}:\n  Pred:  '{preds[i]}'\n  Label: '{labels[i]}'\n  Similarity: {similarity:.4f}\n"
    )
    
#endregion

#region FUNCTIONAL CORRECTNESS

# This function is supposed to reverse and capitalize a string,
# but it has a bug: it fails if the string contains a number.
def reverse_and_capitalize(s: str) -> str:
    """Reverse and capitalize a string, with a hidden bug."""
    if any(char.isdigit() for char in s):
        return "ERROR - CONTAINS DIGITS"
    return s[::-1].upper()

# Test cases: one prediction will trigger the bug
code_preds = ["hello", "world1", "python"]
test_labels = ["OLLEH", "1DLROW", "NOHTYP"]

# Run the generated code against the test labels
results = []
for pred_code, label in zip(code_preds, test_labels):
    output = reverse_and_capitalize(pred_code)
    print(f"Input: '{pred_code}' -> Output: '{output}', Expected: '{label}'")
    results.append(output == label)

pass_rate = sum(results) / len(results)
print(f"\nProportion of tests passed: {pass_rate:.2f}")

#endregion

#region PASS@K

def pass_at_k(samples: list[str], label: str) -> int:
    """Return 1 if any sample in the list matches the label, else 0."""
    return int(any(s == label for s in samples))

# The model generated 4 possible answers for "Name a primary color."
label = "blue"
samples = ["red", "yellow", "green", "blue"]

# Check if any of the 4 samples is correct
pass_score = pass_at_k(samples, label)

print(f"Samples: {samples}")
print(f"Label: {label}")
print(f"Pass@4 Score: {pass_score}")

#endregion

#region LLM AS JUDGE

# This is our rubric for the judge.
RUBRIC = """
Score 1.0 if the predicted animal is the same as the label.
Score 0.5 if the prediction is a different animal but from the same biological class (e.g., both are mammals).
Score 0.0 otherwise (e.g., a mammal and a reptile).
"""

# ... A mock function `llm_as_judge` is defined here to simulate an LLM's response ...
def llm_as_judge(pred: str, label: str, rubric: str) -> float:
    """Simulate an LLM judge based on the provided rubric."""
    same_class = {
        ("Lion", "Tiger"): 0.5,
        ("Tiger", "Lion"): 0.5,
        ("Snake", "Lizard"): 0.5,
        ("Lizard", "Snake"): 0.5,
    }
    if pred == label:
        return 1.0
    elif (pred, label) in same_class:
        return same_class[(pred, label)]
    else:
        return 0.0
      
# --- Test Case 1: Perfect Match ---
score1 = llm_as_judge(pred="Lion", label="Lion", rubric=RUBRIC)
print(f"--> Final Score: {score1}\n")

# --- Test Case 2: Same Class ---
score2 = llm_as_judge(pred="Tiger", label="Lion", rubric=RUBRIC)
print(f"--> Final Score: {score2}\n")

# --- Test Case 3: Different Class ---
score3 = llm_as_judge(pred="Snake", label="Lion", rubric=RUBRIC)
print(f"--> Final Score: {score3}\n")

#endregion