# Define a prediction and a reference text
pred = "the quick brown fox"
label = "the fox is quick and brown"

# Load the ROUGE metric from the 'evaluate' library
rouge = load("rouge")

# Compute the scores
results = rouge.compute(predictions=[pred], references=[label])

print(f"ROUGE-1 Score: {results['rouge1']:.4f}")
print(f"ROUGE-L Score: {results['rougeL']:.4f}")