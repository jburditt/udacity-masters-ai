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