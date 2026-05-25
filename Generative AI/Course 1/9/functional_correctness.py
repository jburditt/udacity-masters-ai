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