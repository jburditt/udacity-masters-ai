# This is our rubric for the judge.
RUBRIC = """
Score 1.0 if the predicted animal is the same as the label.
Score 0.5 if the prediction is a different animal but from the same biological class (e.g., both are mammals).
Score 0.0 otherwise (e.g., a mammal and a reptile).
"""

# ... A mock function `llm_as_judge` is defined here to simulate an LLM's response ...

# --- Test Case 1: Perfect Match ---
score1 = llm_as_judge(pred="Lion", label="Lion", rubric=RUBRIC)
print(f"--> Final Score: {score1}\n")

# --- Test Case 2: Same Class ---
score2 = llm_as_judge(pred="Tiger", label="Lion", rubric=RUBRIC)
print(f"--> Final Score: {score2}\n")

# --- Test Case 3: Different Class ---
score3 = llm_as_judge(pred="Snake", label="Lion", rubric=RUBRIC)
print(f"--> Final Score: {score3}\n")