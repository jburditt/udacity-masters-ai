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