# embedding-playground
Hands-on experiments to understand text embeddings, similarity search, kNN, and fine-tuning.

The dataset contains roughly 7,500 short, noisy text descriptions that have already been assigned to categories. The goal is to progressively explore different ways to classify new examples while understanding the vector operations involved.

## Step 1 - Pretrained embeddings

Use an existing embedding model to convert each historical example into a vector.

```text
text
 ↓
embedding model
 ↓
vector
```

Store each vector together with its known category.

For a new example:

1. Create its embedding.
2. Compare it with the historical embeddings using cosine similarity.
3. Find the closest examples.
4. Use their categories to predict the new category.

Initially, this can be implemented directly with NumPy and k-nearest neighbors. With only ~7,500 examples, a vector database is not necessary.

## Step 2 — Simple supervised classifier

Use the labeled dataset to train a traditional text classifier.

For example:

```text
text
 ↓
TF-IDF / character n-grams
 ↓
logistic regression
 ↓
category
```

This gives us a useful baseline to compare against the embedding approach.

## Step 3 — Fine-tuned embeddings

Start from a small pretrained embedding model and fine-tune it using the existing category labels.

The goal is to teach the model that examples belonging to the same category should have similar embeddings, while examples from different categories should be farther apart.

```text
generic embedding model
 ↓
fine-tuning on labeled examples
 ↓
task-specific embedding model
 ↓
k-nearest neighbors
```

This lets us compare:

* generic embeddings + kNN
* fine-tuned embeddings + kNN
* a traditional supervised classifier

## Goal

The main purpose of the project is educational: understand how text is converted into vectors, how vector similarity works, how kNN uses those vectors, and how training can change the geometry of an embedding space.

The implementation should stay small enough that the important operations remain visible and understandable.
