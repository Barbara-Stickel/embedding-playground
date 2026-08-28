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

## Running the code

### Set-up


```bash
cd /Users/barbaraadrien/Code/embedding-playground

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

### Run it

```bash
embedding-playground
```
or 

```bash
python src/main.py
```

The default configuration uses
[`nomic-ai/nomic-embed-text-v1`](https://huggingface.co/nomic-ai/nomic-embed-text-v1).
The first run downloads and caches the model from Hugging Face, so it requires
internet access.

Nomic requires every input to include a task instruction prefix. This project
classifies expenses, so `config.yaml` applies `classification:` to both the
training and test inputs. The stored `ExpenseRecord.text` is not changed; the
prefix is added only when the text is sent to the embedding model.

The model setup is configurable in `config.yaml`:

```yaml
model:
  name: nomic
  batch_size: 32
  device: auto
  prefix: classification
  trust_remote_code: true
```

With `device: auto`, PyTorch uses the Apple GPU through MPS when it is
available and falls back to the CPU otherwise. Set it explicitly to `mps` or
`cpu` to override that behavior.

To return to the original model, the compact `model: mini` setting is still
supported. You can also set `path` to a Hugging Face model ID instead of using
one of the `name` aliases.
