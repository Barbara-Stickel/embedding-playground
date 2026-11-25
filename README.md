# embedding-playground
Hands-on experiments to understand text embeddings, similarity search, kNN, and fine-tuning.

The project supports different datasets through a shared labeled-text format. The original expense dataset contains roughly 7,500 short, noisy descriptions assigned to categories. MAGE and HC3 provide another task: distinguishing human-written from machine-generated text. The goal is to progressively explore different ways to classify new examples while understanding the vector operations involved.

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

`src/main.py` contains the entry point and experiment helpers. Its `main()` function
reads the configuration and starts the experiments; the helpers coordinate dataset
loading, embedding, classification, and evaluation for each seed.
Metric calculations live in `src/evaluation.py`; all printing functions live in
`src/reporting.py`.

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

Progress bars show completed work, processing speed, elapsed time, and estimated remaining time when loading cached vectors, calculating embeddings, and classifying test records. Embedding progress updates after each batch finishes.

The default configuration uses
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
The first run downloads and caches the model from Hugging Face, so it requires
internet access.

When using Nomic, every input must include a task instruction prefix. Set
`prefix: classification` to apply `classification:` to both the
training and test inputs. The stored `TextRecord.text` is not changed; the
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

## Choosing a dataset

`config.yaml` keeps expenses as the default. Presets are available for all three datasets:

```bash
embedding-playground --config configs/expenses.yaml
embedding-playground --config configs/mage.yaml
embedding-playground --config configs/hc3.yaml
```

The same option works with `python src/main.py --config configs/hc3.yaml`.
Start from one of these complete YAML presets. Required experiment settings are read
directly from the configuration, without compatibility handling for the old layout.

Dataset-specific options live under `dataset`:

```yaml
dataset:
  name: hc3
  subset: all
  max_questions: 2000
  test_size: 0.2
  seed: 42
  balance_train: true
  stratify_source: true
```

- **Expenses:** keeps the existing CSV, merchant/price inputs, primary or detailed categories, miscellaneous filtering, and random or chronological splits.
- **[MAGE](https://huggingface.co/datasets/jiumei/MAGE):** reads CSVs and preserves their supplied train, validation, and test assignments. Labels are normalized to `human` (original label `1`) and `machine` (original label `0`). The validation records are retained for future model selection; the current pipeline embeds training and test records and evaluates the test split.
- **[HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3):** uses answers as input text, labeled `human` or `machine`. Questions are retained only in metadata. Duplicate questions with the same text (ignoring surrounding whitespace) stay together even if their source IDs differ. `max_questions` limits the question groups used for the experiment. Available subsets are `all`, `finance`, `medicine`, `open_qa`, `reddit_eli5`, and `wiki_csai`.

The HC3 preset uses source-stratified question splitting (`stratify_source: true`) and balanced training (`balance_train: true`). For each training question, it randomly keeps `min(human_count, machine_count)` non-empty answers from each category using the configured seed. Thus a question with three human answers and one machine answer contributes one of each; duplicate question rows are balanced together. Training is 50/50 within each question and source. Surplus training answers remain unused, and a training question without both answer types contributes no records. All non-empty answers from test questions are retained; surplus training answers never move into testing. The loader prints training counts and the number of unused answers. Set `balance_train: false` to reproduce the selection of all training answers.

The question split preserves proportions of the sample's sources when each source has enough question groups and both splits can contain all sources. Small samples fall back to a random question split with a message. `test_size` is a fraction of question groups, so the final answer counts need not follow the same fraction after training is balanced.

MAGE and HC3 default to downloading data files through the [Hugging Face cache](https://huggingface.co/docs/huggingface_hub/guides/download). They read the source CSV/JSONL files directly, without running dataset-specific scripts. To use local downloads, set HC3's `file` or MAGE's `files` mapping as shown in the presets. Relative data paths resolve against the project root. Local `data/mage/` and `data/hc3/` folders are ignored by Git.

The sample limits reduce embedding and classification work; complete source files are still downloaded and read before sampling. Set limits to `null` to use all records. Text casing is preserved for MAGE and HC3. Long answers still use the existing tokenizer truncation behavior.

Each run ends with a confusion matrix of counts for the complete test split: rows are actual categories and columns are predicted categories. Correct predictions are on the green diagonal. The matrix includes every category in the dataset, including categories with zero test predictions, and grows to support more than two categories.

Each YAML file explicitly sets `evaluation.max_incorrect_predictions`. The table title
reports the total number of mistakes; accuracy always uses the complete test split.
Use `null` to display all mistakes, `0` to hide them, or an integer to limit them:

```yaml
evaluation:
  seeds: [42]
  max_incorrect_predictions: 10 # null shows all; 0 hides the table
```

`data_processing.load_dataset` dispatches to the loader selected by `dataset.name`. Each loader returns a `TextDataset` of `TextRecord` objects with `text`, `label`, optional embeddings and predictions, and dataset-specific `metadata`. Add a new loader and register it in `data_processing/loader.py` to support another labeled-text source.

## Measuring selection variability

Every run reports ordinary accuracy and [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html), the mean recall across categories in the test split. Balanced accuracy gives human and machine performance equal weight even though test data retains its original class imbalance.

To measure how sensitive results are to question splits and training-answer selection, run several seeds with the same model and classifier settings:

```yaml
evaluation:
  seeds: [42, 43, 44, 45, 46]
  max_incorrect_predictions: 10
```

Each distinct seed overrides `dataset.seed` for that run, changing both the question selection/split and balanced training-answer selection. Use `seeds: [42]` for one run; `evaluation.seeds` is required. The summary shows accuracy, balanced accuracy, and recall for each category per seed, followed by their means and sample standard deviations. Keep the same seed list when comparing classifiers.

This is repeated evaluation on randomly shuffled group splits, using sampling without replacement. Classical [bootstrapping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) resamples with replacement; it is a different method for estimating uncertainty. These repeated splits can overlap across runs, so the reported spread is descriptive variability rather than a confidence interval. Repeating runs measures and averages the effect of selection choices; it does not remove bias or combine classifiers into an ensemble.

Embedding vectors are reused from the cache. A newly selected answer is computed once if it has not previously been cached. Each seed still performs classification, so a list of five seeds does more classifier work than a single run. Detailed mistakes and the final confusion matrix belong to the final seed; the matrix title identifies that seed. The aggregate table covers all runs.

## Reusing embeddings

Embeddings are cached by default under `data/embeddings/`, independently of the dataset's train/test split and the classifier. The first run calculates and saves vectors; later runs look up each input text and calculate only vectors that are missing. When all inputs are cached, the embedding model and tokenizer are not loaded. Duplicate input texts are calculated once.

```yaml
embedding_cache:
  enabled: true
  directory: data/embeddings
```

Set `enabled: false` to calculate embeddings without reading or writing the cache. Relative cache paths resolve against the project root. Cache files are ignored by Git.

Each model configuration has a separate SQLite file containing the model identity and NumPy vectors. The cache identity includes the resolved model ID/path, model revision, task prefix, pooling/normalization version, and `model.cache_version`. For local checkpoints, the model directory's contents are fingerprinted so changing weights or tokenizer files invalidates older vectors. Changing test size, split seed, category labels, classifier, batch size, or device reuses existing vectors. Changing input text creates a new entry.

For a new model, change `model.name` or `model.path`; its embeddings are stored separately. You can return to an earlier model and reuse its cache. To use a particular Hugging Face model version, set `model.revision` to its commit hash or tag. The default revision is `main`; if you intentionally update that model version, bump `cache_version` to start a fresh cache:

```yaml
model:
  name: mini
  cache_version: "2"
```

Completed batches are committed to disk immediately, so an interrupted run can reuse them on its next attempt. Existing runs that predate caching need one initial embedding pass to populate the cache. Cached vectors preserve their values and dtype; labels, predictions, and split assignments are not stored with them.

`embedding_cache.py` handles storage and model/input identities; `embedding_eval.py` restores cached vectors and sends the remaining records through one embedding path. Its batch loop saves completed vectors when a cache is enabled. Increment `EMBEDDING_PIPELINE_VERSION` in `embedding_core.py` whenever the tokenization, pooling, or normalization implementation changes.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use local fixtures and mocked models/downloads, so they do not require model or dataset downloads.

## Fine-tuning a pair or batch

`embedding_training.train` performs one optimizer step. Pass matching text
pairs and target cosine similarities: `1.0` pulls a pair together, while `0.0`
makes it orthogonal (use `-1.0` to push it toward opposite directions).

```python
import torch

from embedding_core import create_model_config, load_model
from embedding_training import train

config = create_model_config("mini")
tokenizer, model, device = load_model(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

loss = train(
    ["coffee shop", "coffee beans"],
    ["local cafe", "hardware store"],
    [1.0, 0.0],
    tokenizer,
    model,
    optimizer,
    task_prefix=config.prefix,
    device=device,
)
```

Call `train` for each batch and epoch. It switches the model to training mode,
clears old gradients, computes normalized mean-pooled embeddings, backpropagates
the mean-squared cosine-similarity loss, updates the model, and returns the loss
as a `float`. Call `model.save_pretrained(...)` and
`tokenizer.save_pretrained(...)` when you want to keep the fine-tuned model.
