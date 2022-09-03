# Duplicate Question Identification

A deep learning system for identifying semantically similar questions using Siamese LSTM networks with triplet loss and hard negative mining.

## 🎯 Problem Statement

Question-answering platforms like Quora often have multiple versions of the same question posted by different users. This creates redundancy and fragments answers across similar questions. This project tackles the challenge of automatically identifying duplicate questions to improve content organization and user experience.

## 🏗️ Architecture

The system uses a **Siamese Neural Network** architecture with shared weights to process question pairs:

```
Question 1 ──→ [Embedding] ──→ [LSTM] ──→ [Mean Pool] ──→ [L2 Norm] ──→ v1
                                                                           ├──→ Cosine Similarity
Question 2 ──→ [Embedding] ──→ [LSTM] ──→ [Mean Pool] ──→ [L2 Norm] ──→ v2
```

**Key Components:**
- **Embedding Layer**: Maps tokens to 128-dimensional dense vectors
- **LSTM Layer**: Captures sequential dependencies in questions
- **Mean Pooling**: Aggregates variable-length sequences into fixed-size representations
- **L2 Normalization**: Normalizes vectors for cosine similarity comparison

**Training Strategy:**
- **Triplet Loss** with hard negative mining
- Combines closest negative and mean negative for robust training
- Adam optimizer with learning rate warmup and inverse square root decay

## 📊 Performance

- **Accuracy**: ~69% on test set
- **Dataset**: Quora Question Pairs (~400K pairs)
- **Vocabulary Size**: 36,268 unique tokens
- **Training Samples**: 89,188 duplicate pairs

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/duplicate-question-identification.git
cd duplicate-question-identification

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## 💻 Usage

### Quick Start

```python
from src.model import create_siamese_model, create_triplet_loss
from src.evaluate import predict
from src.utils import data_generator
import pickle

# Load vocabulary (assumes you have vocab.pkl)
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load trained model
model = create_siamese_model(vocab_size=len(vocab), d_model=128)
model.init_from_file('models/model.pkl.gz')

# Predict on new question pairs
question1 = "How do I learn Python programming?"
question2 = "What's the best way to learn Python?"

is_duplicate, similarity = predict(
    question1, question2,
    threshold=0.7,
    model=model,
    vocab=vocab,
    data_generator=data_generator,
    verbose=True
)

print(f"Duplicate: {is_duplicate}, Similarity: {similarity:.4f}")
```

### Training from Scratch

```python
from src.data_preprocessing import load_data, prepare_data
from src.model import create_siamese_model, create_triplet_loss
from src.train import train_model
from src.utils import data_generator, set_random_seed

# Set random seed for reproducibility
set_random_seed(42)

# Load and prepare data
data_train, data_test = load_data('data/questions.csv')
vocab, train_Q1, train_Q2, val_Q1, val_Q2, Q1_test, Q2_test, y_test = prepare_data(
    data_train, data_test
)

# Create data generators
batch_size = 256
train_gen = data_generator(train_Q1, train_Q2, batch_size, vocab['<PAD>'])
val_gen = data_generator(val_Q1, val_Q2, batch_size, vocab['<PAD>'])

# Train model
training_loop = train_model(
    model_fn=lambda: create_siamese_model(vocab_size=len(vocab)),
    loss_fn=lambda: create_triplet_loss(margin=0.25),
    train_generator=train_gen,
    val_generator=val_gen,
    output_dir='models/'
)

# Run training
training_loop.run(n_steps=10000)
```

### Evaluation

```python
from src.evaluate import classify

# Evaluate on test set
accuracy = classify(
    Q1_test, Q2_test, y_test,
    threshold=0.7,
    model=model,
    vocab=vocab,
    data_generator=data_generator,
    batch_size=512
)

print(f"Test Accuracy: {accuracy:.2%}")
```

## 📁 Project Structure

```
duplicate-question-identification/
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_preprocessing.py      # Data loading and preparation
│   ├── model.py                   # Siamese network architecture
│   ├── train.py                   # Training utilities
│   ├── evaluate.py                # Evaluation and prediction
│   └── utils.py                   # Helper functions
├── examples/
│   └── demo.py                    # Usage demonstration
├── notebooks/
│   └── Duplicate_Question_Identification.ipynb  # Exploratory analysis
├── models/                        # Saved model weights
└── data/                          # Dataset (not included)
```

## 🔬 Technical Details

### Loss Function

The triplet loss encourages similar questions to have close embeddings while pushing dissimilar ones apart:

```
L1 = max(0, margin - cos(Q1, Q2_dup) + closest_negative)
L2 = max(0, margin - cos(Q1, Q2_dup) + mean_negative)
L = mean(L1 + L2)
```

### Hard Negative Mining

Instead of using random negative examples, the model identifies the "hardest" negatives (most confusing non-duplicates) within each batch to improve learning efficiency.

### Data Strategy

- Training uses only duplicate pairs; negatives are implicitly generated from batch structure
- For batch size N, each question Q1[i] has 1 positive (Q2[i]) and N-1 negatives (Q2[j] where j≠i)

## 🎓 Key Learnings

1. **Siamese Networks**: Effective for learning similarity functions with shared representations
2. **Triplet Loss**: Better than binary classification for ranking and similarity tasks
3. **Hard Negative Mining**: Significantly improves convergence and final performance
4. **Batch Structure**: Clever data organization can generate multiple training signals from a single batch

## 🔮 Future Improvements

- [ ] Implement attention mechanisms for better sequence modeling
- [ ] Experiment with transformer-based encoders (BERT, RoBERTa)
- [ ] Add data augmentation (paraphrasing, back-translation)
- [ ] Implement dynamic threshold selection
- [ ] Deploy as REST API with FastAPI
- [ ] Add visualization of embeddings (t-SNE, UMAP)
- [ ] Experiment with different similarity metrics

## 📝 License

MIT License - feel free to use this project for learning or commercial purposes.

## 🙏 Acknowledgments

- Dataset: [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
- Framework: [Google Trax](https://github.com/google/trax)

---

**Author**: Gaurav Misra  
**Contact**: [Your Email/LinkedIn]  
**GitHub**: [Your GitHub Profile]

