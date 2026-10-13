# 24CA31A4 - Natural Language Processing & Large Language Models Lab

This repository contains the complete implementation of the **NLP & LLM Laboratory (Course Code: 24CA31A4)**. Each experiment covers foundational to advanced techniques in text preprocessing, statistical text representation, distributed neural embeddings, deep sequence modeling, and transformer-based Named Entity Recognition (NER), executed on custom domain corpora.

---

## 📚 Lab Curriculum & Notebook Index

| Lab No. | Notebook | Topic | Techniques & Models | Domain Corpus |
| :---: | :--- | :--- | :--- | :--- |
| **Lab 1** | [`LAB-1.ipynb`](./LAB-1.ipynb) | **Text Cleaning & Normalization** | Regex, BeautifulSoup, Case folding, Noise elimination | AI/ML announcements & E-commerce hardware reviews |
| **Lab 2** | [`LAB-2.ipynb`](./LAB-2.ipynb) | **NLP Preprocessing Pipeline** | Tokenization, Stop-word removal, Stemming (Porter), Lemmatization (WordNet), POS tagging, NER | Turing research, Autonomous tech, Aerospace & Medicine |
| **Lab 3** | [`LAB-3.ipynb`](./LAB-3.ipynb) | **Text Representation Techniques** | One-Hot Encoding, Bag-of-Words (BoW), CountVectorizer, Word Cloud | Deep learning & NLP chatbot representations |
| **Lab 4** | [`LAB-4.ipynb`](./LAB-4.ipynb) | **TF-IDF from Mathematical Principles** | TF Matrix, Document Frequency (DF), Smoothed IDF, Heatmap visualization | Cloud infrastructure, Scalable storage, Quantum computing |
| **Lab 5** | [`LAB-5.ipynb`](./LAB-5.ipynb) | **Text Classification (NB & SVM)** | Multinomial Naïve Bayes (Spam detection), LinearSVC (3-class Sentiment), Multi-topic classification | Cybersecurity phishing alerts, Hardware audio reviews, Multi-genre news |
| **Lab 6** | [`LAB-6.ipynb`](./LAB-6.ipynb) | **Word Embeddings with Word2Vec** | Word2Vec (Skip-gram/CBOW), Vector averaging, Sentence Cosine Similarity | Autonomous planetary robotics & Marine biology |
| **Lab 7** | [`LAB-7.ipynb`](./LAB-7.ipynb) | **GloVe Pretrained Word Embeddings** | GloVe (Global Vectors 100d), Vector averaging, Semantic similarity mapping | Speech recognition & Neural voice processing |
| **Lab 8** | [`LAB-8.ipynb`](./LAB-8.ipynb) | **Word2Vec (CBOW) Semantic Analysis** | Continuous Bag-of-Words (CBOW), Vector extraction, Top-K nearest neighbors | Cyber security infrastructure, Network defense, Cyberpunk/Fantasy |
| **Lab 9** | [`lab-9.ipynb`](./lab-9.ipynb) | **Sequential Sentiment Analysis (LSTM)** | Sequence tokenization, Post-padding, Embedding layer, LSTM recurrent cell, Sigmoid output | Hospitality, Hotel suites & Guest dining reviews |
| **Lab 10** | [`Lab-10.ipynb`](./Lab-10.ipynb) | **Pretrained BERT Transformer NER** | Hugging Face Transformers (`dslim/bert-base-NER`), Subword aggregation | Global technology leaders, Space exploration (ISRO, NASA), Cloud datacenters |

---

## 🛠️ Environment & Prerequisites

Install required dependencies via `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn wordcloud gensim spacy tensorflow transformers torch
```

Download required NLTK corpora and spaCy models:

```python
import nltk
nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'])

# In terminal
python -m spacy download en_core_web_sm
```

---

## 🚀 Running the Notebooks

Launch Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook
```

Navigate to any notebook (`LAB-1.ipynb` through `Lab-10.ipynb`) to inspect the code cells, markdown walkthroughs, and formatted output visualizations.
