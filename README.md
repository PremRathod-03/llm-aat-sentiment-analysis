# Sentiment Analysis of Movie Reviews — 24AM7PCLLM AAT

Compares 7 approaches on the Pang & Lee (2005) sentence polarity dataset (10,662 sentences):
VADER lexicon, Multinomial Naive Bayes, Logistic Regression and Linear SVM on TF-IDF, static word
vectors + LR, frozen RoBERTa embeddings + LR, and an ensemble (TF-IDF LR + RoBERTa LR).

| Model | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Ensemble (TF-IDF LR + RoBERTa LR) | 0.808 | 0.806 | 0.883 |
| Linear SVM (TF-IDF) | 0.779 | 0.782 | 0.857 |
| Logistic Regression (TF-IDF) | 0.775 | 0.779 | 0.858 |
| Multinomial Naive Bayes | 0.774 | 0.772 | 0.846 |
| RoBERTa embeddings + LR | 0.753 | 0.746 | 0.835 |
| Static word vectors + LR | 0.696 | 0.691 | 0.772 |
| VADER (no training) | 0.642 | 0.660 | 0.690 |

## Deliverables
- `deliverables/LLM_AAT_Report_Sentiment_Analysis.docx` (+ `.pdf`) – AAT report
- `deliverables/LLM_AAT_Presentation_Sentiment_Analysis.pptx` – presentation

## Run
```bash
pip install scikit-learn nltk spacy scipy matplotlib torch
python -m spacy download en_core_web_md
python -m spacy download en_core_web_trf
python -c "import nltk; nltk.download('vader_lexicon')"

python predict.py "a masterpiece of quiet, heartbreaking performances"   # ensemble (needs results/roberta_features.npy)
python predict.py --fast "boring, predictable and far too long"         # TF-IDF only
python run_experiments.py      # all experiments -> results/results.json (RoBERTa encoding ~9 min on CPU, cached)
python posthoc_negation.py     # post-hoc negation-marking analysis
python make_figures.py         # figures/
```
`build_report.py` / `build_slides.py` regenerate the report and slides from the templates.
