"""Fills the professor's LLM_AAT_Report_Template.docx with the project report."""
import copy
import json
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

TEMPLATE, OUT = sys.argv[1], sys.argv[2]
R = json.load(open("results/results.json"))
E1 = R["E1_models"]
TNR = "Times New Roman"
TOPIC = "Sentiment Analysis of Movie Reviews: Lexicon, Classical ML and Transformer Embeddings"

doc = Document(TEMPLATE)
body = doc.element.body
sectPr = body[-1]


def pct(x):
    return f"{100 * x:.1f}%"


def f3(x):
    return f"{x:.3f}"


# ---------- cover: topic ----------
for p in doc.paragraphs:
    if p.text.strip().startswith("Topic:"):
        runs = p.runs
        for r in runs:
            if "_" in r.text:
                r.text = r.text.replace("_", "").rstrip() + " " + TOPIC + " "
        if "_" in p.text:  # fallback
            for r in runs[1:]:
                r.text = ""
            runs[0].text = "Topic: " + TOPIC
        break


# ---------- helpers ----------
def _font(run, size=12, bold=False, italic=False, color=None):
    run.font.name = TNR
    run._element.rPr.rFonts.set(qn("w:eastAsia"), TNR)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _move(p):
    body.remove(p._p)
    sectPr.addprevious(p._p)
    return p


def para(text="", size=12, bold=False, italic=False, align="justify", space_after=6, indent=None,
         keep_next=False):
    p = _move(doc.add_paragraph())
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.15
    if keep_next:
        pf.keep_with_next = True
    pf.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                    "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    if indent is not None:
        pf.left_indent = Inches(indent)
    if text:
        # **bold** inline markup
        parts = text.split("**")
        for i, part in enumerate(parts):
            if part:
                _font(p.add_run(part), size, bold or i % 2 == 1, italic)
    return p


def bullets(items, size=12):
    for it in items:
        p = para("", size=size, space_after=3, indent=0.3)
        p.paragraph_format.first_line_indent = Inches(-0.2)
        parts = ("•  " + it).split("**")
        for i, part in enumerate(parts):
            if part:
                _font(p.add_run(part), size, i % 2 == 1)


def h2(text):
    p = _move(doc.add_paragraph(style="Heading 2"))
    r = p.add_run(text)
    r.style = doc.styles["Strong"]
    r.font.name = TNR
    r.bold = True
    p.paragraph_format.keep_with_next = True
    return p


def h3(text):
    p = para(text, size=12, bold=True, align="left", space_after=4, keep_next=True)
    p.paragraph_format.space_before = Pt(8)
    return p


def page_break():
    p = _move(doc.add_paragraph())
    p.add_run().add_break(WD_BREAK.PAGE)


def figure(path, caption, width=6.0):
    p = para(align="center", space_after=2, keep_next=True)
    p.add_run().add_picture(path, width=Inches(width))
    para(caption, size=10, italic=True, align="center", space_after=10)


def _borders(tbl):
    tblPr = tbl._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for e in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        el = OxmlElement(f"w:{e}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), "808080")
        b.append(el)
    tblPr.append(b)


def table(header, rows, caption=None, widths=None, size=10):
    if caption:
        para(caption, size=10, bold=True, align="center", space_after=3, keep_next=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    _move_tbl = t._tbl
    body.remove(_move_tbl); sectPr.addprevious(_move_tbl)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _borders(t)
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        _font(c.paragraphs[0].add_run(h), size, True)
        sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "DCE9F8")
        c._tc.get_or_add_tcPr().append(sh)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.rows[i + 1].cells[j]
            c.text = ""
            parts = str(v).split("**")
            for k, part in enumerate(parts):
                if part:
                    _font(c.paragraphs[0].add_run(part), size, k % 2 == 1)
    if widths:
        t.autofit = False
        tblPr = t._tbl.tblPr
        for el in tblPr.findall(qn("w:tblW")):
            tblPr.remove(el)
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(int(sum(widths) * 1440))); tw.set(qn("w:type"), "dxa")
        tblPr.append(tw)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        grid = t._tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(widths[j] * 1440)))
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Inches(w)
    para("", space_after=4)
    return t



# ===================================== BODY =====================================
D = R["data"]
N = ["VADER (lexicon, rule-based)", "Multinomial Naive Bayes (BoW)", "Logistic Regression (TF-IDF)",
     "Linear SVM (TF-IDF)", "Static word vectors + LR", "RoBERTa embeddings (frozen) + LR",
     "Ensemble (TF-IDF LR + RoBERTa LR)"]
L = ["VADER (lexicon)", "Multinomial Naive Bayes", "Logistic Regression (TF-IDF)", "Linear SVM (TF-IDF)",
     "Static word vectors + LR", "RoBERTa embeddings + LR", "Ensemble (TF-IDF LR + RoBERTa LR)"]
LR_, SVM, NB, VAD, WV, RB, ENS = "Logistic Regression (TF-IDF)", "Linear SVM (TF-IDF)", N[1], N[0], N[4], N[5], N[6]
E2, E3, E5, E6 = R["E2_preprocessing"], R["E3_learning_curve"], R["E5_slices"], R["E6_negation_posthoc"]
MC = R["E4_mcnemar"]

page_break()
h2("1. Introduction")
h3("1.1 Background of Natural Language Processing")
para("Natural Language Processing (NLP) enables computers to analyse and generate human language. The "
     "field moved from hand-written rules and lexicons, through statistical machine learning on "
     "bag-of-words features (Naive Bayes, Logistic Regression, Support Vector Machines), to neural "
     "representations: static word embeddings such as word2vec and GloVe, and contextual embeddings "
     "from Transformer models such as BERT and RoBERTa, which form the basis of today's Large Language "
     "Models (LLMs). Each generation trades interpretability and cost for accuracy, and choosing the "
     "right one for a task requires measuring them side by side.")
h3("1.2 Importance of the Selected NLP Topic")
para("Sentiment analysis, the task of deciding whether a text expresses a positive or negative opinion, "
     "is one of the most widely deployed NLP applications. Companies track customer reviews and social "
     "media to measure satisfaction, film studios and streaming platforms analyse audience reaction, and "
     "investors monitor news sentiment. Short opinion sentences are linguistically hard: meaning depends "
     "on negation ('not good'), contrast ('stunning visuals, but a dull story'), intensifiers and "
     "sarcasm. This makes sentiment analysis an ideal task for comparing lexicon-based, statistical and "
     "Transformer-based approaches covered in the 24AM7PCLLM syllabus, and for showing where each "
     "succeeds or fails. Building reliable opinion-mining tools for industry supports SDG 9.5 (upgrading "
     "technological capabilities and encouraging innovation).")
h3("1.3 Objective of the Study")
bullets([
    "Study the sentiment-analysis task and identify a real-time application and APIs that perform it.",
    "Apply Unit 1 preprocessing (cleaning, tokenisation, stopword removal, negation handling) and measure "
    "its effect with cross-validation.",
    "Implement and compare seven approaches: a rule-based lexicon (VADER), three classical classifiers on "
    "bag-of-words/TF-IDF features, static word embeddings, RoBERTa contextual embeddings and an ensemble.",
    "Evaluate with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, training time and "
    "inference speed, and test statistical significance (McNemar's test).",
    "Analyse linguistic patterns (negation, contrast, intensifiers, sentence length, sarcasm) and data "
    "efficiency (learning curves).",
    "Deliver a working command-line sentiment analyser that explains its decisions.",
])
h3("1.4 Scope of the Report")
para(f"The study uses the public Sentence Polarity dataset v1.0 of Pang and Lee (2005): "
     f"{D['n']:,} movie-review sentences labelled positive or negative. All models run on a CPU. Transformer "
     "embeddings are used frozen (feature extraction) rather than fine-tuned because no GPU was available; "
     "fine-tuning, multi-class (neutral) sentiment, aspect-based sentiment and Indian-language reviews are "
     "outside the scope and listed as future work.")

h2("2. Selected NLP Topic Overview")
h3("2.1 Description of the NLP Task (e.g., Text Classification, NER, MT, Chatbot)")
para("Sentiment analysis is a **binary text-classification** task: given a sentence x, predict a label "
     "y ∈ {negative, positive}. A classifier learns a function f(x) = P(y = positive | x) from labelled "
     "examples, and predicts positive when f(x) ≥ 0.5. Lexicon-based systems instead sum the prior "
     "polarity of known words and adjust it with rules for negation, intensifiers and punctuation.")
h3("2.2 Real-World Use Cases")
bullets([
    "**Product and service reviews:** e-commerce and food-delivery platforms flag negative reviews for "
    "follow-up and summarise satisfaction trends.",
    "**Brand and social-media monitoring:** companies track opinion about launches and campaigns in real "
    "time.",
    "**Film and entertainment:** review aggregators and studios measure critic and audience reaction "
    "(the domain of this project's dataset).",
    "**Customer support:** tickets and chats with strong negative sentiment are prioritised.",
    "**Finance:** sentiment of news headlines and reports is used as a market signal.",
])
h3("2.3 Relevance to the Syllabus and Industry")
table(["Syllabus unit", "Concept used in this project"], [
    ["Unit 1", "Regular expressions, normalisation, tokenisation (incl. RoBERTa's byte-level BPE), stopwords, "
               "negation handling; ambiguity (why NLP is hard)"],
    ["Unit 2", "Bag-of-words, TF-IDF weighting, n-grams, static word embeddings, contextual embeddings"],
    ["Unit 3", "N-gram features, pre-trained language model RoBERTa, Transformer self-attention"],
    ["Unit 4", "Case study of an NLP/LLM-powered application (opinion mining)"],
    ["Unit 5", "Deployment trade-offs (speed vs accuracy), responsible AI: bias and misclassification risk"],
], caption="Table 2.1: Mapping of the project to the 24AM7PCLLM syllabus", widths=[1.2, 5.0])
para("In industry, sentiment analysis is offered as managed APIs (Google Cloud Natural Language, Amazon "
     "Comprehend, Azure AI Language) and is a standard benchmark task for language models (the "
     "Stanford Sentiment Treebank in GLUE is derived from the same movie-review source used here).")
h3("2.4 Key Challenges in This NLP Task")
bullets([
    "**Negation:** 'not good' and 'not bad at all' reverse the polarity of the word that follows.",
    "**Contrast:** 'the visuals are stunning, but the story never comes alive' — the clause after 'but' "
    "usually decides the overall sentiment.",
    "**Sarcasm and irony:** 'oh great, another two hours of my life I will never get back' uses positive "
    "words to express a negative opinion.",
    "**Domain-specific vocabulary:** words like 'unpredictable' are positive for a plot but negative for a "
    "car; general-purpose lexicons miss film-specific usage.",
    "**Short context:** sentences average only about 21 words, so there is little evidence per example.",
])

h2("3. Real-Time Application / API Identified")
h3("3.1 Application or API Name")
para("**Movie-Review Sentiment Analyser**: a command-line application (predict.py) that classifies a "
     "review sentence as positive or negative, reports the probability, and lists the words that drove "
     "the decision. It uses the following APIs and libraries: **NLTK VADER** (an off-the-shelf "
     "rule-based sentiment API, used as the baseline), **scikit-learn** (TF-IDF, Naive Bayes, Logistic "
     "Regression, Linear SVM), **spaCy 3.8** (en_core_web_md word vectors) and the **RoBERTa-base** "
     "Transformer through spaCy's en_core_web_trf pipeline (PyTorch back-end). Cloud APIs such as Google "
     "Cloud Natural Language (analyzeSentiment) and Amazon Comprehend (DetectSentiment) provide the same "
     "function as a paid web service.")
h3("3.2 Purpose and Functionality")
bullets([
    "Classify a review sentence as positive or negative with a confidence score.",
    "Explain the decision by listing the words with the largest positive or negative contribution.",
    "Show the VADER score alongside for comparison with a lexicon baseline.",
    "Offer a fast mode (TF-IDF only, no deep-learning dependency) and an accurate mode (ensemble with "
    "RoBERTa).",
])
h3("3.3 Architecture or Platform on Which It Is Built")
para("The application is written in Python 3 and runs on CPU. At start-up it trains the TF-IDF + Logistic "
     "Regression model on the training split (about 2 seconds) and, in accurate mode, a Logistic "
     "Regression on cached RoBERTa sentence embeddings. Each input sentence is tokenised, vectorised by "
     "both feature extractors, scored by both classifiers and the two probabilities are averaged. The same "
     "functions can be exposed behind a REST endpoint for real-time use. The full pipeline is shown in "
     "Figure 4.1.")
h3("3.4 Input/Output Behavior")
table(["Stage", "Description", "Example"], [
    ["Input", "One review sentence in English", "\"a masterpiece of quiet, heartbreaking performances.\""],
    ["Processing", "Clean → tokenise → TF-IDF features + RoBERTa embedding → two Logistic Regression "
                   "classifiers → average probability", "TF-IDF LR 0.99, RoBERTa LR 1.00"],
    ["Output", "Label, P(positive), key words, VADER baseline score",
     "POSITIVE (0.99); key words: masterpiece (+), performances (+)"],
], caption="Table 3.1: Input-processing-output stages of the application", widths=[1.0, 2.8, 2.4])
h3("3.5 Limitations and Practical Considerations")
bullets([
    "Trained only on movie-review sentences from professional critics; casual or other-domain text (for "
    "example product reviews written informally) can be misclassified.",
    "Binary output: neutral or mixed sentences are forced into one class.",
    "RoBERTa embedding takes about 50 ms per sentence on CPU, roughly 2,000 times slower than TF-IDF.",
    "Sarcasm is not handled reliably by any of the models tested.",
])

h2("4. NLP Model / Architecture Study")
h3("4.1 Theoretical Foundations of the Model")
para("Three families of sentiment models are studied. **Lexicon-based** methods (VADER, Hutto and "
     "Gilbert, 2014) assign each word a human-rated valence and combine them with rules. **Statistical "
     "classifiers** learn word weights from labelled data using bag-of-words or TF-IDF vectors: Naive "
     "Bayes is a generative model that applies Bayes' rule under a word-independence assumption, while "
     "Logistic Regression and Support Vector Machines are discriminative linear models. **Embedding-based** "
     "methods represent a sentence as a dense vector — an average of static word vectors, or the "
     "contextual token representations produced by a pre-trained Transformer (RoBERTa, Liu et al., 2019) "
     "— and train a linear classifier on top. An **ensemble** averages the probabilities of two "
     "complementary models.")
h3("4.2 Algorithm(s) Used (e.g., Naive Bayes, LSTM, BERT, Transformers)")
table(["#", "Approach", "Features", "Classifier / rule"], [
    ["1", "VADER", "7,500-entry valence lexicon + 5 heuristics", "compound ≥ 0.05 → positive"],
    ["2", "Multinomial Naive Bayes", "Word counts (bag-of-words)", "Bayes' rule, Laplace smoothing α"],
    ["3", "Logistic Regression", "TF-IDF (sublinear tf)", "Sigmoid of weighted sum, L2 penalty C"],
    ["4", "Linear SVM", "TF-IDF (sublinear tf)", "Maximum-margin hyperplane, hinge loss, C"],
    ["5", "Static word vectors + LR", "Mean of 300-d spaCy vectors", "Logistic Regression"],
    ["6", "RoBERTa embeddings + LR", "Mean of 768-d RoBERTa last layer (frozen)", "Logistic Regression"],
    ["7", "Ensemble", "TF-IDF + RoBERTa", "Average of the two LR probabilities"],
], caption="Table 4.1: Approaches implemented and compared", widths=[0.3, 1.8, 2.3, 1.9])
h3("4.3 Architecture Diagram / Layer-Wise Design")
figure("figures/fig_architecture.png", "Figure 4.1: Architecture of the sentiment-analysis system: shared "
       "preprocessing, four feature extractors and their classifiers", width=6.3)
para("The system has six components: (1) **input** of a review sentence; (2) **preprocessing** "
     "(cleaning, lowercasing, tokenisation, optional negation marking); (3) **feature extractors** "
     "(bag-of-words/TF-IDF, static vectors, RoBERTa, VADER lexicon); (4) **classifiers** (Naive Bayes, "
     "Logistic Regression, SVM, VADER rule); (5) **ensemble** layer; and (6) **output** of label, "
     "probability and explanation. RoBERTa-base itself is a 12-layer Transformer encoder: each layer "
     "applies 12-head self-attention followed by a feed-forward network with residual connections and "
     "layer normalisation, producing a 768-dimensional vector for every sub-word token of the 50,265-token "
     "byte-level BPE vocabulary.")
h3("4.4 Data Flow in the Model")
bullets([
    "**Step 1 - Input:** a sentence such as 'it is not a good film .'",
    "**Step 2 - Cleaning:** lowercase, remove HTML remnants, collapse whitespace.",
    "**Step 3 - Tokenisation:** split into tokens (the corpus already separates punctuation and 'n't').",
    "**Step 4 - Feature extraction:** sparse TF-IDF vector over 19,023 unigrams, and/or a 768-d RoBERTa "
    "embedding (mean of sub-word states).",
    "**Step 5 - Classification:** each Logistic Regression outputs P(positive).",
    "**Step 6 - Ensemble and decision:** average the probabilities; positive if ≥ 0.5.",
    "**Step 7 - Explanation:** list the words whose weight × TF-IDF value contributed most.",
])
h3("4.5 Mathematical Formulation (if applicable)")
para("**TF-IDF:** w(t, d) = (1 + log tf(t, d)) × (ln((1 + N) / (1 + df(t))) + 1), then L2-normalised.")
para("**Multinomial Naive Bayes:** ŷ = argmax_c [ log P(c) + Σ_t count(t, d) · log P(t | c) ], with "
     "P(t | c) = (count(t, c) + α) / (Σ_t' count(t', c) + α|V|).")
para("**Logistic Regression:** P(y = 1 | x) = σ(wᵀx + b) = 1 / (1 + e^−(wᵀx + b)); trained by minimising "
     "cross-entropy + ‖w‖² / (2C).")
para("**Linear SVM:** minimise ½‖w‖² + C Σ_i max(0, 1 − y_i(wᵀx_i + b)).")
para("**VADER compound score:** normalised sum of rule-adjusted valences s, compound = s / √(s² + 15).")
para("**Self-attention (RoBERTa):** Attention(Q, K, V) = softmax(QKᵀ / √d_k) V; sentence embedding "
     "e = (1/n) Σ_i h_i over the n sub-word states h_i of the last layer.")
para("**Metrics:** Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = 2PR / (P + R); "
     "Accuracy = (TP + TN) / N; ROC-AUC = area under the true-positive vs false-positive rate curve.")
h3("4.6 Why This Model Suits the Selected NLP Task")
para("Short, opinionated sentences contain strong lexical cues ('dull', 'wonderful'), which linear models "
     "on TF-IDF features capture efficiently and interpretably. Contextual embeddings encode word order "
     "and context, which should help with negation and contrast, and they transfer knowledge from "
     "pre-training, which helps when labelled data are scarce. Combining the two in an ensemble uses the "
     "strengths of both, while the lexicon baseline shows what is possible without any training data.")

h2("5. Dataset and Preprocessing")
h3("5.1 Dataset Description (Source, Size, Structure)")
table(["Property", "Value"], [
    ["Name / source", "Sentence Polarity Dataset v1.0, Pang and Lee (2005); snippets from Rotten Tomatoes "
                      "reviews (public, via the Cornell movie-review data)"],
    ["Size", f"{D['n']:,} sentences: {D['n_pos']:,} positive, {D['n_neg']:,} negative (balanced)"],
    ["Structure", "Two text files (one sentence per line), already lowercased and tokenised"],
    ["Length", f"Average {D['avg_words']:.1f} words, median {D['median_words']:.0f}, maximum {D['max_words']}"],
    ["Vocabulary", f"{D['vocab_size']:,} distinct tokens"],
    ["Split", f"{D['n_train']:,} training (80%) / {D['n_test']:,} test (20%), stratified, seed 42"],
], caption="Table 5.1: Dataset summary", widths=[1.6, 4.6])
h3("5.2 Data Cleaning Steps")
bullets([
    "Read files as UTF-8, ignoring undecodable bytes from the original Latin-1 encoding.",
    "Removed empty lines and HTML remnants with regular expressions; collapsed repeated whitespace.",
    "Lowercased all text (already lowercase in the source) so that features are case-insensitive.",
    "Checked class balance (exactly 50/50), so accuracy is a meaningful headline metric.",
])
h3("5.3 Tokenization, Lemmatization, Stopwords Removal")
para("The corpus is pre-tokenised in Penn-Treebank style (punctuation and clitics such as 'n't' and ''s' "
     "are separate tokens), so whitespace tokenisation is used for the bag-of-words models; RoBERTa uses "
     "its own byte-level BPE sub-word tokeniser. Four preprocessing variants were compared with 5-fold "
     "cross-validation on the training set, each with its own tuned regularisation strength C:")
figure("figures/fig_preprocessing.png", "Figure 5.1: Preprocessing ablation (5-fold CV accuracy, mean ± std)",
       width=5.4)
sw = E2["+ Stopword removal"]["cv_acc_mean"]; uni = E2["Lowercase unigrams"]["cv_acc_mean"]
para(f"**Stopword removal hurts** ({pct(uni)} → {pct(sw)}) because standard stopword lists contain "
     "sentiment-bearing function words such as 'not', 'no', 'too' and 'very'; it was therefore not used. "
     "Lemmatisation was not applied for the same reason: inflections like 'bored/boring' carry different "
     "sentiment. Negation marking (prefixing NOT_ to every word after a negation until the next "
     f"punctuation, e.g. 'not a good film' → 'not NOT_a NOT_good NOT_film') reached {pct(E2['Uni+bi + negation marking']['cv_acc_mean'])}, "
     f"statistically tied with plain unigrams ({pct(uni)}, std ≈ 1.2 points). Following the protocol, the "
     "best cross-validation variant (lowercase unigrams) was selected for the classical models.")
h3("5.4 Train/Test Split Strategy")
para(f"A stratified 80/20 split (seed 42) gives {D['n_train']:,} training and {D['n_test']:,} test "
     "sentences with the same class balance. All hyperparameters (Naive Bayes α; C for Logistic "
     "Regression and SVM; C for the embedding classifiers) and the preprocessing variant were chosen by "
     "5-fold stratified cross-validation on the training set only. The test set was used once, for the "
     "final evaluation.")
h3("5.5 Feature Engineering or Embedding Approach (TF-IDF, Word2Vec, BERT Embeddings)")
bullets([
    f"**Bag-of-words counts** (Naive Bayes) and **TF-IDF** with sublinear term frequency (LR, SVM): "
    f"{R['n_features']:,} unigram features.",
    "**Static word embeddings:** mean of spaCy en_core_web_md 300-d vectors (trained on OSCAR, Wikipedia, "
    "OpenSubtitles and news), standardised.",
    "**Contextual embeddings:** mean of RoBERTa-base last-layer states (768-d), frozen, standardised; "
    "encoding all 10,662 sentences took about 9 minutes on CPU.",
    "**Lexicon features:** VADER valence scores (no training).",
])

h2("6. Implementation Methodology")
h3("6.1 Tools, Libraries, and Environment Used")
table(["Item", "Details"], [
    ["Language / OS", "Python 3.13 on Linux; CPU only (2 cores, 8 GB RAM), no GPU"],
    ["NLP libraries", "NLTK (VADER), spaCy 3.8 (en_core_web_md, en_core_web_trf / RoBERTa-base), PyTorch 2"],
    ["Machine learning", "scikit-learn (CountVectorizer, TfidfVectorizer, MultinomialNB, LogisticRegression, "
                         "LinearSVC, GridSearchCV, StratifiedKFold), SciPy (binomial test)"],
    ["Visualisation", "matplotlib"],
    ["Project files", "data/ (corpus), run_experiments.py, posthoc_negation.py, make_figures.py, "
                      "predict.py (application), results/ (metrics, cached embeddings), figures/"],
], caption="Table 6.1: Tools and environment", widths=[1.5, 4.7])
h3("6.2 Model Implementation Steps")
bullets([
    "Loaded and inspected the corpus; created the stratified train/test split.",
    "Ran the preprocessing ablation with cross-validated Logistic Regression and chose the feature set.",
    "Evaluated VADER directly on the test set (no training).",
    "Tuned and trained Naive Bayes, Logistic Regression and Linear SVM with 5-fold grid search.",
    "Extracted static and RoBERTa sentence embeddings for all sentences and trained tuned Logistic "
    "Regression classifiers on them.",
    "Built the probability-averaging ensemble; computed metrics, confusion matrices, learning curves, "
    "McNemar's test and accuracy on linguistic slices.",
    "Packaged the best models into the predict.py application.",
])
h3("6.3 API Integration (if used)")
para("VADER is called through NLTK's SentimentIntensityAnalyzer.polarity_scores(text), which returns "
     "negative, neutral, positive and compound scores. RoBERTa embeddings come from the spaCy pipeline "
     "API: nlp.pipe(texts) runs the Transformer and doc._.trf_data.last_hidden_layer_state exposes the "
     "last-layer token states, which are mean-pooled. The application in Section 3 wraps both behind a "
     "single analyse(text) function; the same function could call a cloud API such as Google Cloud "
     "Natural Language for comparison.")
h3("6.4 Pseudocode / Workflow Diagram")
para("procedure TRAIN():\n"
     "    X_train, X_test ← stratified_split(corpus, 0.8, seed = 42)\n"
     "    for each preprocessing variant v: cv_acc[v] ← 5-fold CV of TF-IDF + LR (tune C)\n"
     "    v* ← argmax cv_acc\n"
     "    for each model m in {NB, LR, SVM}: m ← GridSearchCV(features(v*), params(m)).fit(X_train)\n"
     "    E ← RoBERTa_mean_embeddings(all sentences)        # cached\n"
     "    rob ← GridSearchCV(StandardScaler + LR).fit(E[train])\n\n"
     "procedure PREDICT(sentence):\n"
     "    p1 ← LR.predict_proba(TFIDF(tokenise(clean(sentence))))\n"
     "    p2 ← rob.predict_proba(RoBERTa_mean_embedding(sentence))\n"
     "    p ← (p1 + p2) / 2\n"
     "    return ('positive' if p ≥ 0.5 else 'negative', p, top_contributing_words)", size=10,
     align="left", indent=0.3)
h3("6.5 Hyperparameters and Training Configuration")
bp = lambda n: ", ".join(f"{k.split('__')[-1]} = {v}" for k, v in E1[n]["best_params"].items())
table(["Model", "Search space", "Selected (5-fold CV)"], [
    ["Multinomial NB", "α ∈ {0.1, 0.3, 0.5, 1.0}", bp(NB)],
    ["Logistic Regression (TF-IDF)", "C ∈ {1, 4, 10, 30}", bp(LR_)],
    ["Linear SVM (TF-IDF)", "C ∈ {0.1, 0.3, 1, 3}", bp(SVM)],
    ["Static vectors + LR", "C ∈ {0.01 … 1}", bp(WV)],
    ["RoBERTa + LR", "C ∈ {0.01 … 1}", bp(RB)],
    ["VADER", "standard threshold", "compound ≥ 0.05"],
    ["Ensemble", "-", "equal-weight average of LR and RoBERTa-LR probabilities"],
], caption="Table 6.2: Hyperparameters", widths=[2.0, 2.0, 2.2])
para("No network weights were fine-tuned; RoBERTa was used as a frozen feature extractor.")

h2("7. Results and Evaluation")
h3("7.1 Performance Metrics (Accuracy, F1-Score, BLEU, Precision/Recall, etc.)")
rows = []
best = max(N, key=lambda n: E1[n]["accuracy"])
for n, l in zip(N, L):
    v = E1[n]
    c = (lambda s: f"**{s}**") if n == best else (lambda s: s)
    tr = "-" if v.get("train_s") in (None, 0.0) else f"{v['train_s']:.1f}"
    rows.append([c(l)] + [c(f3(v[m])) for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]] + [tr])
table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Tuning + train (s)"], rows,
      caption=f"Table 7.1: Results on the held-out test set (n = {D['n_test']:,}); best in bold",
      widths=[2.0, 0.7, 0.75, 0.65, 0.6, 0.7, 0.85], size=9.5)
para("Training time includes the 5-fold grid search. Embedding extraction time is not included (RoBERTa: "
     "about 9 minutes for the whole corpus on CPU).", size=10, italic=True)
h3("7.2 Visualization of Results (Confusion Matrix, Graphs)")
figure("figures/fig_models.png", "Figure 7.1: Accuracy, precision, recall and F1 of all models", width=6.0)
figure("figures/fig_confusion.png", "Figure 7.2: Confusion matrices of Logistic Regression (TF-IDF) and the "
       "ensemble", width=5.4)
figure("figures/fig_learning_curve.png", "Figure 7.3: Test accuracy as a function of training-set size",
       width=5.2)
figure("figures/fig_slices.png", "Figure 7.4: Accuracy on sentences with specific linguistic patterns",
       width=6.0)
h3("7.3 Qualitative and Quantitative Analysis")
para("**Quantitative observations:**", space_after=3)
e3 = {k: E3[k] for k in sorted(E3, key=float)}
first, last = e3[list(e3)[0]], e3[list(e3)[-1]]
bullets([
    f"**Observation 1:** The ensemble of TF-IDF Logistic Regression and RoBERTa Logistic Regression is the "
    f"best model: accuracy {f3(E1[ENS]['accuracy'])}, F1 {f3(E1[ENS]['f1'])}, ROC-AUC "
    f"{f3(E1[ENS]['roc_auc'])}, about {100 * (E1[ENS]['accuracy'] - E1[SVM]['accuracy']):.1f} points above the "
    "best single model. The two models make different mistakes, so averaging helps.",
    f"**Observation 2:** The classical linear models are strong and nearly identical: Linear SVM "
    f"{f3(E1[SVM]['accuracy'])}, Logistic Regression {f3(E1[LR_]['accuracy'])}, Naive Bayes "
    f"{f3(E1[NB]['accuracy'])}. McNemar's test between SVM and Logistic Regression ({MC['a_only_correct']} vs "
    f"{MC['b_only_correct']} discordant sentences, p = {MC['p_value']:.2f}) shows the difference is not "
    "statistically significant.",
    f"**Observation 3:** Frozen RoBERTa embeddings alone ({f3(E1[RB]['accuracy'])}) are below TF-IDF with the "
    "full training set, but the learning curve shows they are better with little data: with "
    f"{first['n']} training sentences RoBERTa reaches {f3(first['roberta_lr'])} vs {f3(first['tfidf_lr'])} for "
    "TF-IDF. The curves cross at about 4,000 sentences, after which TF-IDF keeps improving while the "
    "frozen embedding saturates.",
    f"**Observation 4:** Averaged static word vectors ({f3(E1[WV]['accuracy'])}) lose word identity and "
    "order; averaging dilutes the few sentiment words in a sentence.",
    f"**Observation 5:** VADER, with no training, reaches only {f3(E1[VAD]['accuracy'])}. "
    f"{pct(R['vader_neutral_fraction'])} of test sentences get a near-zero (neutral) score because "
    "film-review vocabulary ('engrossing', 'tedious', 'pretentious') is missing from or weak in its "
    "general-purpose lexicon.",
    f"**Observation 6:** Linguistic patterns matter: on sentences with negation, accuracy drops to "
    f"{f3(E5['Contains negation'][LR_])} for TF-IDF LR and {f3(E5['Contains negation'][VAD])} for VADER; on "
    f"contrast sentences ('but', 'yet') RoBERTa drops to {f3(E5[list(E5)[1]][RB])}. Short sentences are easiest "
    f"(ensemble {f3(E5[list(E5)[2]][ENS])}).",
    f"**Observation 7 (post-hoc, not used for model selection):** the negation-marked uni+bigram Logistic "
    f"Regression, tied in cross-validation, scores {f3(E6['accuracy'])} on the test set and "
    f"{f3(E6['negation_slice_acc'])} on negation sentences (vs {f3(E5['Contains negation'][LR_])}), and it "
    "correctly flips 'it is not a good film' to negative. Explicit negation handling is the most promising "
    "cheap improvement.",
])
para("**Qualitative analysis - demonstrated test cases:**", space_after=3)
TC = R["test_cases"]


def mark(p, e):
    return ("✓ " if (p >= 0.5) == bool(e) else "✗ ") + f"{p:.2f}"


rows = []
for i, t in enumerate(TC):
    vd = t["vader_compound"]
    rows.append([f"TC{i + 1}", t["text"], "Pos" if t["expected"] else "Neg",
                 mark(t["tfidf_lr_p_pos"], t["expected"]), mark(t["roberta_lr_p_pos"], t["expected"]),
                 mark(E6["test_cases"][t["text"]], t["expected"]),
                 ("✓ " if (vd >= 0.05) == bool(t["expected"]) else "✗ ") + f"{vd:+.2f}"])
table(["#", "Sentence", "Gold", "TF-IDF LR", "RoBERTa LR", "LR + negation", "VADER"], rows,
      caption="Table 7.2: Demonstrated test cases (P(positive) for classifiers, compound score for VADER; "
              "✓ correct, ✗ wrong)", widths=[0.45, 2.45, 0.5, 0.75, 0.8, 0.8, 0.65], size=8.5)
h3("7.4 Error Analysis")
err = R["errors"]
para("The most confident mistakes of the deployed TF-IDF Logistic Regression on the test set fall into "
     "four groups:")
bullets([
    f"**Contrast and concession:** \"{err[0]['text']}\" is labelled positive by the annotators but the model "
    f"gives P(positive) = {err[0]['p_pos']:.2f}, because 'depressing' and 'boring' dominate while 'never' and "
    "'but' are ignored by a bag-of-words model.",
    f"**Double negation / hedged praise:** \"{err[1]['text']}\" (gold positive, P = {err[1]['p_pos']:.2f}).",
    f"**Faint praise read as positive:** \"{err[5]['text']}\" (gold negative, P = {err[5]['p_pos']:.2f}) — "
    "'solid' is a strong positive feature, but the second clause makes the review negative.",
    "**Sarcasm and out-of-domain phrasing:** TC7 (sarcasm) is caught only by chance by the classifiers, and a "
    "casual sentence such as 'The acting was superb and the story kept me hooked' was misclassified by "
    "the demo application because function words ('was', 'me', 'kept') carry small negative weights "
    "learned from critic snippets.",
])
h3("7.5 Comparison Between Approaches (If multiple methods used)")
table(["Approach", "Strengths", "Weaknesses", "Verdict"], [
    ["VADER", "No training; fast; handles some negation and intensifiers",
     "Generic lexicon; many neutral scores; lowest accuracy", "Baseline only"],
    ["Naive Bayes", "Fastest to train; robust with small data", "Independence assumption", "Good baseline"],
    ["LR / Linear SVM (TF-IDF)", "Accurate, interpretable weights, ~0.02 ms per sentence",
     "Blind to word order; negation and contrast errors", "Best single model; deploy in fast mode"],
    ["Static vectors + LR", "Dense, small", "Averaging blurs sentiment words", "Not recommended"],
    ["RoBERTa + LR", "Context-aware; best with little data", "Slow on CPU; frozen features saturate",
     "Useful in ensemble"],
    ["Ensemble", "Highest accuracy, F1 and AUC", "Needs Transformer at inference", "**Best overall**"],
], caption="Table 7.3: Comparison of approaches", widths=[1.4, 1.9, 1.9, 1.1], size=9.5)

h2("8. Discussion")
h3("8.1 Interpretation of Results")
para(f"On a balanced sentence-level benchmark, simple linear models on TF-IDF features remain hard to beat "
     f"(about {pct(E1[SVM]['accuracy'])} accuracy), which is consistent with published results on this "
     "dataset for linear models. Using a large pre-trained Transformer as a frozen feature extractor does "
     "not automatically help; its value shows when labelled data are scarce and when its predictions are "
     f"combined with lexical models ({pct(E1[ENS]['accuracy'])}). Fine-tuning the Transformer end-to-end "
     "would be expected to perform best, but requires a GPU.")
h3("8.2 Strengths and Weaknesses of the Model")
table(["Strengths", "Weaknesses"], [
    ["Best model improves accuracy and AUC over every single model", "Errors on contrast, double negation and "
     "sarcasm remain"],
    ["Fast mode is interpretable and runs in microseconds", "Binary labels cannot express mixed opinions"],
    ["Rigorous protocol: CV tuning, single test use, significance test", "Domain-specific: trained only on "
     "critic snippets"],
    ["Works on CPU without fine-tuning", "RoBERTa features are frozen, not adapted to the task"],
], caption="Table 8.1: Strengths and weaknesses", widths=[3.1, 3.1])
h3("8.3 Challenges Encountered During Implementation")
bullets([
    "No GPU: RoBERTa had to be used as a frozen encoder, and embedding 10,662 sentences took about 9 "
    "minutes, so embeddings were cached to disk.",
    "A first preprocessing comparison used one fixed C for all variants, which unfairly penalised the "
    "larger bigram feature sets; it was corrected by tuning C per variant.",
    "Tokenisation of user input had to be made consistent with the corpus (splitting 'n't' and "
    "punctuation) for the application to work.",
    "VADER's default threshold leaves many sentences neutral; the binary mapping (neutral → negative) "
    "lowers its recall of negatives.",
])
h3("8.4 Impact of Dataset Quality and Preprocessing")
para("Because the dataset is balanced and clean, accuracy is a fair metric and differences between "
     "models are meaningful. Preprocessing choices had measurable effects: removing stopwords cost about "
     f"{100 * (uni - sw):.1f} points of cross-validated accuracy, and negation marking improved accuracy on "
     "negated sentences by about "
     f"{100 * (E6['negation_slice_acc'] - E5['Contains negation'][LR_]):.0f} points. The narrow domain "
     "(professional film critics) limits generalisation to casual reviews.")
h3("8.5 How API or Application Enhances the NLP Task")
para("Wrapping the models in an application adds explanations (top contributing words), a choice between "
     "fast and accurate modes, and a side-by-side lexicon baseline, which makes the system useful to "
     "non-experts and transparent about why it decided. Off-the-shelf APIs such as VADER give instant "
     "results without training data, while the trained models show the gain from domain-specific "
     "labelled data.")

h2("9. Conclusion")
h3("9.1 Summary of the Work Done")
para(f"Seven sentiment-analysis approaches were implemented and compared on {D['n']:,} movie-review "
     f"sentences. The ensemble of TF-IDF Logistic Regression and RoBERTa embeddings achieved the best "
     f"accuracy ({pct(E1[ENS]['accuracy'])}), F1 ({f3(E1[ENS]['f1'])}) and ROC-AUC ({f3(E1[ENS]['roc_auc'])}); "
     f"Linear SVM was the best single model ({pct(E1[SVM]['accuracy'])}); VADER without training reached "
     f"{pct(E1[VAD]['accuracy'])}. Preprocessing ablations, learning curves, significance testing and "
     "linguistic error analysis were carried out, and a working command-line analyser was delivered.")
h3("9.2 Key Learnings and Insights")
bullets([
    "Stopword removal can hurt sentiment analysis: negations and intensifiers are function words.",
    "Pre-trained Transformer embeddings shine with little labelled data; with enough data, simple "
    "features catch up.",
    "Combining complementary models is a cheap, reliable gain.",
    "Evaluation should go beyond accuracy: slices, significance tests and examples reveal what models "
    "actually understand.",
])
h3("9.3 Real-World Applicability")
para("The fast mode is suitable for high-volume monitoring of reviews or social-media posts, and the "
     "accurate mode for lower-volume, higher-stakes analysis such as customer-escalation triage. The same "
     "pipeline can be retrained on any labelled review data (products, restaurants, apps).")
h3("9.4 Future Improvements or Extensions")
bullets([
    "Fine-tune RoBERTa or DistilBERT end-to-end on a GPU, or use parameter-efficient fine-tuning (LoRA, "
    "Unit 3).",
    "Add explicit negation scope detection and contrast-aware weighting of the clause after 'but'.",
    "Compare zero-shot and few-shot prompting of an instruction-tuned LLM.",
    "Extend to three classes (with neutral) and aspect-based sentiment (acting, story, music).",
    "Support Indian languages and code-mixed (Hinglish / Kanglish) reviews.",
])

h2("10. References")
refs = [
    "D. Jurafsky and J. H. Martin, Speech and Language Processing, 3rd ed. draft, 2024 (Ch. 4: Naive Bayes "
    "and Sentiment Classification). (Course textbook)",
    "B. Pang and L. Lee, \"Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with "
    "Respect to Rating Scales,\" Proc. ACL, 2005. (Sentence polarity dataset v1.0)",
    "B. Pang, L. Lee and S. Vaithyanathan, \"Thumbs up? Sentiment Classification using Machine Learning "
    "Techniques,\" Proc. EMNLP, 2002.",
    "C. J. Hutto and E. Gilbert, \"VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social "
    "Media Text,\" Proc. ICWSM, 2014.",
    "S. Das and M. Chen, \"Yahoo! for Amazon: Extracting Market Sentiment from Stock Message Boards,\" "
    "Proc. APFA, 2001. (Negation marking)",
    "S. Wang and C. D. Manning, \"Baselines and Bigrams: Simple, Good Sentiment and Topic "
    "Classification,\" Proc. ACL, 2012.",
    "Y. Kim, \"Convolutional Neural Networks for Sentence Classification,\" Proc. EMNLP, 2014.",
    "R. Socher et al., \"Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank,\" "
    "Proc. EMNLP, 2013.",
    "Y. Liu et al., \"RoBERTa: A Robustly Optimized BERT Pretraining Approach,\" arXiv:1907.11692, 2019.",
    "A. Vaswani et al., \"Attention Is All You Need,\" Advances in NeurIPS, 2017.",
    "Q. McNemar, \"Note on the Sampling Error of the Difference Between Correlated Proportions or "
    "Percentages,\" Psychometrika, vol. 12, pp. 153-157, 1947.",
    "F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" JMLR, vol. 12, 2011; S. Bird, "
    "E. Klein and E. Loper, Natural Language Processing with Python (NLTK), O'Reilly, 2009.",
    "L. Tunstall, L. von Werra and T. Wolf, Natural Language Processing with Transformers, O'Reilly, "
    "2022. (Course reference book)",
    "E. M. Bender et al., \"On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?\" "
    "Proc. ACM FAccT, 2021.",
]
for i, r_ in enumerate(refs):
    p = para(f"[{i + 1}] {r_}", size=11, align="left", space_after=4, indent=0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)

doc.save(OUT)
print("saved", OUT)
