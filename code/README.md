# HW3 — AI-generated text detection (UCU AI 2026) — code

Five Kaggle notebooks, run in this order. Each one rebuilds the same fixed split
(`make_split`, seed 42: 10,000 validation rows stratified by label × language, 40,000 training rows).
The first three validation ids are printed in every notebook and must be `['09b7f8', 'c6e5ab', '4c07aa']`.

| # | Notebook | Kaggle settings | Inputs (Add Input) | Main outputs | Runtime |
|---|---|---|---|---|---|
| 01 | `01_eda_split.ipynb` | Accelerator None | competition data | EDA tables/plots, `split_ids.csv` | ~10 min |
| 02 | `02_baseline_tfidf.ipynb` | Accelerator None | competition data | `baseline_grid.csv`, `baseline_confusion.png`, `baseline_{val,test}_proba.csv`, `submission_baseline.csv` | ~40 min |
| 03 | `03_encoder_xlmr.ipynb` | GPU T4 ×2, Internet On | competition data | `xlmr_model/`, `long_text_comparison.csv`, `xlmr_{val,test}_proba_*.csv`, `submission_xlmr.csv` | ~2 h 15 min |
| 04 | `04_ensemble.ipynb` | Accelerator None | competition data + outputs of 02 and 03 | `ensemble_results.csv`, `final_confusion.png`, `submission_ensemble.csv` | ~5 min |
| 05 | `05_xai.ipynb` | Accelerator None (GPU optional) | competition data + output of 03 | `xai_perturbation.csv`, `xai_top_words.png`, `xai_examples.html` | ~1–2 h on CPU |

How to run each notebook on Kaggle: competition page → Code → New Notebook → File → Import Notebook →
set the inputs/accelerator from the table → Save Version → Save & Run All (Commit).

Final submission: `submission_ensemble.csv` from notebook 04 (stacking: logistic regression on XLM-R and TF-IDF
log-probabilities + simple stylometric features + language/domain, fitted on the validation split).

Libraries: the default Kaggle Python image (pandas, scikit-learn, PyTorch, transformers). Pretrained weights:
`xlm-roberta-base` from the Hugging Face Hub. No external data, no closed LLMs.

References: Sun et al. (2019) "How to Fine-Tune BERT for Text Classification?" (head+tail truncation);
Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks" (Integrated Gradients).
Code was written with help from an AI assistant (allowed for code by the course rules).
