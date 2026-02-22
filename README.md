# Gait Biometric Classification: ML Methodology Study

Problem Statement
-----------------
- Can gait patterns reliably distinguish between individuals using wearable sensor data? This project investigates that question under a severe real-world constraint: only 48 samples across 16 classes. Rather than forcing a single model to fit inadequate data, this study systematically evaluates nine ML pipeline configurations to identify which methodologies hold up under extreme data scarcity, and which ones silently fail.
- The central contribution is not a deployed classifier but a rigorous comparative analysis of where standard ML practices break down and what alternatives remain valid when data is fundamentally limited.

Key Findings
---------------
- A single methodological error (oversampling before the train/test split) inflated test accuracy from ~50% to 99.4%, a textbook example of data leakage that would be invisible without deliberate investigation. This study documents exactly how and why it occurs across three different leakage scenarios.
- Under correct methodology with no leakage, the best achievable result on this dataset was 90% accuracy using StratifiedKFold (2-fold) with in-pipeline oversampling and mean-aggregated features, but with the important caveat that train and test sets share subjects, so this does not represent person-independent generalisation.
- Unsupervised clustering (DBSCAN) offered a valid alternative framing: rather than classifying 16 individuals, it reduced the problem to 2 natural clusters, sidestepping the supervised learning limitations entirely.

Dataset
----------
- Data Source/Credit: https://archive.ics.uci.edu/dataset/604/gait+classification
- Original: gait_final_output.csv
- Cleaned: gait_final_output_updated.csv
- This is a genuinely difficult ML problem. With ~3 samples per class, standard train/test splits leave approximately 1 test sample per class, making conventional accuracy metrics unreliable and most oversampling strategies dangerous.

Nine Pipeline Configurations Evaluated
---------------------------------------
| Version | Strategy | Leakage? | Valid? | Key Result |
|---------|----------|----------|--------|------------|
| v1_1 | Baseline 67/33 split | No | Limited | 0.00 metrics: data too small |
| v1_2 | GridSearchCV, no split | No | Limited | KNN best; ~2 samples/class |
| v1_3 | Aggressive oversampling | No | Invalid | Perfect train, poor test: overfitting |
| v1_4 | Oversample before split | YES | Invalid | Artificially inflated accuracy |
| v1_6 | Oversample before CV (LOOCV) | YES | Invalid | 99.4% accuracy: leakage artifact |
| v1_7 | StratifiedKFold in-pipeline | No | **VALID** | 90%: best valid methodology |
| v1_8 | Standard train/test pipeline | No | Limited | 88.2% but unstable |
| v1_9 | Unsupervised clustering | N/A | **VALID** | 16 classes -> 2 clusters (DBSCAN) |


What This Study Demonstrates
-------------------------------
- Data leakage detection: three configurations (v1_3, v1_4, v1_6) produce misleadingly high accuracy through different leakage mechanisms, each documented and explained.
- Feature engineering under high dimensionality: aggregating tri-trial features (R1, R2, R3) into mean/median/std creates 321 features from 48 samples. Mean aggregation (107 features) outperforms using all 321, and performs equivalently to std aggregation. More features do not help when samples are scarce.
- Cross-validation strategy selection: StratifiedKFold (2-fold) is the only CV method that uses all data while maintaining class balance without leakage at this sample size. LOOCV appears attractive but enables leakage when combined with pre-CV oversampling.

Fundamental Limitations & What a Real System Would Need
---------------------------------------------------------
- This dataset cannot support a person-independent biometric classifier.
- All supervised versions share subjects between train and test sets. A production gait recognition system would require 100+ samples per subject, GroupKFold splits by subject ID, and separate enrollment vs. verification sets.
- This study treats those limitations as part of the analysis rather than hiding them.


Repository Structure
-----------------------
```plaintext
├── notebooks/          # Nine pipeline versions (v1_1 through v1_9)
├── data/
│   ├── gait_final_output.csv          # Original
│   └── gait_final_output_updated.csv  # Cleaned
└── README.md
```








