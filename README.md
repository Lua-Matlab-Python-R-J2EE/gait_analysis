# Machine learning based supervised and un-supervised gait analysis


DATASET
=========
1. original: gait_final_output.csv
2. cleaned: gait_final_output_updated.csv

GAIT ANALYSIS NOTEBOOK VERSIONS - QUICK REFERENCE
==================================================

MAIN COMPARISON TABLE
---------------------

| Version | Name                       | Split Strategy                         | Oversampling                                   | CV Method                    | Data Leakage? | Person-Independent? | Key Result                                              | Valid For                               | Status        |
|--------|-----------------------------|-----------------------------------------|------------------------------------------------|------------------------------|---------------|----------------------|---------------------------------------------------------|-------------------------------------------|--------------|
| v1_1   | Baseline Approach           | 67/33 train/test                        | SMOTEENN / SMOTETomek (in pipeline)            | Adaptive (StratifiedKFold or none) | No            | No                   | 0.00 metrics for many classes, 2 train samples/class   | Establishing baseline methodology          | Data too small |
| v1_2   | GridSearchCV Optimization   | None (use all data)                     | SMOTEENN / SMOTETomek (in pipeline)            | StratifiedKFold (3-fold)     | No            | No                   | KNN best model, limited by 2 train samples/class       | Feature analysis                           | Data too small |
| v1_3   | Aggressive Oversampling     | 68/32 train/test                        | RandomOverSampler (100/class, in pipeline)     | None                         | No            | No                   | Perfect train, poor test (overfitting)                 | Demonstrates oversampling limits            | VERY BAD      |
| v1_4   | Pre-Split Oversampling      | 65/35 train/test **after oversample**   | SMOTE / RandomOverSampler (before split)       | None                         | **YES**       | No                   | Artificially high test accuracy                        | What NOT to do                              | VERY BAD      |
| v1_5   | Outside-Pipeline Oversample | 65/35 train/test                        | SMOTE / RandomOverSampler (outside pipeline)   | None                         | Unclear       | No                   | Unknown                                               | Need more info                              | Unclear       |
| v1_6   | LOOCV Pre-Oversample        | None (use all data)                     | SMOTE / RandomOverSampler (before CV)          | LOOCV (480 folds)            | **YES**       | No                   | 99.4% accuracy (data leakage)                          | Proves feature discriminability only         | Invalid methodology |
| v1_7   | StratifiedKFold Pipeline    | None (use all data)                     | RandomOverSampler (in pipeline)                | StratifiedKFold (2-fold)     | No            | No                   | Best k=10 features, stable results                    | Feature analysis, model comparison           | Best for tiny data |
| v1_8   | Train/Test Pipeline         | 65/35 train/test                        | RandomOverSampler (in pipeline)                | None                         | No            | No                   | 88.2% but unstable (1 test sample/class)              | Standard ML practice                         | Data waste    |



SPLIT STRATEGY COMPARISON
--------------------------

Approach | Train Samples | Test Samples | Data Utilization | Stability
---------|---------------|--------------|------------------|----------
No split (v1_2, v1_6, v1_7) | All 48 via CV | All 48 via CV | 100% | High (averaging)
67/33 split (v1_1) | ~32 (~2/class) | ~16 (~1/class) | 67% training | Low (single split)
68/32 split (v1_3) | ~32 (~2/class) | ~16 (~1/class) | 68% training | Low (single split)
65/35 split (v1_4, v1_5, v1_8) | ~31 (~2/class) | ~17 (~1/class) | 65% training | Low (single split)


OVERSAMPLING STRATEGY COMPARISON
---------------------------------

Method | Timing | Location | Creates New Info? | Data Leakage Risk
-------|--------|----------|-------------------|------------------
In Pipeline (v1_1, v1_2, v1_3, v1_7, v1_8) | After split | Inside sklearn pipeline | Limited (duplicates) | Low (if split first)
Before Split (v1_4) | Before split | Outside pipeline | Yes, but contaminates test | HIGH
Before CV (v1_6) | Before CV | Outside pipeline | Yes, but contaminates folds | HIGH
Outside Pipeline (v1_5) | After split? | Outside pipeline | Unclear | Depends on order


CROSS-VALIDATION METHODS
-------------------------

Method | Folds | Samples/Fold | Best For | Used In
-------|-------|--------------|----------|--------
Adaptive CV (v1_1) | Variable (0-2) | ~16 train if used | Very small data with fallback | v1_1
StratifiedKFold (3-fold) | 3 | 16 train, 16 test | Balanced small data | v1_2
StratifiedKFold (2-fold) | 2 | 24 train, 24 test | Very small data | v1_7
LOOCV | 480 (after oversample) | 479 train, 1 test | Maximum data use | v1_6
None (single split) | 1 | ~31-32 train, ~16-17 test | Standard practice | v1_1, v1_3, v1_4, v1_8


PERFORMANCE SUMMARY
-------------------

Version | Accuracy | Precision/Recall Issues | Reliability
--------|----------|------------------------|-------------
v1_1 | Very poor | Many 0.00 metrics | Low (2 train, 1 test per class)
v1_2 | Variable | Many 0.00 metrics | Low (2 samples/class)
v1_3 | Perfect train, poor test | Many 0.00 metrics | Invalid (overfitting)
v1_4 | Artificially high | Inflated by leakage | Invalid (leakage)
v1_5 | Unknown | Unknown | Unknown
v1_6 | 99.4% | Nearly perfect | Invalid (leakage)
v1_7 | Good with k=10 | Stable across folds | Reliable for this task
v1_8 | 88.2% | Subjects 1,11 = 0.00 | Unstable (1 test/class)


RECOMMENDATIONS BY USE CASE
----------------------------

BEST PRACTICES (For This Dataset):

1. Feature Analysis - Use v1_7 (StratifiedKFold + Pipeline)
   - No data leakage; Uses all 48 samples; Stable results across folds; Tests multiple feature counts

2. Model Comparison - Use v1_2 (GridSearchCV)
   - Systematic hyperparameter tuning; Multiple models tested; Cross-validated results

3. Learn What NOT to Do:
   - v1_3: Shows limits of aggressive oversampling   
   - v1_4: Shows why oversampling before split is wrong
   - v1_6: Shows why oversampling before CV is wrong   

4. AVOID THESE APPROACHES:
   - v1_4, v1_6: Data leakage invalidates all results
   - v1_3: Overfitting demonstration, not useful results


QUICK DECISION GUIDE
---------------------

Want to:
  - Compare models systematically? --> v1_2
  - Understand data leakage? --> v1_4, v1_6 (what NOT to do)  
  - Analyze features properly? --> v1_7
  - Learn correct methodology? --> v1_7, v1_8
  - Deploy biometric system? --> None (need person-independent approach with more data)
  - Get real-world accuracy? --> None (all test on same subjects seen in training)


KEY CONCEPTS EXPLAINED
-----------------------

Data Leakage:
  - No: Oversampling happens after split or inside pipeline (correct)
  - YES: Oversampling happens before split or before CV (incorrect - test data contaminated)

Person-Independent:
  - No: Train and test both contain same subjects (different samples)
  - Yes: Train and test contain completely different subjects (would need GroupKFold)

Status Legend:
  - Valid:   Methodologically sound for stated purpose
  - Limited: Correct methodology but limited by data size
  - Invalid: Fundamental methodological flaw (data leakage)


FUNDAMENTAL LIMITATIONS (ALL VERSIONS)
---------------------------------------

Shared Issues:
  - Not person-independent: All versions test on subjects seen during training
  - Insufficient data: 3 samples/subject too small for reliable ML
  - High dimensionality: 321 features vs 48 samples (curse of dimensionality)

What's Needed for Real Biometric System:
  - 100+ samples per subject (not 3)
  - Person-independent evaluation (GroupKFold by subject)
  - Separate enrollment and verification sets
  - Robustness testing across conditions


CONCLUSION
----------

Best Notebook for Learning: v1_7 (StratifiedKFold + Pipeline)
  - Methodologically sound; No data leakage; Efficient data use; Stable results; Proper pipeline implementation

Worst Notebooks (Data Leakage): v1_4, v1_6
  - Fundamental methodological flaws; Results completely invalid; Good educational examples of what NOT to do

Key Insight: 
  - With only 48 samples (3 per class), no approach can achieve true person-independent biometric identification. 
  - The best we can do is understand feature discriminability and learn proper ML methodology.
