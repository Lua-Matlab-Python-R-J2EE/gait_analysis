# Machine Learning–Based Supervised and Unsupervised Gait Analysis

ABOUT THE PROJECT
-----------------
- This is a machine learning (ML) study of gait data, exploring nine supervised (classifcation) and unsupervised (clustering) approaches to understand model behavior, feature importance, and methodological best practices. 
- It demonstrates strong applied ML skills, including data cleaning, pipeline engineering, cross-validation design, oversampling strategies, evaluation under extreme data constraints, and the prevention of data leakage. 
- The work highlights both what works and what fails, providing a clear, rigorous analysis suitable for real-world ML problem-solving.


DATASET
=========
1. Original: `gait_final_output.csv`
2. Cleaned: `gait_final_output_updated.csv`
> Data Source/Credit: https://archive.ics.uci.edu/dataset/604/gait+classification


GAIT ANALYSIS NOTEBOOK VERSIONS - QUICK REFERENCE
==================================================

MAIN COMPARISON TABLE
---------------------

| Version | Name                       | Split Strategy                           | Oversampling                                   | CV Method                    | Data Leakage? | Person-Independent? | Key Result                                              | Valid For                               | Status               |
|--------|-----------------------------|--------------------------------------------|------------------------------------------------|------------------------------|---------------|----------------------|---------------------------------------------------------|-------------------------------------------|-----------------------|
| v1_1   | Baseline Approach           | 67/33 train/test                           | SMOTEENN / SMOTETomek (in pipeline)            | Adaptive (StratifiedKFold or none) | No            | No                   | 0.00 metrics, 2 train samples/class                   | Establish baseline                        | Data too small       |
| v1_1_mf | Baseline + Modified Features | 67/33 train/test                          | SMOTEENN / SMOTETomek (in pipeline)            | Adaptive (StratifiedKFold or none) | No            | No                   | Reduced performance vs v1_1                           | Feature engineering comparison            | Data too small       |
| v1_1_mfr | Baseline + Modified Features Reduced | 67/33 train/test              | SMOTEENN / SMOTETomek (in pipeline)            | Adaptive (StratifiedKFold or none) | No            | No                   | Same performance as v1_1                              | Feature subset comparison                 | Data too small       |
| v1_2   | GridSearchCV Optimization   | None (use all data)                        | SMOTEENN / SMOTETomek (in pipeline)            | StratifiedKFold (3-fold)     | No            | No                   | KNN best model, still only 2 samples/class             | Feature analysis                           | Data too small       |
| v1_2_mfr | GridSearchCV + Modified Features Reduced | None (use all data)      | SMOTEENN / SMOTETomek (in pipeline)            | StratifiedKFold (3-fold)     | No            | No                   | KNN best: mean 0.81, median 0.54, std 0.71             | Feature transformation comparison          | Data too small       |
| v1_3   | Aggressive Oversampling     | 68/32 train/test                           | RandomOverSampler (100/class, in pipeline)     | None                         | No            | No                   | Perfect train, poor test (overfitting)                 | Oversampling limits                        | VERY BAD             |
| v1_4   | Pre-Split Oversampling      | 65/35 train/test (after oversample)        | SMOTE / RandomOverSampler (before split)       | None                         | **YES**       | No                   | Artificially inflated test accuracy                    | What NOT to do                              | VERY BAD             |
| v1_5   | Outside-Pipeline Oversample | 65/35 train/test                           | SMOTE / RandomOverSampler (outside pipeline)   | None                         | Unclear       | No                   | Unknown                                               | Need more info                              | Unclear              |
| v1_6   | LOOCV Pre-Oversample        | None (use all data)                        | SMOTE / RandomOverSampler (before CV)          | LOOCV (480 folds)            | **YES**       | No                   | 99.4% accuracy (leakage)                               | Feature discriminability only               | Invalid methodology  |
| v1_7   | StratifiedKFold Pipeline    | None (use all data)                        | RandomOverSampler (in pipeline)                | StratifiedKFold (2-fold)     | No            | No                   | Best k=15 features, stable results                    | Feature analysis, model comparison           | Best for tiny data   |
| v1_7_mfr | StratifiedKFold + Modified Features Reduced | None (use all data)   | RandomOverSampler (in pipeline)                | StratifiedKFold (2-fold)     | No            | No                   | mean k=25 (0.90), median k=20 (0.77), std k=20 (0.90) | Feature transformation analysis              | Best for tiny data   |
| v1_8   | Train/Test Pipeline         | 65/35 train/test                           | RandomOverSampler (in pipeline)                | None                         | No            | No                   | 88.2%, but unstable (1 test/class)                    | Standard ML practice                         | Data waste           |
| v1_9   | Unsupervised Clustering     | None (use all data)                        | None (unsupervised)                            | None (clustering)            | N/A           | N/A                  | DBSCAN: reduces 16 classes -> 2 clusters               | Exploratory analysis, class reduction        | Valid alternative    |




SPLIT STRATEGY COMPARISON
--------------------------

| Approach                         | Train Samples      | Test Samples       | Data Utilization | Stability                          |
|----------------------------------|--------------------|--------------------|------------------|-------------------------------------|
| No split (v1_2, v1_2_mfr, v1_6, v1_7, v1_7_mfr, v1_9) | All 48             | All 48             | 100%             | High (CV) or N/A (unsupervised)     |
| 67/33 split (v1_1, v1_1_mf, v1_1_mfr) | ~32 (~2/class)     | ~16 (~1/class)     | 67%              | Low                                 |
| 68/32 split (v1_3)               | ~32 (~2/class)     | ~16 (~1/class)     | 68%              | Low                                 |
| 65/35 split (v1_4, v1_5, v1_8)   | ~31 (~2/class)     | ~17 (~1/class)     | 65%              | Low                                 |



OVERSAMPLING STRATEGY COMPARISON
---------------------------------

| Method                   | Timing        | Location            | Creates New Info? | Leakage Risk |
|--------------------------|---------------|----------------------|-------------------|--------------|
| In pipeline (v1_1, v1_1_mf, v1_1_mfr, v1_2, v1_2_mfr, v1_3, v1_7, v1_7_mfr, v1_8) | After split   | sklearn pipeline     | Limited       | Low          |
| Before split (v1_4)      | Before split  | Outside pipeline     | Yes               | **HIGH**      |
| Before CV (v1_6)         | Before CV     | Outside pipeline     | Yes               | **HIGH**      |
| Outside pipeline (v1_5)  | After split?  | Outside pipeline     | Unclear           | Varies        |
| None – Unsupervised (v1_9) | N/A        | N/A                  | N/A               | N/A          |



CROSS-VALIDATION METHODS
-------------------------

| Method                | Folds | Samples/Fold             | Best For                     | Used In  |
|-----------------------|--------|---------------------------|-------------------------------|----------|
| Adaptive CV           | Variable | ~16 train if used        | Very small datasets           | v1_1, v1_1_mf, v1_1_mfr |
| StratifiedKFold (3)   | 3      | 16 train, 16 test         | Balanced small data           | v1_2, v1_2_mfr |
| StratifiedKFold (2)   | 2      | 24 train, 24 test         | Tiny datasets                 | v1_7, v1_7_mfr |
| LOOCV                 | 480    | 47 train, 1 test (after OS) | Maximum use of small data   | v1_6     |
| None (split)          | 1      | ~31–32 train, ~16–17 test | Standard practice             | v1_1, v1_1_mf, v1_1_mfr, v1_3, v1_4, v1_8 |
| None (unsupervised)   | N/A    | All 48                    | Exploratory analysis          | v1_9     |



FEATURE ENGINEERING VARIANTS
-----------------------------

| Version Suffix | Feature Count | Description | Transformations Applied |
|---------------|---------------|-------------|------------------------|
| (base)        | Original      | Original features from dataset | None |
| _mf           | 321           | Modified Features - R1, R2, R3 transformed into mean, median, std (all 321 used) | Mean, Median, Std aggregation |
| _mfr          | 107           | Modified Features Reduced - Only one transformation at a time (mean OR median OR std) | Single transformation subset |



PERFORMANCE SUMMARY
-------------------

| Version | Accuracy          | Precision/Recall Issues | Reliability                           | Notes                                    |
|---------|-------------------|--------------------------|----------------------------------------|------------------------------------------|
| v1_1    | Very poor         | Many 0.00 metrics        | Low                                    | Baseline                                 |
| v1_1_mf | Reduced vs v1_1   | Many 0.00 metrics        | Low                                    | 321 features hurt performance            |
| v1_1_mfr | Same as v1_1     | Many 0.00 metrics        | Low                                    | 107 features maintains baseline          |
| v1_2    | Variable          | Many 0.00 metrics        | Low                                    | KNN best model                           |
| v1_2_mfr | Mean: 0.81, Median: 0.54, Std: 0.71 | Varies by transformation | Low                  | Mean features perform best               |
| v1_3    | Perfect train     | Many 0.00 metrics        | Invalid (overfitting)                  | Aggressive oversampling                  |
| v1_4    | Artificially high | Inflated by leakage      | Invalid                                | Pre-split oversampling                   |
| v1_5    | Unknown           | Unknown                  | Unknown                                | Unclear methodology                      |
| v1_6    | 99.4%             | Nearly perfect           | Invalid (leakage)                      | Pre-CV oversampling                      |
| v1_7    | Good (k=15)       | Stable                   | Reliable                               | Best methodology for tiny data           |
| v1_7_mfr | Mean: 0.90, Median: 0.77, Std: 0.90 | Stable | Reliable                               | Mean and std features perform similarly  |
| v1_8    | 88.2%             | Subjects 1,11 = 0.00     | Unstable (1 test/class)                | Standard practice but wasteful           |
| v1_9    | N/A               | N/A                      | Valid (exploratory)                    | Clustering reduces to 2 classes          |


RECOMMENDATION BY USE CASE
---------------------------
Best Practices
- Model Comparison: v1_2
- Feature Analysis: v1_7
- Feature Transformation Analysis: v1_2_mfr, v1_7_mfr
- Class Reduction: v1_9


What NOT to Do
- v1_3: Aggressive oversampling - overfitting
- v1_4: Oversampling before split
- v1_6: Oversampling before CV

Avoid These
- v1_3 - Invalid performance
- v1_4, v1_6 - Contaminated test folds


QUICK DECISION GUIDE
---------------------
- Compare models? - v1_2  
- Understand leakage? - v1_4, v1_6  
- Analyze features? - v1_7  
- Compare feature transformations? - v1_2_mfr, v1_7_mfr
- Reduce classes? - v1_9  
- Learn correct methodology? - v1_7, v1_7_mfr, v1_8  
- Deploy biometric system? - None (need person-independent data)  
- Get real-world accuracy? - None (train/test overlap subjects)

KEY CONCEPTS
-------------
Data Leakage
- Correct: Oversampling after split or in-pipeline
- Incorrect: Oversampling before split or before CV

Person-Independent
- No: Train/test contain same subjects (all current versions)
- Yes: Requires GroupKFold by subject (not available here)

Feature Engineering
- Modified Features (mf): Aggregating R1, R2, R3 trials into mean, median, std creates 321 features
- Modified Features Reduced (mfr): Using only one transformation (mean OR median OR std) gives 107 features
- Finding: Mean features generally perform best; using all 321 features can reduce performance vs baseline

Status Legend
- Valid: Methodologically correct
- Limited: Correct but insufficient data
- Invalid: Methodology broken (usually leakage)


FUNDAMENTAL LIMITATIONS (All Supervised Versions)
--------------------------------------------------
- Train/test not person-independent
- Only 3 samples/subject - too small
- 321 features, 48 samples - severe dimensionality issues
- Feature engineering (mean/median/std) doesn't overcome fundamental data limitations

What is Needed for Real Biometric System
----------------------------------------
- 100+ samples per subject
- Person-independent splits
- Enrollment vs verification sets
- Condition variation testing

CONCLUSION
-----------
- Best Notebook for Learning: v1_7 - StratifiedKFold Pipeline: Methodologically correct, no leakage, solid for tiny datasets.
- Best for Feature Transformation Study: v1_7_mfr - Shows mean and std features perform similarly (0.90 accuracy), both outperforming median (0.77)
- Best for Exploration: v1_9 - Unsupervised Clustering: Simplifies 16 classes to 2, avoids train/test issues.
- Worst (Due to Leakage): v1_4, v1_6
- Feature Engineering Insight: Simple aggregations (mean/median/std) don't solve the fundamental problem of insufficient samples
