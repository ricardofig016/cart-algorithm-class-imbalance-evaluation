# Assignment

## **Phase 1: Baseline Implementation & Analysis**

(Complete by checkpoint: 24th April – 2nd May)

### **Step 1: Set Up Environment**

1. Create a Jupyter Notebook.
2. Install required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn` (for data splitting/metrics only).
3. Clone or download code for a basic CART implementation (e.g., [MLAlgorithms](https://github.com/rushter/MLAlgorithms/tree/master)) as a starting point.

---

### **Step 2: Understand CART Algorithm**

1. Study the CART code from your chosen source.
2. Document how the algorithm:
   - Computes Gini impurity for splits.
   - Handles categorical vs. continuous features.
   - Grows the tree (stopping criteria, pruning).
3. Write a summary of CART’s weaknesses in class imbalance (e.g., bias toward majority class).

---

### **Step 3: Select & Preprocess Datasets**

1. Use **Dataset Group 2 (Class Imbalance)**. Recommended datasets:
   - Credit Card Fraud Detection (Kaggle)
   - Spambase (UCI)
   - Medical Insurance Fraud (synthetic if needed).
2. Preprocess:
   - Handle missing values (drop or impute).
   - Encode categorical variables (one-hot encoding).
   - Normalize numerical features (min-max scaling).
   - Split data into train/test (80/20) with stratification to preserve imbalance.

---

### **Step 4: Implement Baseline CART**

1. Adapt the cloned CART code to work with your datasets.
2. Train the model on the training set.
3. Evaluate performance on the test set using:
   - **Metrics**: F1-score, ROC-AUC, Precision-Recall curve (critical for imbalance).
   - Baseline accuracy (compare to majority-class dummy classifier).

---

### **Step 5: Document Phase 1**

1. Save results in a table (e.g., baseline metrics for all datasets).
2. Write a brief analysis:
   - How does class imbalance affect CART’s performance?
   - Which datasets are most challenging?

---

## **Phase 2: Modify CART for Class Imbalance**

(Complete by final deadline: 25th May)

### **Step 6: Propose Modification**

1. Modify CART to handle imbalance. Options:
   - **Adjust split criterion**: Use weighted Gini impurity (weight minority class more).
   - **Cost-sensitive learning**: Assign higher misclassification cost to minority class.
   - **Hybrid approach**: Combine CART with SMOTE (synthetic oversampling).
2. Justify your choice (e.g., "Weighted Gini reduces bias toward majority class").

---

### **Step 7: Implement Modified CART**

1. Edit the CART code:
   - Add class weights to Gini calculation (e.g., `weighted_gini = (1 - (p_minority^2 + p_majority^2)) * class_weight`).
   - Modify splitting logic to prioritize minority-class-friendly splits.
2. Validate code with unit tests (e.g., check if splits change with class weights).

---

### **Step 8: Evaluate Modified CART**

1. Train the modified model on the same datasets.
2. Compare results to baseline using:
   - Same metrics (F1, ROC-AUC).
   - Statistical tests (e.g., paired t-test for significance).
3. Visualize improvements (e.g., ROC curves side-by-side).

---

### **Step 9: Final Documentation**

1. Update the notebook with:
   - Clean, commented code for both baseline and modified CART.
   - Results tables/visualizations.
   - Conclusion (e.g., "Weighted Gini improved F1 by 15% on fraud dataset").
2. Prepare slides following the **Presentation Guidelines** (see template below).

---

## **Presentation Slide Template**

1. **Cover Slide**: Group members’ names, student IDs.
2. **Executive Summary**: "Modified CART with weighted Gini to address class imbalance."
3. **CART & Class Imbalance**:
   - Diagram of Gini impurity with/without weights.
   - Example of imbalanced split in baseline vs. modified CART.
4. **Proposal**: Formula for weighted Gini, pseudocode snippet.
5. **Results**:
   - Table comparing F1/AUC across datasets.
   - Learning curves showing stability.
6. **Conclusions**: Strengths/limitations of the modification.

---

## **Timeline**

- **Week 1 (Now)**: Set up codebase, preprocess datasets.
- **Week 2**: Implement baseline CART, run initial tests.
- **Week 3**: Propose modification, start coding.
- **Week 4**: Finalize implementation, run evaluations.
- **Week 5**: Prepare slides, polish notebook.
