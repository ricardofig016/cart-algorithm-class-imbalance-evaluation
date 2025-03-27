# Practical Assignment

## **Phase 1: Algorithm Selection & Baseline Evaluation**

Deadline: 24th April – 2nd May (Checkpoint)

### **Step 1: Choose a Classification Algorithm**

1. **Pick a simple, well-understood algorithm** (e.g., Decision Tree, Logistic Regression, Naive Bayes, or k-NN).
2. **Find open-source code** for the chosen algorithm (e.g., from [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms)).
   - Example: Use `decision_tree.py` for Decision Trees.
   - **Reference the source code** in your report.

### **Step 2: Understand the Algorithm**

1. **Study the code line-by-line** and map it to the algorithm’s mathematical theory.
2. **Write pseudocode** for the algorithm’s key steps (e.g., entropy calculation for Decision Trees).
3. **Hypothesize weaknesses**: Which data characteristic (noise, imbalance, multiclass) impacts it most?
   - Example: Decision Trees overfit noisy data.

### **Step 3: Select a Data Challenge**

1. Choose **one dataset group**:
   - **Group 1 (Noise/Outliers)**: Use datasets like Iris with added noise.
   - **Group 2 (Class Imbalance)**: Use datasets like Credit Fraud.
   - **Group 3 (Multiclass)**: Use datasets like MNIST or CIFAR-10.
2. **Download benchmark datasets** from UCI Machine Learning Repository or Kaggle.

### **Step 4: Preprocess Data**

1. **Clean datasets**: Remove duplicates, handle missing values.
2. **Modify datasets** to match your chosen challenge:
   - _Noise_: Add Gaussian noise to 20% of features.
   - _Imbalance_: Downsample the majority class.
3. **Split data**: 70% training, 30% testing.

### **Step 5: Baseline Evaluation**

1. **Run the original algorithm** on your datasets.
2. **Evaluate performance** using metrics:
   - Accuracy, Precision, Recall (for imbalance).
   - Confusion matrix (for multiclass).
3. **Save results** in a table for comparison later.

---

## **Phase 2: Algorithm Modification & Evaluation**

Deadline: 25th May (Final Submission)

### **Step 6: Propose an Algorithm Modification**

1. **Brainstorm changes** to address your chosen data challenge:
   - _Noise_: Add pruning to Decision Trees.
   - _Imbalance_: Implement class weighting in Logistic Regression.
   - _Multiclass_: Modify k-NN to use cosine similarity.
2. **Document the theory** behind your modification.

### **Step 7: Implement the Modified Algorithm**

1. **Edit the original code** to include your change.
   - Example: Add a `max_depth` parameter to Decision Trees to limit overfitting.
2. **Test the code** on a small dataset to debug.

### **Step 8: Evaluate the Modified Algorithm**

1. **Re-run experiments** using the same datasets and splits as Phase 1.
2. **Compare results** with the baseline using:
   - Performance metrics.
   - Visualizations (e.g., ROC curves for imbalance).

---

## **Final Deliverables Preparation**

### **Step 9: Build the Jupyter Notebook**

1. **Structure the notebook** with sections:
   - Introduction, Methodology, Results, Conclusion.
2. **Include code**, visualizations, and explanations.
3. **Ensure reproducibility**: Add comments and save datasets locally.

### **Step 10: Create Presentation Slides**

Follow the **Presentation Guidelines**:

1. **Cover Slide**: Group member names.
2. **Executive Summary**: Goals, approach, key results.
3. **Algorithm & Data Challenge**: Weakness analysis.
4. **Proposal**: Motivation and implementation.
5. **Results**: Side-by-side comparison of original vs. modified algorithm.
6. **Conclusion**: Impact of your modification.

---

## **Key Deadlines & Tips**

- **Checkpoint (24th April – 2nd May)**:
  - Complete Phase 1.
  - Prepare a 1-slide summary of your progress for the professor.
- **Final Submission (25th May)**:
  - Submit compressed file with notebook, slides, and datasets.
- **Tips**:
  - Test code incrementally.
  - Use GitHub for version control.
  - Start slides early and rehearse timing.
