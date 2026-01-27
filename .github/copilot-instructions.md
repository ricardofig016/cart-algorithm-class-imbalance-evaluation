# CART Class Imbalance Research Project

## Project Architecture

This is a comparative research project evaluating **baseline CART vs. modified CART** (with weighted Gini) for handling class-imbalanced datasets. The codebase has a dual-pipeline design with parallel implementations for comparison.

### Core Components

```
src/cart/
  ├── cart.py              # Baseline CART (unweighted Gini)
  └── modified_cart.py     # Modified CART (weighted Gini with class_weight="balanced")

src/utils/
  ├── preprocess.py        # Data cleaning, encoding, normalization (shared)
  ├── evaluation.py        # Baseline evaluation pipeline
  ├── newevaluation.py     # Modified CART evaluation pipeline
  ├── visualization.py     # Baseline results visualization
  ├── newvisualization.py  # Modified results visualization
  └── comparison.py        # Statistical comparison (paired t-tests)
```

**Key Insight**: Files with `new` prefix (newevaluation.py, newvisualization.py) correspond to the modified CART implementation. This naming pattern is intentional for phase-based development.

## Critical Implementation Rules

### 1. Dual Pipeline Consistency

When modifying evaluation/visualization logic:

- **Always update BOTH pipelines** (base and modified) unless the change is specific to weighted Gini
- Both `evaluation.py` and `newevaluation.py` must have identical:
  - Hyperparameters (currently: `max_depth=5, criterion="gini"`)
  - Metrics (accuracy, precision, recall, f1, roc_auc)
  - Data loading logic

Example:

```python
# evaluation.py
tree = DecisionTree(max_depth=5, criterion="gini")

# newevaluation.py
tree = DecisionTree(max_depth=5, criterion="gini", class_weight="balanced")
```

### 2. Data Structure Contract

Preprocessed datasets follow a strict structure in `data/processed/class_imbalance/{dataset_name}/`:

- `X_train.csv`, `X_test.csv`: Numerical features (already normalized 0-1)
- `y_train.csv`, `y_test.csv`: Integer labels (0-indexed)

**Never** modify this structure - the entire pipeline depends on it. When adding datasets, follow `preprocess.py` output format exactly.

### 3. Weighted Gini Implementation

The class imbalance modification lives in `modified_cart.py`:

- `_compute_class_weights()`: Computes inverse frequency weights when `class_weight="balanced"`
- `_calculate_weighted_impurity()`: Applies weights to Gini calculation **before** splitting
- `_most_common()`: Uses weighted majority voting for leaf predictions

**Critical**: Weight application happens at impurity calculation, NOT at the split decision level.

### 4. No sklearn for Trees

- Baseline CART is a **clean-room implementation** adapted from zziz/cart
- Use sklearn ONLY for: metrics (`sklearn.metrics`), preprocessing (`train_test_split`, `MinMaxScaler`), label encoding
- Never import `sklearn.tree.DecisionTreeClassifier` - defeats the research purpose

## Development Workflows

### Running Complete Evaluation (All Datasets)

Execute notebooks cells sequentially in `assignment.ipynb`:

1. **Setup**: Adds `src/` to path
2. **Preprocessing**: Runs once to generate train/test splits
3. **Evaluation**: Trains on ~50 datasets, saves to `results/{evaluation_data.csv, newevaluation_data.csv}`
4. **Comparison**: Generates statistical plots in `results/comparisons/`

**Timing**: Full evaluation takes 10-15 minutes per pipeline.

### Quick Testing (3 Datasets)

Modify evaluation calls:

```python
results = evaluate(data_dir="data/processed/class_imbalance/", max_datasets=3)
```

### Adding New Metrics

1. Update `evaluation.py` to import and compute metric
2. Append to `dataset_object` dictionary
3. Apply **identical changes** to `newevaluation.py`
4. Update `comparison.py` metrics list (line 53)
5. Update `perform_statistical_tests()` to include in t-tests

## Statistical Validation Conventions

- Use **paired t-tests** (`scipy.stats.ttest_rel`) for metric comparisons - datasets are paired (same train/test splits)
- Significance markers: `***` (p<0.001), `**` (p<0.01), `*` (p<0.05)
- Report both mean difference AND p-value on plots (see `comparison.py:plot_improvement_summary`)

## Common Pitfalls

1. **Label Conversion**: `modified_cart.py` handles string labels via `_convert_labels()` - always test with both numeric and categorical targets
2. **Zero Division**: Check `if len(y) == 0` before computing impurities (edge case in highly imbalanced splits)
3. **Results Overwrite**: `save_results()` overwrites CSV files - back up manually if iterating on experiments
4. **Preprocessing Idempotence**: Re-running `preprocess_datasets()` regenerates splits (changes results) - only run once per experiment session

## File Operation Patterns

- Results CSVs use `index=True` for readability but aren't loaded with `index_col=0`
- Matplotlib figures use `tight_layout()` and conditionally save/show based on function params
- Always use `os.path.join()` for cross-platform paths (project used on Windows/Unix)

## Reference Files

- [PLAN.md](../PLAN.md): Original project phases and timeline
- [Assignment.pdf](../Assignment.pdf): Full research specifications
- `src/cart/cart.py` header comments: Detailed implementation notes vs. reference code
