# Machine Learning-Based Similarity Analysis of Source Programs

Compiler Design mini project (E325 Devayani Hekare, E308 Manasvi Bhardwaj).
Compares two **C-subset** programs by structure (tokens + AST), not by text, and uses a
trained ML model to output a similarity score, a High/Moderate/Low class and an explanation.

## Pipeline
`Source A/B -> Preprocess -> Lexer (tokens) -> Parser (AST) -> Feature extraction -> Pair feature vector -> ML model -> Score + class + report`

| Stage | File |
|---|---|
| Preprocessing (comments, directives, whitespace) | `simcode/preprocess.py` |
| Lexical analysis (regex scanner, token types) | `simcode/lexer.py` |
| Syntax analysis (recursive-descent parser -> AST) | `simcode/parser.py` |
| Feature extraction + pair feature vector | `simcode/features.py` |
| Labelled dataset generator (similar / moderate / dissimilar pairs) | `simcode/programs.py` |
| Model comparison, selection, saving | `simcode/training.py`, `train.py` |
| Prediction, explanation, chart | `simcode/report.py`, `compare.py` |
| Tests | `tests/test_pipeline.py` |

## Setup (Python 3.9+)
```
pip install -r requirements.txt
```

## Run
```
python train.py                                   # builds dataset, compares 4 models, saves models/ + results/
python compare.py samples/a_sum.c samples/b_add.c # High
python compare.py samples/c_factorial_loop.c samples/d_factorial_while.c --plot   # Moderate + chart
python compare.py samples/a_sum.c samples/e_bubble_sort.c                         # Low
python compare.py samples/a_sum.c samples/bad_syntax.c                            # syntax error report
python compare.py A.c B.c --tokens --ast          # show tokens and ASTs
python -m unittest discover -s tests -v           # 12 tests
```
`compare.py` trains automatically if no model exists.

## ML details
* Candidates: Logistic Regression, Decision Tree, Random Forest, SVM (RBF). Selected by 5-fold CV macro-F1.
* Labels: High = same algorithm and implementation (renamed/reformatted/commented), Moderate = same algorithm with a
  different implementation (e.g. for vs while, iterative vs recursive) or different algorithms of the same group
  (e.g. sum vs product of an array), Low = algorithms from different groups.
* Dataset: 29 algorithm families (53 implementation variants) in 12 groups; 5 families are held out for the unseen test set.
* Score = sum of class probability x representative value (Low 10, Moderate 55, High 95);
  class from score: >=75 High, >=35 Moderate, else Low.
* Evaluation includes a test set of algorithms the model never saw in training (`results/metrics.txt`).

## Limitations (as per scope)
C subset only (functions, declarations, if/else, for, while, do-while, switch, return, expressions, arrays, calls).
No preprocessor macro expansion, structs/typedefs beyond basic parsing. Training data is synthetic, so scores are
indicative; similarity is not proof of plagiarism. No internet-wide search, authorship detection or obfuscation handling.
