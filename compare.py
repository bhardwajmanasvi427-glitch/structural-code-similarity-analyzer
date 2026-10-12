"""Compare two C source programs.

    python compare.py samples/a_sum.c samples/b_add.c
    python compare.py A.c B.c --plot results/compare.png --tokens --ast
"""
import argparse
import os
import sys

from simcode.errors import CompilerError
from simcode.features import analyze
from simcode.report import plot_comparison, predict, render_text
from simcode.training import load_model


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("program_a")
    ap.add_argument("program_b")
    ap.add_argument("--plot", metavar="PNG", help="save a comparison chart (default: results/comparison.png)",
                    nargs="?", const="results/comparison.png")
    ap.add_argument("--tokens", action="store_true", help="print the token stream of both programs")
    ap.add_argument("--ast", action="store_true", help="print the AST of both programs")
    args = ap.parse_args(argv)

    analyses = []
    for tag, path in (("A", args.program_a), ("B", args.program_b)):
        try:
            src = read(path)
        except OSError as e:
            print("Cannot read %s: %s" % (path, e))
            return 2
        try:
            analyses.append(analyze(src, name=os.path.basename(path)))
        except CompilerError as e:
            print("Program %s (%s) could not be analysed.\n  %s" % (tag, path, e))
            return 1
    a, b = analyses

    if args.tokens:
        for tag, p in (("A", a), ("B", b)):
            print("--- Tokens of program %s ---" % tag)
            print("\n".join("%-8s %-10s (line %d)" % (t.type, t.value, t.line) for t in p.tokens))
    if args.ast:
        for tag, p in (("A", a), ("B", b)):
            print("--- AST of program %s ---" % tag)
            print(p.ast.pretty())

    bundle = load_model()
    score, category, probs, feats = predict(a, b, bundle)
    print(render_text(a, b, score, category, probs, feats, bundle["model_name"]))
    if args.plot:
        print("\nChart saved to", plot_comparison(a, b, score, category, feats, args.plot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
