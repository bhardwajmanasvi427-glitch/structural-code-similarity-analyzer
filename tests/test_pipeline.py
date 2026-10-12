"""Run with:  python -m unittest discover -s tests -v   (from the project root)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simcode.errors import LexError, ParseError
from simcode.features import analyze
from simcode.lexer import tokenize
from simcode.preprocess import preprocess
from simcode.report import predict
from simcode.training import load_model

A = "int sum(int a, int b) {\n    return a + b;\n}\n"
B = "int add(int x, int y) {\n    return x + y;\n}\n"
LOOP = "int f(int n){int r=1;for(int i=2;i<=n;i++){r*=i;}return r;}"
BUBBLE = ("void s(int a[], int n){for(int i=0;i<n-1;i++){for(int j=0;j<n-i-1;j++){"
          "if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;}}}}")


class Preprocess(unittest.TestCase):
    def test_comments_removed_strings_kept(self):
        r = preprocess('int x; // c1\n/* c2\n c3 */ char *s = "// not a comment";\n#include <stdio.h>\n')
        self.assertNotIn("c1", r.clean)
        self.assertNotIn("c3", r.clean)
        self.assertIn('"// not a comment"', r.clean)
        self.assertNotIn("#include", r.clean)
        self.assertEqual(r.comments_removed, 2)


class Lexer(unittest.TestCase):
    def test_token_types(self):
        types = [t.type for t in tokenize("int x = 3.5 + y;")]
        self.assertEqual(types, ["KEYWORD", "IDENT", "OP", "NUMBER", "OP", "IDENT", "DELIM", "EOF"])

    def test_bad_character(self):
        with self.assertRaises(LexError):
            tokenize("int a = 3 @ 4;")

    def test_unterminated_string(self):
        with self.assertRaises(LexError):
            tokenize('char *s = "abc;')


class Parser(unittest.TestCase):
    def test_ast_shape(self):
        a = analyze(A)
        self.assertEqual(a.metrics["functions"], 1)
        self.assertEqual(a.metrics["params"], 2)
        self.assertEqual(a.metrics["returns"], 1)

    def test_renaming_does_not_change_structure(self):
        a, b = analyze(A), analyze(B)
        self.assertEqual(a.kind_seq, b.kind_seq)
        self.assertEqual(a.norm_tokens, b.norm_tokens)

    def test_syntax_errors(self):
        for bad in ["int f(int a {\n return a; }", "int f() { return a +; }", "int f() { if (x) }",
                    "int f() { int ; }", "int f() { return 1 }"]:
            with self.assertRaises(ParseError, msg=bad):
                analyze(bad)

    def test_nesting_metrics(self):
        m = analyze(BUBBLE).metrics
        self.assertEqual(m["loops"], 2)
        self.assertEqual(m["loop_depth"], 2)
        self.assertEqual(m["ctrl_depth"], 3)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def check(self, s1, s2):
        return predict(analyze(s1), analyze(s2), self.model)

    def test_renamed_is_high(self):
        score, cat, _, _ = self.check(A, B)
        self.assertEqual(cat, "High")
        self.assertGreaterEqual(score, 75)

    def test_identical_is_high(self):
        self.assertEqual(self.check(LOOP, LOOP)[1], "High")

    def test_different_algorithms_are_low(self):
        score, cat, _, _ = self.check(A, BUBBLE)
        self.assertEqual(cat, "Low")
        self.assertLess(score, 35)

    def test_symmetric(self):
        s1 = self.check(LOOP, BUBBLE)[0]
        s2 = self.check(BUBBLE, LOOP)[0]
        self.assertAlmostEqual(s1, s2, places=6)


if __name__ == "__main__":
    unittest.main()
