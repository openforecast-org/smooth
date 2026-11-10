"""Hand-checkable tests of the exact oracle (run: python3 -m unittest test_oracle)."""

import itertools
import random
import unittest
from fractions import Fraction

from flint import fmpq_poly

import oracle as O


def poly_fraction(coeffs):
    """fmpq_poly from a list of Fractions."""
    return fmpq_poly([O.fmpq(c.numerator, c.denominator) for c in coeffs])


def det_fraction(A):
    """Determinant of a square matrix of Fractions by Gaussian elimination."""
    A = [row[:] for row in A]
    n, det = len(A), Fraction(1)
    for p in range(n):
        piv = next((i for i in range(p, n) if A[i][p] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != p:
            A[p], A[piv] = A[piv], A[p]
            det = -det
        det *= A[p][p]
        for i in range(p + 1, n):
            f = A[i][p] / A[p][p]
            for j in range(p, n):
                A[i][j] -= f * A[p][j]
    return det


def minor_expansion(M, lags):
    """det(I - M L(B)) by the sum over principal minors (claim C20)."""
    k = len(lags)
    c = [Fraction(0)] * (sum(lags) + 1)
    for r in range(k + 1):
        for S in itertools.combinations(range(k), r):
            sub = [[M[i][j] for j in S] for i in S]
            c[sum(lags[j] for j in S)] += (-1) ** r * (det_fraction(sub) if S else 1)
    return poly_fraction(c)


def expanded(M, lags):
    """The lag-1 expansion of v_t = M v_{t-l} (D2): component j with l_j copies."""
    start = list(itertools.accumulate([0] + lags[:-1]))
    n = sum(lags)
    T = [[Fraction(0)] * n for _ in range(n)]
    for i in range(len(lags)):
        for j in range(len(lags)):
            T[start[i]][start[j] + lags[j] - 1] = M[i][j]  # head of i from last copy of j
        for s in range(1, lags[i]):
            T[start[i] + s][start[i] + s - 1] = Fraction(1)  # copy s from copy s - 1
    return T


def frac_matrix(rng, k, scale):
    return [[Fraction(rng.randint(-40, 40), scale) for _ in range(k)] for _ in range(k)]


class TestDeterminant(unittest.TestCase):
    def test_bareiss_equals_minor_expansion(self):
        rng = random.Random(1)
        for k, lags in [(1, [3]), (2, [1, 4]), (3, [1, 1, 12]), (4, [1, 2, 3, 5]), (5, [1, 1, 4, 4, 7])]:
            M = frac_matrix(rng, k, 17)
            Mq = [[O.fmpq(x.numerator, x.denominator) for x in row] for row in M]
            self.assertEqual(O.lag_det(Mq, lags), minor_expansion(M, lags))

    def test_characteristic_polynomial_of_expansion(self):
        # det(lam I - M~) = lam^(sum l) det(I - M L(1/lam)) (claim C15), exactly at lam = 3/7
        rng = random.Random(2)
        lam = Fraction(3, 7)
        for lags in [[1, 3], [1, 1, 4], [2, 3, 1]]:
            M = frac_matrix(rng, len(lags), 11)
            T = expanded(M, lags)
            n = len(T)
            lhs = det_fraction([[(lam if i == j else 0) - T[i][j] for j in range(n)] for i in range(n)])
            theta = O.lag_det([[O.fmpq(x.numerator, x.denominator) for x in row] for row in M], lags)
            val = sum(Fraction(int(c.p), int(c.q)) * (1 / lam) ** e for e, c in enumerate(theta.coeffs()))
            self.assertEqual(lhs, lam ** n * val)

    def test_ets_ana_by_hand(self):
        # theta = (1 - (1-a)B)(1 - (1-c)B^m) - a c B^(m+1), phi = (1 - B)(1 - B^m) (E9)
        a, c, m = 0.3, 0.45, 5
        o = O.oracle({"index": 0, "family": "t", "F": [[1.0, 0.0], [0.0, 1.0]],
                      "g": [a, c], "w": [1.0, 1.0], "lags": [1, m]})
        A, Cq = Fraction(a), Fraction(c)
        th = [Fraction(0)] * (m + 2)
        th[0] += 1
        th[1] -= 1 - A
        th[m] -= 1 - Cq
        th[m + 1] += (1 - A) * (1 - Cq) - A * Cq
        D = [[O.fmpq(1) - O.exact(a), -O.exact(a)], [-O.exact(c), O.fmpq(1) - O.exact(c)]]
        self.assertEqual(O.lag_det(D, [1, m]), poly_fraction(th))
        self.assertEqual(o["degree_theta"], m + 1)
        self.assertEqual(o["degree_cyclotomic"], 1)  # the root B = 1 shared with phi (C9)
        self.assertEqual(o["degree_noncyclotomic_shared"], 0)


class TestDecision(unittest.TestCase):
    def one(self, F, g, w, lags):
        return O.oracle({"index": 0, "family": "t", "F": F, "g": g, "w": w, "lags": lags})

    def test_local_level(self):
        # theta = 1 - (1 - a) B, phi = 1 - B: modulus |1 - a|
        for a, status, mod in [(2.5, "inadmissible", 1.5), (0.5, "admissible", 0.5), (1.75, "admissible", 0.75)]:
            o = self.one([[1.0]], [a], [1.0], [1])
            self.assertEqual(o["gcd"]["status"], status)
            self.assertLessEqual(o["gcd"]["lo"], mod)
            self.assertGreaterEqual(o["gcd"]["hi"], mod)
            self.assertLess(o["gcd"]["hi"] - o["gcd"]["lo"], 1e-15)

    def test_unreachable_unit_root_cancelled(self):
        # a = 0: theta = phi = 1 - B, cancelled by Phi_1, nothing left
        o = self.one([[1.0]], [0.0], [1.0], [1])
        self.assertEqual(o["degree_cyclotomic"], 1)
        self.assertEqual(o["cancelled"]["status"], "admissible")
        self.assertEqual(o["cancelled"]["hi"], 0.0)

    def test_roots_on_the_unit_circle_proved(self):
        # ETS(A,N,A), m = 3, alpha = gamma = 1: theta* = 1 + B + B^2 + B^3
        o = self.one([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0], [1.0, 1.0], [1, 3])
        self.assertEqual(o["degree_theta_cancelled"], 3)
        for r in (o["gcd"], o["cancelled"]):
            self.assertEqual(r["status"], "inadmissible")
            self.assertEqual(r["unit_roots"], 3)
            self.assertEqual((r["lo"], r["hi"]), (1.0, 1.0))
            self.assertTrue(O.near_edge(r))

    def test_noncyclotomic_shared_factor(self):
        # an unreachable state with F = 2 (L8): exact gcd cancels 1 - 2B, the cyclotomic cancellation keeps it
        o = self.one([[2.0, 0.0], [0.0, 1.0]], [0.0, 0.3], [1.0, 1.0], [1, 1])
        self.assertEqual(o["gcd"]["status"], "admissible")
        self.assertEqual(o["cancelled"]["status"], "inadmissible")
        self.assertEqual(o["shared"], "inside")
        # with F = 0.5 the shared factor's root B = 2 lies outside the unit disc
        o = self.one([[0.5, 0.0], [0.0, 1.0]], [0.0, 0.3], [1.0, 1.0], [1, 1])
        self.assertEqual(o["shared"], "outside")
        self.assertEqual(o["degree_noncyclotomic_shared"], 1)

    def test_close_to_the_circle(self):
        # modulus 1 - 2^-40 is separated from 1 by the enclosures
        a = 2.0 ** -40
        o = self.one([[1.0]], [a], [1.0], [1])
        self.assertEqual(o["gcd"]["status"], "admissible")
        self.assertTrue(O.near_edge(o["gcd"]))


if __name__ == "__main__":
    unittest.main()
