"""
Gauss Elimination with Partial Pivoting + Back-substitution
"""

import numpy as np


def gauss_elimination_with_pivoting(A, b):
    """
    Solve Ax = b using Gauss Elimination with Partial Pivoting.

    Parameters:
        A (list or np.ndarray): n x n coefficient matrix
        b (list or np.ndarray): n x 1 right-hand side vector

    Returns:
        x (np.ndarray): solution vector
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Augmented matrix [A | b]
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Forward Elimination with Partial Pivoting
    for col in range(n):
        # Find pivot: row with maximum absolute value in current column
        max_row = col + np.argmax(np.abs(Ab[col:, col]))

        if Ab[max_row, col] == 0:
            raise ValueError("Matrix is singular — no unique solution.")

        # Swap current row with pivot row
        if max_row != col:
            Ab[[col, max_row]] = Ab[[max_row, col]]
            print(f"  Pivot: swapped row {col} <-> row {max_row}")

        # Eliminate entries below the pivot
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]

    print("\n  Upper triangular matrix (augmented):")
    print(np.round(Ab, 6))

    # Back Substitution
    x = back_substitution(Ab[:, :n], Ab[:, n])
    return x


def back_substitution(U, b):
    """
    Solve Ux = b where U is upper triangular.
    """
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


if __name__ == "__main__":
    # ── Case 1: A_good (well-conditioned tridiagonal) ──────────────────────
    A_good = [
        [4.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [-1.0, 4.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  4.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0,  0.0, -1.0,  4.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0, -1.0,  4.0, -1.0,  0.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  0.0, -1.0,  4.0, -1.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  4.0, -1.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  4.0, -1.0,  0.0],
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  4.0, -1.0],
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  4.0],
    ]
    b_good = [3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0]

    # ── Case 2: A_bad (near-zero first pivot → tests pivoting necessity) ───
    A_bad = [
        [1.0e-20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0,     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0,     0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0,     0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0,     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0,     0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0,     0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0,     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0,     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0,     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    b_bad = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    for label, A, b in [("A_good / b_good", A_good, b_good),
                        ("A_bad  / b_bad ", A_bad,  b_bad)]:
        print("=" * 60)
        print(f"Gauss Elimination WITH Pivoting  --  {label}")
        print("=" * 60)
        print("\n--- Forward Elimination ---")
        x = gauss_elimination_with_pivoting(A, b)
        print("\n--- Solution ---")
        for i, xi in enumerate(x):
            print(f"  x{i + 1:02d} = {xi:.6f}")
        residual = np.linalg.norm(np.array(A, dtype=float) @ x - np.array(b))
        print(f"\n  Residual ||Ax - b|| = {residual:.2e}")
        print()
