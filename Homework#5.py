import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_1d(N: int):
    """Solve u_xx = -pi^2 sin(pi x), u(0)=u(1)=0 using 2nd-order FDM."""
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)

    # Internal unknowns: u_1, ..., u_{N-1}
    x_internal = x[1:-1]
    m = N - 1

    # Matrix system:
    # u_{i-1} - 2u_i + u_{i+1} = -pi^2 h^2 sin(pi x_i)
    A = -2.0 * np.eye(m)
    A += np.eye(m, k=1)
    A += np.eye(m, k=-1)

    b = -(np.pi ** 2) * (h ** 2) * np.sin(np.pi * x_internal)

    u_internal = np.linalg.solve(A, b)

    # Add boundary values u_0 = u_N = 0
    u_numeric = np.zeros(N + 1)
    u_numeric[1:-1] = u_internal

    u_exact = np.sin(np.pi * x)
    abs_error = np.abs(u_numeric - u_exact)

    return x, u_numeric, u_exact, abs_error, A, b


Ns = [4, 8, 16]

for N in Ns:
    x, u_num, u_ex, err, A, b = solve_poisson_1d(N)

    print(f"\nN = {N}, h = {1.0 / N:.5f}")
    print("A =")
    print(A)
    print("b =")
    print(b)

    print("i      x_i        u_numeric      u_exact        abs_error")
    for i in range(N + 1):
        print(f"{i:2d}  {x[i]:9.5f}  {u_num[i]:13.8f}  {u_ex[i]:13.8f}  {err[i]:13.8e}")

    print(f"max abs error = {err.max():.8e}")


# Plot result
x_fine = np.linspace(0.0, 1.0, 400)
u_fine = np.sin(np.pi * x_fine)

plt.figure(figsize=(8, 5))
plt.plot(x_fine, u_fine, label="Exact solution: sin(pi x)", linewidth=2)

for N in Ns:
    x, u_num, _, _, _, _ = solve_poisson_1d(N)
    plt.plot(x, u_num, marker="o", linestyle="--", label=f"FDM N={N}")

plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("1D Poisson Equation by 2nd-order Central Difference FDM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()