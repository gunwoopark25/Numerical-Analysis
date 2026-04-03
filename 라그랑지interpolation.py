import numpy as np
import matplotlib.pyplot as plt

# 점 3개 입력 (예: 1 2)
x = []
y = []
for i in range(3):
    xi, yi = map(float, input(f"{i+1}번째 점 입력 (x y): ").split())
    x.append(xi)
    y.append(yi)

if len(set(x)) != 3:
    raise ValueError("x 값 3개는 서로 달라야 합니다.")

def L(i, t):
    value = 1.0
    for j in range(3):
        if i != j:
            value *= (t - x[j]) / (x[i] - x[j])
    return value

t = np.linspace(min(x) - 1, max(x) + 1, 400)

L0 = np.array([L(0, ti) for ti in t])
L1 = np.array([L(1, ti) for ti in t])
L2 = np.array([L(2, ti) for ti in t])

# basis function 그래프
plt.figure(figsize=(8, 5))
plt.plot(t, L0, label="L0(x)")
plt.plot(t, L1, label="L1(x)")
plt.plot(t, L2, label="L2(x)")
plt.axhline(0, color="gray", linewidth=0.8)
plt.title("Lagrange Basis Functions (3 points)")
plt.xlabel("x")
plt.ylabel("value")
plt.legend()
plt.grid(True)
plt.show()
