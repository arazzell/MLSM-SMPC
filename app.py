import cvxpy as cp
import numpy as np

N = 1000

n = 10
m = 5
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

xi = np.random.randn(N, n)

x = cp.Variable(n)
x_expanded = cp.reshape(x, (1, n))
diff = x_expanded - xi
objective = cp.Minimize(cp.max(cp.sum(cp.square(diff), axis=1)))
constraints = [A @ x <= b]
prob = cp.Problem(objective, constraints)

prob.solve()

print("Optimal value: ", prob.value)
print("Optimal solution: ", x.value)