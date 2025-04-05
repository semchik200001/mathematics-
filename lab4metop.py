import numpy as np
from scipy.optimize import minimize_scalar

# Целевая функция
f = lambda x: 6*x[0]**2 + x[1]**2 - x[0]*x[1] + 4*x[0] - 8*x[1] + 1

# Градиент функции
def grad_f(x):
    df_dx1 = 12*x[0] - x[1] + 4
    df_dx2 = 2*x[1] - x[0] - 8
    return np.array([df_dx1, df_dx2])

# Округление до 4 значащих цифр
def round_significant(x, digits=4):
    return np.array([float(f"{xi:.{digits}g}") for xi in x])

# Метод покоординатного спуска
def coordinate_descent(x0, eps=1e-4):
    x = np.array(x0, dtype=float)
    iterations = 0
    while True:
        x_prev = x.copy()
        iterations += 1

        # Минимизация по x1
        f1 = lambda x1: f([x1, x[1]])
        res1 = minimize_scalar(f1)
        x[0] = res1.x

        # Минимизация по x2
        f2 = lambda x2: f([x[0], x2])
        res2 = minimize_scalar(f2)
        x[1] = res2.x

        if np.linalg.norm(x - x_prev) < eps:
            break

    print(f"Coordinate Descent iterations: {iterations}")
    return round_significant(x)

# Метод градиентного спуска
def gradient_descent(x0, alpha=0.05, eps=1e-4):
    x = np.array(x0, dtype=float)
    iterations = 0
    while True:
        x_prev = x.copy()
        iterations += 1
        g = grad_f(x)
        x = x - alpha * g
        if np.linalg.norm(x - x_prev) < eps:
            break
    print(f"Gradient Descent iterations: {iterations}")
    return round_significant(x)

# Метод наискорейшего спуска
def steepest_descent(x0, eps=1e-4):
    x = np.array(x0, dtype=float)
    iterations = 0
    while True:
        x_prev = x.copy()
        iterations += 1
        g = grad_f(x)
        direction = -g

        # Одномерная минимизация вдоль направления
        phi = lambda alpha: f(x + alpha * direction)
        res = minimize_scalar(phi)
        alpha = res.x

        x = x + alpha * direction
        if np.linalg.norm(x - x_prev) < eps:
            break
    print(f"Steepest Descent iterations: {iterations}")
    return round_significant(x)

# Тестирование
x0 = [2, 2]
print("Coordinate Descent result:", coordinate_descent(x0))
print("Gradient Descent result:", gradient_descent(x0))
print("Steepest Descent result:", steepest_descent(x0))

