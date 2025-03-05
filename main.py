import numpy as np
import matplotlib.pyplot as plt

def f1(x, y):
    return np.sin(x) + y - 1

def f2(x, y):
    return x**2 + y**2 - 4

def g1(x, y):
    return np.cos(y) - x

def g2(x, y):
    return y - np.exp(x) + 1

def phi1_1(x, y):
    return 1 - np.sin(x)

def phi1_2(x, y):
    return np.sqrt(4 - x**2) if x**2 <= 4 else y

def phi2_1(x, y):
    return np.cos(y)

def phi2_2(x, y):
    return np.exp(x) - 1

def simple_iteration(x0, y0, phi1, phi2, tol=1e-6, max_iter=100):
    x, y = x0, y0
    values = [(x, y)]
    for k in range(max_iter):
        x_new, y_new = phi1(x, y), phi2(x, y)
        error = np.sqrt((x_new - x)**2 + (y_new - y)**2)
        values.append((x_new, y_new))
        print(f"Iteration {k+1}: x = {x_new:.6f}, y = {y_new:.6f}, error = {error:.6e}")
        if error < tol:
            return x_new, y_new, k+1, values
        x, y = x_new, y_new
    print("Предупреждение: Метод не сошелся за указанное число итераций!")
    return x, y, max_iter, values

def plot_functions(f1, f2):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z1 = f1(X, Y)
    Z2 = f2(X, Y)
    plt.contour(X, Y, Z1, levels=[0], colors='r')
    plt.contour(X, Y, Z2, levels=[0], colors='b')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Графики уравнений")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    while True:
        mode = input("Выберите режим (equation/system) или 'exit' для выхода: ").strip().lower()
        if mode == 'exit':
            break
        elif mode == 'system':
            print("Выберите систему уравнений:")
            print("1. sin(x) + y = 1 и x^2 + y^2 = 4")
            print("2. cos(y) - x = 0 и y - exp(x) + 1 = 0")
            choice = input("Введите номер системы (1 или 2): ").strip()
            if choice == '1':
                f_1, f_2, phi_1, phi_2 = f1, f2, phi1_1, phi1_2
            elif choice == '2':
                f_1, f_2, phi_1, phi_2 = g1, g2, phi2_1, phi2_2
            else:
                print("Некорректный выбор.")
                continue
            try:
                x0, y0 = map(float, input("Введите начальные приближения x0 и y0 через пробел: ").split())
                max_iter = int(input("Введите максимальное число итераций: "))
                x_sol, y_sol, iterations, values = simple_iteration(x0, y0, phi_1, phi_2, max_iter=max_iter)
                print(f"Решение найдено: x = {x_sol:.6f}, y = {y_sol:.6f} за {iterations} итераций")
                plot_functions(f_1, f_2)
            except ValueError as e:
                print(f"Ошибка: {e}")
            except Exception as e:
                print(f"Некорректный ввод: {e}")
        else:
            print("Неверный ввод. Попробуйте снова.")
