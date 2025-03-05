import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, sin, cos, lambdify


def get_input(prompt, type_func=float, from_file=None):
    if from_file:
        value = from_file.readline().strip()
        print(f"{prompt} {value}")
        return type_func(value)
    while True:
        try:
            return type_func(input(prompt))
        except ValueError:
            print("Некорректный ввод. Попробуйте снова.")


def plot_function(f, a, b, root):
    x_vals = np.linspace(a, b, 400)
    y_vals = [f(x) for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="f(x)", color="blue")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(root, color="red", linestyle="--", label="Найденный корень")
    plt.grid()
    plt.legend()
    plt.title("График функции")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def bisection_method(f, a, b, tol):
    if f(a) * f(b) >= 0:
        print("Некорректный интервал: f(a) и f(b) должны иметь разные знаки.")
        return None, []

    iterations = []
    while abs(b - a) > tol:
        c = (a + b) / 2
        iterations.append([a, b, c, f(a), f(b), f(c), abs(b - a)])

        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return c, iterations


def secant_method(f, x0, x1, tol):
    iterations = []
    while abs(x1 - x0) > tol:
        if f(x1) - f(x0) == 0:
            print("Ошибка: Деление на ноль в методе секущих.")
            return None, []
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        iterations.append([x0, x1, f(x0), f(x1), abs(x1 - x0)])
        x0, x1 = x1, x2
    return x1, iterations


def newton_method(f, df, x0, tol):
    iterations = []
    while True:
        if df(x0) == 0:
            print("Ошибка: Производная равна нулю, метод Ньютона не применим.")
            return None, []
        x1 = x0 - f(x0) / df(x0)
        iterations.append([x0, f(x0), df(x0), x1, abs(x1 - x0)])
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x1, iterations


def simple_iteration_method(g, x0, tol):
    iterations = []
    while True:
        x1 = g(x0)
        iterations.append([x0, x1, abs(x1 - x0)])
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x1, iterations


def solve_nonlinear_equation():
    x = symbols('x')
    functions = {
        1: sin(x) - x / 2,
        2: x ** 3 - 4 * x + 1,
        3: x ** 2 - 2,
    }

    from_file = None
    if input("Ввести данные из файла? (y/n): ").strip().lower() == 'y':
        filename = input("Введите имя файла: ").strip()
        from_file = open(filename, "r")

    print("Выберите уравнение:")
    for k, v in functions.items():
        print(f"{k}: {v}")
    choice = get_input("Введите номер уравнения: ", int, from_file)
    if choice not in functions:
        print("Некорректный выбор уравнения.")
        return

    f_sym = functions[choice]
    f = lambdify(x, f_sym, 'numpy')
    df = lambdify(x, diff(f_sym, x), 'numpy')

    method = input("Выберите метод (bisection, secant, newton, iteration): ").strip().lower()
    if method not in ["bisection", "secant", "newton", "iteration"]:
        print("Некорректный метод.")
        return

    tol = get_input("Введите точность: ", float, from_file)

    if method == "bisection":
        a = get_input("Введите левую границу: ", float, from_file)
        b = get_input("Введите правую границу: ", float, from_file)
        root, iterations = bisection_method(f, a, b, tol)
    else:
        x0 = get_input("Введите начальное приближение: ", float, from_file)
        if method == "secant":
            x1 = get_input("Введите x1: ", float, from_file)
            root, iterations = secant_method(f, x0, x1, tol)
        elif method == "newton":
            root, iterations = newton_method(f, df, x0, tol)
        elif method == "iteration":
            g = lambdify(x, x - f_sym, 'numpy')
            root, iterations = simple_iteration_method(g, x0, tol)
        a, b = root - 2, root + 2

    if from_file:
        from_file.close()

    if root is None:
        print("Ошибка вычислений. Попробуйте снова с другим интервалом или методом.")
        return

    function_value = f(root)
    result = f"Найденный корень: {root}\nЗначение функции в корне: {function_value}\nЧисло итераций: {len(iterations)}\n"
    print(result)

    save_to_file = input("Сохранить результат в файл? (y/n): ").strip().lower()
    if save_to_file == 'y':
        with open("result.txt", "w") as file:
            file.write(result)
        print("Результат сохранен в result.txt")

    plot_function(f, a, b, root)


def main():
    while True:
        mode = input("Выберите режим (equation/system) или 'exit' для выхода: ").strip().lower()
        if mode == "equation":
            solve_nonlinear_equation()
        elif mode == "system":
            print("Функционал для систем еще не реализован.")
        elif mode == "exit":
            break
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
