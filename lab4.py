import numpy as np
import math
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Функции для чтения данных
# ---------------------------------------------------------

def read_data_console():
    """
    Считывание данных из консоли.
    Пользователь вводит количество точек n (8..12),
    затем n пар (x_i, y_i).
    """
    print("Чтение данных ИЗ КОНСОЛИ.")
    print("Введите количество точек (8..12):")
    n = int(input().strip())
    if not (8 <= n <= 12):
        raise ValueError("Число точек должно быть в диапазоне [8..12].")

    x_vals = []
    y_vals = []
    print(f"Введите {n} пар (x_i, y_i), каждая в новой строке:")
    for _ in range(n):
        line = input().strip().split()
        if len(line) < 2:
            raise ValueError("Недостаточно значений в строке.")
        x, y = float(line[0]), float(line[1])
        x_vals.append(x)
        y_vals.append(y)

    return np.array(x_vals), np.array(y_vals)


def read_data_file():
    """
    Считывание данных из файла.
    Формат файла (пример):
    Первая строка: n (число точек)
    Далее n строк, в каждой по два числа: x_i, y_i
    """
    print("Чтение данных ИЗ ФАЙЛА.")
    print("Введите путь к файлу (например, data.txt):")
    filename = input().strip()

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().strip().splitlines()

    if not lines:
        raise ValueError("Пустой файл.")

    n = int(lines[0].strip())
    if not (8 <= n <= 12):
        raise ValueError("Число точек должно быть в диапазоне [8..12].")

    if len(lines) < n + 1:
        raise ValueError(f"В файле недостаточно строк. Ожидалось {n} пар данных.")

    x_vals, y_vals = [], []
    for i in range(n):
        parts = lines[i + 1].split()
        if len(parts) < 2:
            raise ValueError(f"Недостаточно значений в строке {i + 2} файла.")
        x, y = float(parts[0]), float(parts[1])
        x_vals.append(x)
        y_vals.append(y)

    return np.array(x_vals), np.array(y_vals)


def choose_data_input():
    """
    Предлагает пользователю выбрать способ ввода данных:
     1) из консоли,
     2) из файла.
    Возвращает (x_data, y_data).
    """
    print("Выберите способ ввода данных:")
    print(" [1] Консоль")
    print(" [2] Файл")
    choice = input("Ваш выбор: ").strip()

    if choice == '1':
        return read_data_console()
    elif choice == '2':
        return read_data_file()
    else:
        raise ValueError("Некорректный выбор способа ввода данных.")


# ---------------------------------------------------------
# Модели аппроксимации (МНК).
# ---------------------------------------------------------
def fit_linear(x, y):
    """ Линейная: f(x) = a + b*x """
    # polyfit вернет [b, a]
    b, a = np.polyfit(x, y, deg=1)
    return (a, b)


def linear_func(x, coeffs):
    a, b = coeffs
    return a + b * x


def fit_poly2(x, y):
    """ Полином 2-й степени: f(x)= a + b*x + c*x^2 """
    c, b, a = np.polyfit(x, y, deg=2)
    return (a, b, c)


def poly2_func(x, coeffs):
    a, b, c = coeffs
    return a + b * x + c * (x ** 2)


def fit_poly3(x, y):
    """ Полином 3-й степени: f(x)= a + b*x + c*x^2 + d*x^3 """
    d, c, b, a = np.polyfit(x, y, deg=3)
    return (a, b, c, d)


def poly3_func(x, coeffs):
    a, b, c, d = coeffs
    return a + b * x + c * (x ** 2) + d * (x ** 3)


def fit_exponential(x, y):
    """
    Экспоненциальная: f(x) = A*exp(B*x)
    Нужно y>0.
    """
    if any(val <= 0 for val in y):
        raise ValueError("Невозможно экспоненциальное приближение: y <= 0.")
    ln_y = np.log(y)
    B, lnA = np.polyfit(x, ln_y, deg=1)
    A = math.exp(lnA)
    return (A, B)


def exponential_func(x, coeffs):
    A, B = coeffs
    return A * np.exp(B * x)


def fit_logarithmic(x, y):
    """
    Логарифмическая: f(x)= a + b*ln(x).
    Нужно x>0.
    """
    if any(val <= 0 for val in x):
        raise ValueError("Невозможно логарифмическое приближение: x <= 0.")
    ln_x = np.log(x)
    b, a = np.polyfit(ln_x, y, deg=1)
    return (a, b)


def logarithmic_func(x, coeffs):
    a, b = coeffs
    return a + b * np.log(x)


def fit_power(x, y):
    """
    Степенная: f(x)= A*x^B.
    Нужно x>0, y>0.
    """
    if any(val <= 0 for val in x):
        raise ValueError("Невозможно степенное приближение: x <= 0.")
    if any(val <= 0 for val in y):
        raise ValueError("Невозможно степенное приближение: y <= 0.")
    ln_x = np.log(x)
    ln_y = np.log(y)
    B, lnA = np.polyfit(ln_x, ln_y, deg=1)
    A = math.exp(lnA)
    return (A, B)


def power_func(x, coeffs):
    A, B = coeffs
    return A * (x ** B)


# ---------------------------------------------------------
# Оценки (СКО, R^2, Пирсон).
# ---------------------------------------------------------
def sse(x, y, func, coeffs):
    return np.sum((func(x, coeffs) - y) ** 2)


def rmse(x, y, func, coeffs):
    n = len(x)
    return math.sqrt(sse(x, y, func, coeffs) / n)


def determination_coefficient(x, y, func, coeffs):
    ss_res = sse(x, y, func, coeffs)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return 0
    return 1 - (ss_res / ss_tot)


def pearson_correlation(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = math.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    return num / den if den != 0 else 0


def interpret_r2(r2):
    if r2 < 0.3:
        return "Очень слабая зависимость (R^2 < 0.3)."
    elif r2 < 0.5:
        return "Слабая зависимость (0.3 <= R^2 < 0.5)."
    elif r2 < 0.7:
        return "Зависимость средней силы (0.5 <= R^2 < 0.7)."
    elif r2 < 0.9:
        return "Достаточно сильная зависимость (0.7 <= R^2 < 0.9)."
    else:
        return "Очень сильная зависимость (R^2 >= 0.9)."


# ---------------------------------------------------------
# Набор рассматриваемых моделей
# ---------------------------------------------------------
MODELS = [
    ("Линейная", fit_linear, linear_func),
    ("Полиномиальная 2", fit_poly2, poly2_func),
    ("Полиномиальная 3", fit_poly3, poly3_func),
    ("Экспоненциальная", fit_exponential, exponential_func),
    ("Логарифмическая", fit_logarithmic, logarithmic_func),
    ("Степенная", fit_power, power_func),
]


# ---------------------------------------------------------
# Рисование итогового графика
# ---------------------------------------------------------
def plot_results(x, y, results):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='black', label='Исходные точки')

    x_min, x_max = min(x), max(x)
    dx = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    x_plot = np.linspace(x_min - dx, x_max + dx, 300)

    for (name, coeffs, func_obj, this_rmse, this_r2) in results:
        # Лог/степень -> x>0
        if "Логарифмическая" in name or "Степенная" in name:
            x_pos = x_plot[x_plot > 0]
            if len(x_pos) == 0:
                continue
            y_plot = func_obj(x_pos, coeffs)
            plt.plot(x_pos, y_plot, label=name)
        else:
            y_plot = func_obj(x_plot, coeffs)
            plt.plot(x_plot, y_plot, label=name)

    plt.title("Сравнение аппроксимаций")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------------------------------------
# Вспомогательная функция для вывода текста и (опционально) записи в файл
# ---------------------------------------------------------
def print_to_console_and_file(text, file_obj=None, end="\n"):
    """
    Выводит строку text в консоль (print) и дублирует её в файл file_obj,
    если file_obj не None.
    """
    print(text, end=end)
    if file_obj is not None:
        file_obj.write(text + end)


# ---------------------------------------------------------
# Основная программа
# ---------------------------------------------------------
def main():
    # 1) Читаем данные
    try:
        x_data, y_data = choose_data_input()
    except Exception as e:
        print(f"Ошибка при вводе данных: {e}")
        return

    # 2) Обрабатываем все модели
    results = []  # (model_name, coeffs, func, rmse_val, r2_val)

    for model_name, fit_f, func_f in MODELS:
        try:
            coeffs = fit_f(x_data, y_data)
            curr_rmse = rmse(x_data, y_data, func_f, coeffs)
            curr_r2 = determination_coefficient(x_data, y_data, func_f, coeffs)
            results.append((model_name, coeffs, func_f, curr_rmse, curr_r2))
        except Exception as exc:
            # Не удалось применить модель (напр., логарифм при x<=0)
            print(f"[Предупреждение] Модель «{model_name}» не подходит: {exc}")

    if not results:
        print("Не удалось построить ни одной аппроксимации.")
        return

    # 2б) Спросим у пользователя, хочет ли он сохранить результаты в файл.
    save_to_file = False
    outfile = None
    print("Хотите ли сохранить результаты в файл? (y/n)")
    ans = input().strip().lower()
    if ans == 'y':
        print("Введите имя файла для сохранения (например, output.txt):")
        out_filename = input().strip()
        try:
            outfile = open(out_filename, 'w', encoding='utf-8')
            save_to_file = True
        except Exception as e:
            print(f"Не удалось открыть файл на запись: {e}")

    # 3) Вывод результатов в консоль (и файл, если нужно)
    best_model, best_rmse, best_coeffs, best_func = None, float('inf'), None, None

    print_to_console_and_file("\nРезультаты аппроксимации:", outfile)

    for (model_name, coeffs, func_f, curr_rmse, curr_r2) in results:
        print_to_console_and_file("", outfile)  # пустая строка
        print_to_console_and_file(f"Модель: {model_name}", outfile)
        print_to_console_and_file(f"  Коэффициенты: {coeffs}", outfile)
        print_to_console_and_file(f"  RMSE (СКО): {curr_rmse:.6f}", outfile)
        print_to_console_and_file(f"  R^2       : {curr_r2:.6f}", outfile)
        print_to_console_and_file(f"   -> {interpret_r2(curr_r2)}", outfile)

        # Линейная — Пирсон
        if model_name == "Линейная":
            r_xy = pearson_correlation(x_data, y_data)
            print_to_console_and_file(f"  Коэффициент корреляции Пирсона: r_xy = {r_xy:.6f}", outfile)

        # Минимальный RMSE -> лучшая модель
        if curr_rmse < best_rmse:
            best_rmse = curr_rmse
            best_model = model_name
            best_coeffs = coeffs
            best_func = func_f

    print_to_console_and_file("", outfile)
    print_to_console_and_file(f"Лучшая модель по критерию минимума RMSE: {best_model}", outfile)

    # 4) Остатки
    print_to_console_and_file("", outfile)
    print_to_console_and_file(f"Остатки eps_i = f(x_i) - y_i для лучшей модели:", outfile)
    fvals = best_func(x_data, best_coeffs)
    for i, (xx, yy, fv) in enumerate(zip(x_data, y_data, fvals), start=1):
        eps = fv - yy
        line = (f"  Точка {i}: x={xx}, y={yy}, "
                f"f(x)={fv:.5f}, eps={eps:.5f}")
        print_to_console_and_file(line, outfile)

    # Закроем файл, если нужно
    if outfile is not None:
        outfile.close()
        print("\nРезультаты также сохранены в файл:", out_filename)

    # 5) Построение единого графика
    plot_results(x_data, y_data, results)


# ---------------------------------------------------------
# Запуск
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
