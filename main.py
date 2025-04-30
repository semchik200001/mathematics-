#!/usr/bin/env python3
"""
ЛР-5  (Интерполяция функции)
Вариант 6 — универсальная программа:
  • ввод данных тремя способами
  • таблица конечных разностей
  • методы: Лагранж, Ньютон (разд. и конечные Δ), Гаусс
  • графики и элементарная валидация
"""

import math, sys, json, pathlib
from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
def is_uniform(xs, rtol=1e-12, atol=1e-14):
    diffs = np.diff(xs)
    return np.allclose(diffs, diffs[0], rtol=rtol, atol=atol)
# ────────────────────────────────────  служебные функции ──────────────
def forward_differences(y: List[float]) -> List[List[float]]:
    """Полный треугольник конечных разностей Δᵏyᵢ"""
    table = [y]
    while len(table[-1]) > 1:
        prev = table[-1]
        table.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    return table

def print_table(table: List[List[float]]) -> None:
    w = max(len(table[0]), 7)
    for k, row in enumerate(table):
        print(f'Δ{k:>1} ', end='')
        print(' '.join(f'{v:>10.6f}' for v in row).ljust(w * 12))
    print()

# ───────────────────────────────  интерполяционные методы ─────────────
def lagrange(x: List[float], y: List[float], xp: float) -> float:
    s = 0.0
    for i in range(len(x)):
        li = 1.0
        for j in range(len(x)):
            if i != j:
                li *= (xp - x[j]) / (x[i] - x[j])
        s += y[i] * li
    return s

def newton_backward(x: List[float], y: List[float], xp: float) -> float:
    h = x[1] - x[0]
    u  = (xp - x[-1]) / h
    Δ  = forward_differences(y)
    res = y[-1]
    term, fact = 1.0, 1
    for k in range(1, len(x)):
        term *= (u + k - 1)
        fact *= k
        res  += term * Δ[k][-1] / fact
    return res

def newton_forward(x: List[float], y: List[float], xp: float) -> float:
    h = x[1] - x[0]
    u  = (xp - x[0]) / h
    Δ  = forward_differences(y)
    res = y[0]
    term, fact = 1.0, 1
    for k in range(1, len(x)):
        term *= (u - (k - 1))
        fact *= k
        res  += term * Δ[k][0] / fact
    return res

def gauss_second_backward(x: List[float], y: List[float], xp: float) -> float:
    """Формула Гаусса 2-я, 'назад' (центр — средняя точка)"""
    n   = len(x)
    m   = n // 2                # индекс центра
    h   = x[1] - x[0]
    t   = (xp - x[m]) / h
    Δ   = forward_differences(y)

    # удобные ссылки на нужные ∇-разности
    def nabla(k: int, j: int) -> float:
        """∇ᵏ y_{j} (j отсчитывается от x₀)"""
        return Δ[k][j]

    res = y[m]

    res += t                  * nabla(1, m - 1)
    res += t*(t + 1)/2        * nabla(2, m - 1)
    res += t*(t + 1)*(t - 1)/6* nabla(3, m - 2)
    res += t*(t + 1)*(t - 1)*(t - 2)/24 * nabla(4, m - 2)
    res += t*(t + 1)*(t - 1)*(t - 2)*(t + 2)/120 * nabla(5, m - 3)
    res += t*(t + 1)*(t - 1)*(t - 2)*(t + 2)*(t + 3)/720 * nabla(6, m - 3)

    return res

# ────────────────────────────────  источник данных ────────────────────
def source_keyboard() -> Tuple[List[float], List[float]]:
    n = int(input("Сколько узлов? > "))
    x, y = [], []
    for i in range(n):
        xi = float(input(f"x[{i}] = "))
        yi = float(input(f"y[{i}] = "))
        x.append(xi);  y.append(yi)
    return x, y

def source_file(path: str) -> Tuple[List[float], List[float]]:
    arr = np.loadtxt(path, delimiter=',')
    return arr[:, 0].tolist(), arr[:, 1].tolist()

def source_function(func: Callable[[float], float]) -> Tuple[List[float], List[float]]:
    a = float(input("a = "))
    b = float(input("b = "))
    n = int(input("точек (≥2) = "))
    x = np.linspace(a, b, n)
    y = [func(t) for t in x]
    return x.tolist(), y

# ──────────────────────────────────  графика ──────────────────────────
def draw(x: List[float], y: List[float],
         poly: Callable[[float], float], title: str = "") -> None:
    xs = np.linspace(min(x), max(x), 400)
    plt.figure()
    plt.plot(xs, [poly(t) for t in xs], label='Полином', linewidth=1.2)
    plt.scatter(x, y, marker='o', zorder=3, label='Узлы')
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

# ───────────────────────────────  основная логика ─────────────────────
METHODS = {
    1: ("Лагранж",           lagrange),
    2: ("Ньютон (разд.)",    newton_forward),  # для разн. h программа сама решит F/B
    3: ("Ньютон (Δ, назад)", newton_backward),
}

def run() -> None:
    print("===== Лабораторная работа 5.  Выбор источника данных =====")
    print("1 — Ввод с клавиатуры")
    print("2 — Чтение CSV-файла (x,y)")
    print("3 — Таблица сгенерирована по функции")
    choice = input("> ")

    if choice == "1":
        x, y = source_keyboard()
    elif choice == "2":
        fname = input("Имя файла > ")
        x, y = source_file(fname)
    elif choice == "3":
        funcs = {"sin":  math.sin,
                 "cos":  math.cos,
                 "exp":  math.exp}
        print("Доступные функции:", *funcs)
        fkey = input("> ")
        x, y = source_function(funcs[fkey])
    else:
        sys.exit("неизвестный выбор")

    if not is_uniform(x):
        sys.exit("Для формул с Δ требуется равномерная сетка!")

    print("\n— Таблица конечных разностей —")
    table = forward_differences(y)
    print_table(table)

    xp = float(input("x*, где вычислять полином = "))

    print("\nМетод  значение")
    for num, (name, fn) in METHODS.items():
        try:
            fx = fn(x, y, xp)
            print(f"{num:>2}. {name:<20} {fx:>12.6f}")
        except Exception as e:
            print(f"{num:>2}. {name:<20} ошибка: {e}")

    # пример графика с полиномом Ньютона (Δ, назад),
    # если xp внутри диапазона
    if min(x) <= xp <= max(x):
        draw(x, y, lambda t: newton_backward(x, y, t),
             f"Интерполяция (метод 3)  —  x*={xp}")

if __name__ == "__main__":
    run()
