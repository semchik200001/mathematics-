import numpy as np
import matplotlib.pyplot as plt

class ODESolver:
    def __init__(self, f, exact=None):
        self.f = f       # function f(x,y)
        self.exact = exact  # exact solution y(x)

    def euler(self, x0, y0, xn, h):
        n = int((xn - x0)/h)
        xs = np.linspace(x0, x0 + n*h, n+1)
        ys = np.zeros(n+1)
        ys[0] = y0
        for i in range(n):
            ys[i+1] = ys[i] + h * self.f(xs[i], ys[i])
        return xs, ys

    def rk4(self, x0, y0, xn, h):
        n = int((xn - x0)/h)
        xs = np.linspace(x0, x0 + n*h, n+1)
        ys = np.zeros(n+1)
        ys[0] = y0
        for i in range(n):
            x, y = xs[i], ys[i]
            k1 = self.f(x, y)
            k2 = self.f(x + h/2, y + h*k1/2)
            k3 = self.f(x + h/2, y + h*k2/2)
            k4 = self.f(x + h, y + h*k3)
            ys[i+1] = y + h*(k1 + 2*k2 + 2*k3 + k4)/6
        return xs, ys

    def adams_pc(self, x0, y0, xn, h):
        # 4-step Adams-Bashforth predictor, Adams-Moulton corrector
        xs_rk, ys_rk = self.rk4(x0, y0, x0 + 3*h, h)
        xs = list(xs_rk)
        ys = list(ys_rk)
        n = int((xn - x0)/h)
        for i in range(3, n):
            x_vals = xs[i-3:i+1]
            y_vals = ys[i-3:i+1]
            f_vals = [self.f(x_vals[j], y_vals[j]) for j in range(4)]
            # predictor (Adams-Bashforth)
            y_pred = ys[i] + h*(55*f_vals[3] - 59*f_vals[2] + 37*f_vals[1] - 9*f_vals[0])/24
            x_next = xs[i] + h
            # corrector (Adams-Moulton)
            f_pred = self.f(x_next, y_pred)
            y_corr = ys[i] + h*(9*f_pred + 19*f_vals[3] - 5*f_vals[2] + f_vals[1])/24
            xs.append(x_next)
            ys.append(y_corr)
        return np.array(xs), np.array(ys)

    def runge_error(self, method, p, *args):
        # compute error estimate via Runge rule
        xs1, ys1 = method(*args)
        h = args[3]
        args_half = (args[0], args[1], args[2], h/2)
        xs2, ys2 = method(*args_half)
        # align: ys2 at every 2nd step
        ys2_sub = ys2[::2]
        err = np.max(np.abs((ys2_sub - ys1)/(2**p - 1)))
        return err


def select_problem():
    print("Выберите ОДУ из списка:")
    print("1) y' = y, y(0) = 1, точное: y = exp(x)")
    print("2) y' = x, y(0) = 0, точное: y = x^2/2")
    print("3) y' = x*y, y(0) = 1, точное: y = exp(x^2/2)")
    choice = int(input("Номер задачи: "))
    if choice == 1:
        return lambda x,y: y, lambda x: np.exp(x), 0, 1
    if choice == 2:
        return lambda x,y: x, lambda x: x**2/2, 0, 0
    if choice == 3:
        return lambda x,y: x*y, lambda x: np.exp(x**2/2), 0, 1
    raise ValueError("Неправильный выбор")

if __name__ == '__main__':
    f, exact, x0, y0 = select_problem()
    xn = float(input("Введите x_n (конец отрезка): "))
    h = float(input("Введите шаг h: "))
    eps = float(input("Введите точность ε для оценки (например, 1e-3): "))

    solver = ODESolver(f, exact)

    # вычисления
    xe, ye = solver.euler(x0, y0, xn, h)
    xr, yr = solver.rk4(x0, y0, xn, h)
    xa, ya = solver.adams_pc(x0, y0, xn, h)

    # оценки погрешности
    err_euler = solver.runge_error(solver.euler, 1, x0, y0, xn, h)
    err_rk4 = solver.runge_error(solver.rk4, 4, x0, y0, xn, h)
    true_vals = exact(xr)
    err_adams = np.max(np.abs(true_vals - ya))

    # вывод таблиц
    print("\nТаблица решений:")
    print("i\tx\tEuler\tRK4\tAdams\tExact")
    for i, x in enumerate(xe):
        print(f"{i}\t{x:.4f}\t{ye[i]:.6f}\t{yr[i]:.6f}\t{ya[i]:.6f}\t{exact(x):.6f}")

    print(f"\nОценка погрешности Эйлера (Рунге): {err_euler:.2e}")
    print(f"Оценка погрешности RK4 (Рунге):    {err_rk4:.2e}")
    print(f"Максимальная погрешность Адамса:  {err_adams:.2e}")

    # график
    xs = np.linspace(x0, xn, 200)
    plt.plot(xs, exact(xs), label='Exact')
    plt.plot(xe, ye, 'o-', label='Euler')
    plt.plot(xr, yr, 's-', label='RK4')
    plt.plot(xa, ya, 'd-', label='Adams')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Сравнение методов")
    plt.grid(True)
    plt.show()

    print("\nГотово. Проверьте, что шаг и точность дают приемлемый результат.")
