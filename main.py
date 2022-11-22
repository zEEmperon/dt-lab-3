from tabulate import tabulate
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import numpy as np

# Task data
Mx = 15.
My = 25.
sigma_x = 1.
sigma_y = 0.5
r = -0.9
Y_threshold = My - sigma_x

sigma_coefs = list(set(np.array([*map(lambda x: [-x, x], [0, 1, 2, 3, 4, 5, 10, 15])]).flatten()))
sigma_coefs.sort()


def get_sigma_coef_labels(first_expr, second_expr):
    sigma_label_parts = list(map(
        lambda coef: (' + ' if coef > 0 else ' - ') + str(abs(round(coef * 0.2, 1))) + ' * ' + second_expr,
        sigma_coefs
    ))
    return list(map(lambda x:
                    first_expr + (x if x != ' - 0.0 * ' + second_expr else ''),
                    sigma_label_parts))


def get_W_x_y(x, y):
    a = 1 / (2 * math.pi * sigma_x * sigma_y * math.sqrt(1 - r ** 2))
    b = (1 / (2 * (1 - r ** 2))) * (
            ((x - Mx) / sigma_x) ** 2 - (2 * r * ((x - Mx) / sigma_x) * ((y - My) / sigma_y)) + (
            ((y - My) / sigma_y) ** 2))
    return a * math.exp(-b)


def get_W_y_given_x(x, y):
    a = 1 / (sigma_y * math.sqrt(2 * math.pi * (1 - r ** 2)))
    b = (1 / (2 * sigma_y ** 2 * (1 - r ** 2))) * (y - My * ((sigma_y * (x - Mx)) / sigma_x)) ** 2
    return a * math.exp(-b)


def get_W_x(x):
    a = (1 / sigma_x * math.sqrt(2 * math.pi))
    b = (x - Mx) ** 2 / (2 * sigma_x ** 2)
    return a * math.exp(-b)


def get_P_K1(x, y):
    y_lower = My - 3 * sigma_y
    y_upper = Y_threshold
    x_lower = Mx - 3 * sigma_x
    x_upper = Mx + 3 * sigma_x
    return integrate.nquad(get_W_x_y, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_P_K2(x, y):
    y_lower = Y_threshold
    y_upper = My + 3 * sigma_y * Y_threshold
    x_lower = Mx - 3 * sigma_x
    x_upper = Mx + 3 * sigma_x
    return integrate.nquad(get_W_x_y, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_M_x_if_class_is_K1(x, y):
    y_lower = My - 3 * sigma_y
    y_upper = Y_threshold
    x_lower = Mx - 3 * sigma_x
    x_upper = Mx + 3 * sigma_x
    fun = lambda y_arg, x_arg: x_arg * get_W_y_given_x(x_arg, y_arg)
    return integrate.nquad(fun, [[y_lower, y_upper], [x_lower, x_upper]])[0]


def get_M_x_if_class_is_K2(x, y):
    y_lower = Y_threshold
    y_upper = My + 3 * sigma_y * Y_threshold
    x_lower = Mx - 3 * sigma_x
    x_upper = Mx + 3 * sigma_x
    fun = lambda y_arg, x_arg: x_arg * get_W_y_given_x(x_arg, y_arg)
    return integrate.nquad(fun, [[y_lower, y_upper], [x_lower, x_upper]])[0]


def main():
    # W(x)
    label = "Безумовна густина розподілу ознаки х"
    x_arr = [*map(lambda coef: Mx + coef * sigma_x, sigma_coefs)]
    w_x_arr = [*map(lambda x: get_W_x(x), x_arr)]

    x_labels = get_sigma_coef_labels("Mx", "sigma_x")
    col_names = ["Вираз х", "Значення x", "W(x)"]
    table_data = np.vstack((x_labels, x_arr, w_x_arr)).T

    print()
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    plt.plot(x_arr, w_x_arr)
    plt.xlabel("x")
    plt.ylabel("W(x)")
    plt.title(label)
    plt.show()


if __name__ == '__main__':
    main()

# https://stud.com.ua/151074/ekonomika/dvovimirniy_mirniy_normalniy_zakon_rozpodilu
