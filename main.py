import scipy.integrate as integrate
import math
import numpy as np

# Task data
Mx = 15.
My = 25.
sigma_x = 1.
sigma_y = 0.5
X_arr = range(13, 17)
r = -0.9
Y_threshold = My - sigma_x

sigma_coefs = list(set(np.array([*map(lambda x: [-x, x], [0, 1, 2, 3, 4, 5, 10, 15])]).flatten()))
sigma_coefs.sort()
sigma_label_parts = list(map(
    lambda coef: (' + ' if coef > 0 else ' - ') + str(abs(round(coef * 0.2, 1))) + ' * sigma_y',
    sigma_coefs
))
get_sigma_coef_labels = lambda expression: list(map(lambda x:
                                           expression + (x if x != ' - 0.0 * sigma_y' else ''),
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
    print(get_P_K1(1, 2))
    print(get_P_K2(1, 2))
    print(get_M_x_if_class_is_K1(1, 2))
    print(get_M_x_if_class_is_K2(1, 2))


if __name__ == '__main__':
    main()

# https://stud.com.ua/151074/ekonomika/dvovimirniy_mirniy_normalniy_zakon_rozpodilu
