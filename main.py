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
r = 0.3
Y_threshold = My - sigma_y  # 24.5
X_arr = range(13, 17)

coefs_bases = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
sigma_coefs = list(set(np.array([*map(lambda x: [-x, x], coefs_bases)]).flatten()))
sigma_coefs.sort()

do_show_graphics = False


def print_task(no):
    print("Завдання {}:".format(no))


def get_X_general_limits():
    x_lower = Mx - 3 * sigma_x
    x_upper = Mx + 3 * sigma_x
    return x_lower, x_upper


def get_Y_general_limits():
    y_lower = My - 3 * sigma_y
    y_upper = My + 3 * sigma_y
    return y_lower, y_upper


def get_X_K1_limits():
    x_lower = 0  # by methods the Mx - 3 * sigma_x should be here
    x_upper = get_X_threshold()
    return x_lower, x_upper


def get_X_K2_limits():
    x_lower = get_X_threshold()
    x_upper = Mx + 3 * sigma_x
    return x_lower, x_upper


def get_Y_K1_limits():
    y_lower = My - 3 * sigma_y
    y_upper = Y_threshold
    return y_lower, y_upper


def get_Y_K2_limits():
    y_lower = Y_threshold
    y_upper = My + 3 * sigma_y  # * Y_threshold  # I think it is unneeded
    return y_lower, y_upper


def get_K1_limits():
    x_lower, x_upper = get_X_general_limits()
    y_lower, y_upper = get_Y_K1_limits()
    return x_lower, x_upper, y_lower, y_upper


def get_K2_limits():
    x_lower, x_upper = get_X_general_limits()
    y_lower, y_upper = get_Y_K2_limits()
    return x_lower, x_upper, y_lower, y_upper


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
            ((x - Mx) / sigma_x) ** 2 - (2 * r * ((x - Mx) / sigma_x) * ((y - My) / sigma_y)) +
            (((y - My) / sigma_y) ** 2))
    return a * math.exp(-b)


def get_W_y_given_x(x, y):
    a = 1 / (sigma_y * math.sqrt(2 * math.pi * (1 - r ** 2)))
    b = (1 / (2 * sigma_y ** 2 * (1 - r ** 2))) * (y - My - ((sigma_y * (x - Mx)) / sigma_x)) ** 2
    return a * math.exp(-b)


def get_W_x_given_y(x, y, r_arg=r):
    a = 1 / (sigma_x * math.sqrt(2 * math.pi * (1 - r_arg ** 2)))
    b = (1 / (2 * sigma_x ** 2 * (1 - r_arg ** 2))) * (x - Mx - ((sigma_x * (y - My)) / sigma_y)) ** 2
    return a * math.exp(-b)


def get_W_x(x):
    a = (1 / sigma_x * math.sqrt(2 * math.pi))
    b = (x - Mx) ** 2 / (2 * sigma_x ** 2)
    return a * math.exp(-b)


def get_W_y(y):
    a = (1 / sigma_y * math.sqrt(2 * math.pi))
    b = (y - My) ** 2 / (2 * sigma_y ** 2)
    return a * math.exp(-b)


def get_P_K1():
    x_lower, x_upper, y_lower, y_upper = get_K1_limits()
    return integrate.nquad(get_W_x_y, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_P_K2():
    x_lower, x_upper, y_lower, y_upper = get_K2_limits()
    return integrate.nquad(get_W_x_y, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_M_x_if_class_is_K1():
    x_lower, x_upper, y_lower, y_upper = get_K1_limits()
    fun = lambda y_arg, x_arg: x_arg * get_W_y_given_x(x_arg, y_arg)
    # return (1 / get_P_K1()) * integrate.nquad(fun, [[y_lower, y_upper], [x_lower, x_upper]])[0]
    return integrate.nquad(fun, [[y_lower, y_upper], [x_lower, x_upper]])[0]


def get_M_x_if_class_is_K2():
    x_lower, x_upper, y_lower, y_upper = get_K2_limits()
    fun = lambda y_arg, x_arg: x_arg * get_W_y_given_x(x_arg, y_arg)
    # return (1 / get_P_K2()) * integrate.nquad(fun, [[y_lower, y_upper], [x_lower, x_upper]])[0]
    return integrate.nquad(fun, [[y_lower, y_upper], [x_lower, x_upper]])[0]


def get_W_x_if_class_is_K1(x):
    x_lower, x_upper, y_lower, y_upper = get_K1_limits()
    fun = lambda y: get_W_y_given_x(x, y)
    return (1 / get_P_K1()) * integrate.quad(fun, y_lower, y_upper)[0]
    # return integrate.quad(fun, y_lower, y_upper)[0]


def get_W_x_if_class_is_K2(x):
    x_lower, x_upper, y_lower, y_upper = get_K2_limits()
    fun = lambda y: get_W_y_given_x(x, y)
    return (1 / get_P_K2()) * integrate.quad(fun, y_lower, y_upper)[0]
    # return integrate.quad(fun, y_lower, y_upper)[0]


def get_X_threshold():
    x_lower, x_upper = get_X_general_limits()
    fun = lambda x: get_W_x_y(x, Y_threshold)
    return integrate.quad(fun, x_lower, x_upper)[0]


def get_P_dec_K1():
    y_lower, y_upper = get_Y_general_limits()
    x_lower, x_upper = get_X_K1_limits()
    fun = lambda y, x: get_W_x_y(x, y)
    return integrate.nquad(fun, [[y_lower, y_upper], [x_lower, x_upper]])[0]


def get_P_dec_K2():
    y_lower, y_upper = get_Y_general_limits()
    x_lower, x_upper = get_X_K2_limits()
    fun = lambda y, x: get_W_x_y(x, y)
    return integrate.nquad(fun, [[y_lower, y_upper], [x_lower, x_upper]])[0]


def get_M_y_if_is_decision_K1(r_arg=r):
    x_lower, x_upper = get_X_K1_limits()
    y_lower, y_upper = get_Y_general_limits()
    fun = lambda x_arg, y_arg: y_arg * get_W_x_given_y(x_arg, y_arg, r_arg)
    return integrate.nquad(fun, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_M_y_if_is_decision_K2(r_arg=r):
    x_lower, x_upper = get_X_K2_limits()
    y_lower, y_upper = get_Y_general_limits()
    fun = lambda x_arg, y_arg: y_arg * get_W_x_given_y(x_arg, y_arg, r_arg)
    return integrate.nquad(fun, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_W_y_given_decision_K1(y):
    x_lower, x_upper = get_X_K1_limits()
    fun = lambda x: get_W_x_given_y(x, y)
    # return integrate.quad(fun, x_lower, x_upper)[0]
    return (1 / get_P_dec_K1()) * integrate.quad(fun, x_lower, x_upper)[0]


def get_W_y_given_decision_K2(y):
    x_lower, x_upper = get_X_K2_limits()
    fun = lambda x: get_W_x_given_y(x, y)
    # return integrate.quad(fun, x_lower, x_upper)[0]
    return (1 / get_P_dec_K2()) * integrate.quad(fun, x_lower, x_upper)[0]


def get_L_x(x):
    y_lower_K1, y_upper_K1 = get_Y_K1_limits()
    y_lower_K2, y_upper_K2 = get_Y_K2_limits()
    fun = lambda y: get_W_y_given_x(x, y)
    a = (1 / get_P_K1()) * integrate.quad(fun, y_lower_K1, y_upper_K1)[0]
    b = (1 / get_P_K2()) * integrate.quad(fun, y_lower_K2, y_upper_K2)[0]
    return a / b


def get_P_K1_dec_K2():
    x_lower, x_upper = get_X_K2_limits()
    y_lower, y_upper = get_Y_K1_limits()
    fun = lambda y, x: get_W_x_y(x, y)
    return integrate.nquad(fun, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_P_K2_dec_K1():
    x_lower, x_upper = get_X_K1_limits()
    y_lower, y_upper = get_Y_K2_limits()
    fun = lambda y, x: get_W_x_y(x, y)
    return integrate.nquad(fun, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_P_K1_dec_K1():
    x_lower, x_upper = get_X_K1_limits()
    y_lower, y_upper = get_Y_K1_limits()
    fun = lambda y, x: get_W_x_y(x, y)
    return integrate.nquad(fun, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_P_K2_dec_K2():
    x_lower, x_upper = get_X_K2_limits()
    y_lower, y_upper = get_Y_K2_limits()
    fun = lambda y, x: get_W_x_y(x, y)
    return integrate.nquad(fun, [[x_lower, x_upper], [y_lower, y_upper]])[0]


def get_credibility_criteria():
    return get_P_K2() / get_P_K1()


def main():
    # P(K1), P(K2)
    print()
    print_task(3.3)
    print("Апріорні ймовірності приналежності екземпляра виробу до класу К1 або К2:")
    print("P(K1) =", get_P_K1())
    print("P(K2) =", get_P_K2())

    # M[x/K1], M[x/K2]
    print()
    print_task(3.4)
    print("Математичне сподівання ознаки за умови, що примірник належить до класу К1 або К2:")
    print("M[x/K1] =", get_M_x_if_class_is_K1())
    print("M[x/K2] =", get_M_x_if_class_is_K2())

    # W(x)
    label = "Безумовна густина розподілу ознаки х"
    x_arr = [*map(lambda coef: Mx + coef * sigma_x, sigma_coefs)]
    w_x_arr = [*map(lambda x: get_W_x(x), x_arr)]

    x_labels = get_sigma_coef_labels("Mx", "sigma_x")
    col_names = ["Вираз х", "Значення x", "W(x)"]
    table_data = np.vstack((x_labels, x_arr, w_x_arr)).T

    print()
    print_task(3.5)
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    if do_show_graphics:
        plt.plot(x_arr, w_x_arr)
        plt.xlabel("x")
        plt.ylabel("W(x)")
        plt.title(label)
        plt.show()

    # W(x/K1), W(x/K2)
    label = "Значення умовних густин розподілу ознаки за умови, що примірник належить до класу К1 або К2"

    x_K1_arr = [*map(lambda coef: get_M_x_if_class_is_K1() + coef * sigma_x, sigma_coefs)]
    w_x_K1_arr = [*map(lambda x: get_W_x_if_class_is_K1(x), x_K1_arr)]

    x_K2_arr = [*map(lambda coef: get_M_x_if_class_is_K2() + coef * sigma_x, sigma_coefs)]
    w_x_K2_arr = [*map(lambda x: get_W_x_if_class_is_K2(x), x_K2_arr)]

    x_labels_K1 = get_sigma_coef_labels("M[x/K1]", "sigma_x")
    x_labels_K2 = get_sigma_coef_labels("M[x/K2]", "sigma_x")

    x_K_arr = [*x_K1_arr, *x_K2_arr]
    w_x_K_arr = [*w_x_K1_arr, *w_x_K2_arr]
    x_labels = [*x_labels_K1, *x_labels_K2]

    col_names = ["Вираз х", "Значення x", "W(x/K)"]
    table_data = np.vstack((x_labels, x_K_arr, w_x_K_arr)).T

    print()
    print_task(3.6)
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    if do_show_graphics:
        plt.plot(x_K1_arr, w_x_K1_arr, label="W(x/K1)")
        plt.plot(x_K2_arr, w_x_K2_arr, label="W(x/K2)")
        plt.xlabel("x")
        plt.ylabel("W(x/K)")
        plt.legend()
        plt.title(label)
        plt.show()

    # X_threshold
    X_threshold = get_X_threshold()
    print()
    print_task(3.7)
    print("Порогове значення X:", X_threshold)

    # P(dec K1), P(dec K2)
    print()
    print_task(3.8)
    print("Апріорні ймовірності ухвалення рішення про віднесення екземплярів до класів К1 і К2:")
    print("P(ріш К1) =", get_P_dec_K1())
    print("P(ріш К2) =", get_P_dec_K2())

    # M[y/ріш K1], M[y/ріш K2]
    print()
    print_task(3.9)
    print("Умовне математичне сподівання прог. параметра при умовах віднесення примірника до класу К1 або К2:")
    print("M[y/ріш K1] =", get_M_y_if_is_decision_K1())
    print("M[y/ріш K2] =", get_M_y_if_is_decision_K2())

    # W(y)
    label = "Значення безумовної густини прогнозованого параметра W(y)"
    y_arr = [*map(lambda coef: My + coef * 0.2, sigma_coefs)]
    w_y_arr = [*map(lambda y: get_W_y(y), y_arr)]

    y_labels = get_sigma_coef_labels("My", "sigma_y")

    col_names = ["Вираз y", "Значення y", "W(y)"]
    table_data = np.vstack((y_labels, y_arr, w_y_arr)).T

    print()
    print_task("3.10")
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    if do_show_graphics:
        plt.plot(y_arr, w_y_arr)
        plt.xlabel("y")
        plt.ylabel("W(y)")
        plt.title(label)
        plt.show()

    # W(y|ріш К1), W(y|ріш К2)
    label = "Умовні розподіли прогнозованого параметра за умови, якщо приймається рішення про віднесення до класу К1 " \
            "або К2 "

    y_K1_arr = [*map(lambda coef: get_M_y_if_is_decision_K1() + coef * sigma_y, sigma_coefs)]
    w_y_decision_K1_arr = [*map(lambda y: get_W_y_given_decision_K1(y), y_K1_arr)]

    y_K2_arr = [*map(lambda coef: get_M_y_if_is_decision_K2() + coef * sigma_y, sigma_coefs)]
    w_y_decision_K2_arr = [*map(lambda y: get_W_y_given_decision_K2(y), y_K2_arr)]

    y_labels_K1 = get_sigma_coef_labels("M[y|ріш K1]", "sigma_y")
    y_labels_K2 = get_sigma_coef_labels("M[y|ріш K2]", "sigma_y")

    y_K_arr = [*y_K1_arr, *y_K2_arr]
    w_y_decision_K_arr = [*w_y_decision_K1_arr, *w_y_decision_K2_arr]
    y_labels = [*y_labels_K1, *y_labels_K2]

    col_names = ["Вираз y", "Значення y", "W(y|ріш K)"]
    table_data = np.vstack((y_labels, y_K_arr, w_y_decision_K_arr)).T

    print()
    print_task(3.11)
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    if do_show_graphics:
        plt.plot(y_K1_arr, w_y_decision_K1_arr, label="W(y|ріш K1)")
        plt.plot(y_K2_arr, w_y_decision_K2_arr, label="W(y|ріш K2)")
        plt.xlabel("y")
        plt.ylabel("W(y| ріш K)")
        plt.legend()
        plt.title(label)
        plt.show()

    # Відношення правдоподібності для 4-х значень ознаки x
    print()
    print_task(3.12)
    print("Відношення правдоподібності L(x) для чотирьох значень ознаки X:")
    for i in range(len(X_arr)):
        print("L(x{}) = {}".format(i + 1, get_L_x(X_arr[i])))

    # П (порогове значення відношення правдоподібності)
    print()
    print_task(3.13)
    print("Порогове значення відношення правдоподібності:")
    CC = get_credibility_criteria()
    print("П =", CC)
    for i in range(len(X_arr)):
        l_x = get_L_x(X_arr[i])
        print("X({}) (L = {}) належить класу ".format(i + 1, l_x), end='')
        if CC <= l_x:
            print("K1")
        else:
            print("K2")

    # Класифікація ознак за пороговим значенням
    print()
    print_task(3.14)
    print("Класифікація ознак за пороговим значенням:")
    print("X пор =", X_threshold)
    for i in range(len(X_arr)):
        print("Приймаємо рішення, що X({}) = {} належить до класу ".format(i + 1, X_arr[i]), end='')
        if X_threshold > X_arr[i]:
            print("K1")
        else:
            print("K2")

    #


if __name__ == '__main__':
    main()

# https://stud.com.ua/151074/ekonomika/dvovimirniy_mirniy_normalniy_zakon_rozpodilu
