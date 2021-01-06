import matplotlib.pyplot as plt
import numpy as np
import get_dis

def draw_print(input_path, input_enemy, target):
    x = []
    y = []
    for point in input_path:
        x.append(point[0])
        y.append(point[1])
    # x_e = np.arange(0, 20, 1)
    # y_e = np.arange(0, 20, 1)
    n = 200
    x_e = np.linspace(0, 20, n)
    y_e = np.linspace(0, 20, n)
    X_e, Y_e = np.meshgrid(x_e, y_e)
    Z_e = X_e**2 + Y_e**2
    plt.figure(figsize=(8, 8))
    plt.contour(X_e, Y_e, Ematrix(input_enemy))
    plt.plot(x, y, color='r', marker='o', linestyle='dashed')
    plt.scatter(target[0], target[1], marker='x')
    plt.axis([0, 19, 0, 19])
    plt.show()

def Ematrix(enemy_list):
    E = [[0]*200 for _ in range(200)]
    e_num = len(enemy_list)
    for i in range(e_num):
        td = [[0]*200 for _ in range(200)]
        for x in range(200):
            for y in range(200):
                td[x][y] = get_dis.eucliDist_m([x,y],
                                               [enemy_list[i][0]*10, enemy_list[i][1]*10])
        # te = gaussmf(td, enemy_list[i][2]*10, 0)
        te = gaussmf(td, enemy_list[i][2], 0)
        for x in range(200):
            for y in range(200):
                E[x][y] = 1-(1-E[x][y])*(1-te[x][y]/200)
    return E


def gaussmf(x, sigma=1, mean=0, scale=1):
    te = [[0]*len(x[0]) for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[0])):
            te[i][j] = scale * np.exp(-np.square(x[i][j] - mean) / (2 * sigma ** 2))
    return te
