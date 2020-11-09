import numpy as np
import get_dis
import draw_background
a = [[1,2,3], [4,5,6]]
b = np.array(a, copy=True)
c = [1,2]
c = [c[0]*1/2, c[1]*1/2]
x = []
x.append(a)
x.append(a)
y = x[:]
y.append(c)
path = [[1, 2], [3, 4], [5, 6], [8, 8]]
enemy = [[12, 1, 2], [1, 12, 2], [4, 17, 1]]
target = [13, 13]
draw_background.draw_print(path, enemy, target)

#print(y)
#z = y.pop()
#print(y)
#print(z)
#print(a)
#k = b[0]
#print(k)
#print(b)
#c = np.array(b, copy = True)
#print(c)
#print(y)
#print(x)
#print(get_dis.get_dot_line_dis([0,0], [3,0], [4,2]))