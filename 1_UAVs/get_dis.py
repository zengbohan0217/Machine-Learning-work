import numpy as np
import math

def eucliDist_np(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))

def eucliDist_m(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

def get_dot_line_dis(A,B,C):                                             # 获得点C到A B连线的距离
    s = abs((A[0]-C[0])*(B[1]-C[1]) - (B[0]-C[0])*(A[1]-C[1]))/eucliDist_m(A,B)
    return s