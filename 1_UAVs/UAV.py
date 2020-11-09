import numpy as np
import get_dis
class UAV(object):
    initial_pos = None
    enemy_find_list = []
    pos_enemy_loss = [0]*400                                  # 在敌人范围内的点会有额外的距离加成
    now_pos = None
    map = None                                                 # 假设一张图上有400个标准坐标点,map为点与点之间的距离
    visual_field = None
    step_length = None

    def __init__(self, initial_pos_input, map_input, step_input, visual_input):
        self.initial_pos = initial_pos_input
        self.now_pos = self.initial_pos
        self.map = map_input
        self.step_length = step_input
        self.visual_field = visual_input

    def get_dis_value(self):                                    # 这个函数服务于dijskra函数，获得当前pos与其他点距离,结合map使用
        dis_value = [0]*400
        for i in range(20):
            for j in range(20):
                dis_value[i*20 + j] = get_dis.eucliDist_m(self.now_pos, [i,j])
                for enemy in self.enemy_find_list:
                    if [i, j] != self.now_pos and get_dis.get_dot_line_dis(self.now_pos, [i, j],
                                                                           [enemy[0], enemy[1]]) < enemy[2]:
                        dis_value[i*20 + j] += 10000
                    elif [i, j] == self.now_pos and get_dis.eucliDist_m([i, j], [enemy[0], enemy[1]]) < enemy[2]:
                        dis_value[i*20 + j] += 10000
        return dis_value

    def dijskra_next_pos(self):
        """
        :return: now_pos得到与其他所有点之间的最短路
        """
        visited = [0]*400
        path_record = {i:[] for i in range(400)}
        min_dis_value = self.get_dis_value()                    # 初始化当前点与其他所有点的最短路
        for i in range(400):                                    # 对map上所有点dijskra
            min_num = 99999
            for j in range(400):
                if not visited[j] and min_dis_value[j] < min_num:
                    new_min_index = j
                    min_num = min_dis_value[j]
            visited[new_min_index] = 1
            path_record[new_min_index].append([new_min_index//20, new_min_index % 20])
            for j in range(400):
                if not visited[j] and min_num + self.map[j][new_min_index] < min_dis_value[j]:
                    min_dis_value[j] = min_num + self.map[j][new_min_index]
                    path_record[j] = path_record[new_min_index][:]

        return path_record
                                                                # 遇到敌人额外的距离加成在map的更新中处理，即在plan函数中