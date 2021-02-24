import numpy as np
import get_dis
class UAV(object):
    initial_pos = None
    enemy_find_list = []
    pos_enemy_loss = [0]*400                                  # Points within enemy range have an additional risk
    now_pos = None
    map = None                                                # Assume that there are 400 standard coordinate points on a graph. Map is the distance between points
    visual_field = None
    step_length = None

    def __init__(self, initial_pos_input, map_input, step_input, visual_input):
        self.initial_pos = initial_pos_input
        self.now_pos = self.initial_pos
        self.map = map_input
        self.step_length = step_input
        self.visual_field = visual_input

    def get_dis_value(self):                                    # This function serves the dijskra function to get the distance between the current pos and other points, which is used in combination with the map
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
        min_dis_value = self.get_dis_value()                    # Initializes the shortest circuit between the current point and all other points
        for i in range(400):                                    # dijskra
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
        # The additional risk of encountering an enemy is handled in the map update, that is in the plan function