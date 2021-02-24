import UAV
import numpy as np
import get_dis
import enemy_record

def plan(inital_pos, input_map, target):                         # inital_pos and target are both lists of size 2
    # This function gets the final path
    uav = UAV.UAV(inital_pos, input_map, 1, 2)
    path_record = []
    path_record.append(inital_pos)
    enemyUK = enemy_record.get_enemy_pos()
    new_enemy_list = []
    replanning = 0
    while get_dis.eucliDist_m(uav.now_pos, target) > 0.001:
        if replanning == 1:
            update_map(uav, new_enemy_list)
            replanning = 0
        dis_list = uav.dijskra_next_pos()
        next_pos = dis_list[target[0]*20 + target[1]][0]
        move_length = get_dis.eucliDist_m(next_pos, uav.now_pos)
        if move_length > uav.step_length:
            mid = uav.step_length/move_length
            next_pos = [uav.now_pos[0]+abs(next_pos[0] - uav.now_pos[0])*mid,
                        uav.now_pos[1]+abs(next_pos[1] - uav.now_pos[1])*mid]

        print('get new pos:')
        print(next_pos)
        uav.now_pos = next_pos
        path_record.append(next_pos)
        for enemy in enemyUK:
            if get_dis.eucliDist_m([enemy[0], enemy[1]], uav.now_pos) < uav.visual_field:
                replanning = 1
                new_enemy_list.append(enemy)
                uav.enemy_find_list.append(enemy)
                enemyUK.remove(enemy)
        print('get new enemy')
        print(new_enemy_list)
    print(uav.map)
    return path_record

def update_map(uav, new_enemy_list):
    new_map = np.array(uav.map, copy=True)
    #while new_enemy_list is not None:
    #    try:
    #        new_enemy = new_enemy_list.pop()
    #    except:
    #        break
    for new_enemy in new_enemy_list:
        for i in range(20):
            for j in range(20):
                for k in range(20):
                    for z in range(20):
                        if i != k or j != z:
                            if get_dis.get_dot_line_dis([i,j], [k,z], [new_enemy[0],new_enemy[1]]) < new_enemy[2]:
                                new_map[i*20+j][k*20+z] += 20/(get_dis.get_dot_line_dis([i,j], [k,z], [new_enemy[0],new_enemy[1]])+1)
                                new_map[k*20+z][i*20+j] += 20/(get_dis.get_dot_line_dis([i,j], [k,z], [new_enemy[0],new_enemy[1]])+1)
        new_enemy_list.remove(new_enemy)
    uav.map = new_map
