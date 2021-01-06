import UAV
import planning
import get_dis
import enemy_record
import draw_background

UAVs = [[1,2], [2,3]]
test_inital_pos = [1,2]
test_target = [18,18]
map = [[10000]*400 for _ in range(400)]
enemy_list = enemy_record.get_enemy_pos()

for i in range(20):
    for j in range(20):
        for k in range(20):
            for z in range(20):
                map[i*20+j][k*20+z] = get_dis.eucliDist_m([i,j], [k,z])
                map[k*20+z][i*20+j] = get_dis.eucliDist_m([i,j], [k,z])

# print(map)

plan_list = planning.plan(test_inital_pos, map, test_target)
print(plan_list)
draw_background.draw_print(plan_list, enemy_list, test_target)
#print(map)