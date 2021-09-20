import numpy as np
from common.basedir import BASEDIR
from common.numpy_fast import interp_2d

MPH_TO_MS = 0.44704

model_weights_file = f'{BASEDIR}/models/steering/corolla_model_v5_weights.npz'
w, b = np.load(model_weights_file, allow_pickle=True)['wb']



def predict(x):
  x = np.array(x, dtype=np.float32)
  l0 = np.dot(x, w[0]) + b[0]
  l0 = np.where(l0 > 0, l0, l0 * 0.3)
  l1 = np.dot(l0, w[1]) + b[1]
  l1 = np.where(l1 > 0, l1, l1 * 0.3)
  l2 = np.dot(l1, w[2]) + b[2]
  return l2


lst = []
# for v_ego in [2, 13, 35]:
angle_steers_des = 30
# angle_steers = 10
v_ego = 8 * MPH_TO_MS


# model doesn't like right curves so just use left
ff_left = predict([angle_steers_des, angle_steers_des, 0, 0, v_ego])[0]
_c1, _c2, _c3 = 0.35189607550172824, 7.506201251644202, 69.226826411091
# print(ff_left / angle_steers_des / (_c1 * v_ego ** 2 + _c2 * v_ego + _c3))
# ff_right = -predict([-angle_steers_des, -angle_steers_des, 0, 0, v_ego])[0]


angle_error = 5

k_p_rising = [abs(ff_left - predict([angle_steers_des, angle_steers_des - angle_error, 0, 0, v_ego])[0]) / angle_error,
              # abs(-ff_right - predict([-angle_steers_des, -angle_steers_des + angle_error, 0, 0, v_ego])[0]) / angle_error
              ]
k_p_falling = [abs(ff_left - predict([angle_steers_des, angle_steers_des + angle_error, 0, 0, v_ego])[0]) / angle_error,
               # abs(-ff_right - predict([-angle_steers_des, -angle_steers_des - angle_error, 0, 0, v_ego])[0]) / angle_error
               ]


print('k_p_rising:')
print(f'rising mean: {round(np.mean(k_p_rising), 5)}')
lst.append(round(np.mean(k_p_rising), 5))
print(k_p_rising)
print()
print('k_p_falling:')
print(f'falling mean: {np.mean(k_p_falling)}')
print(k_p_falling)

print('relationship from falling to rising: {}'.format(np.mean(k_p_falling) / np.mean(k_p_rising)))
print('======')
# print(lst)
