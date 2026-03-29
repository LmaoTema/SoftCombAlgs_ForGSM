import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from transmitter.modulator import GMSKModulation 
from receiver.detector.gmsk_det import GMSKDetector

params = {}
block_params = {"matched filter": {"is_working": False}}

mod = GMSKModulation(params)
det = GMSKDetector(params, block_params)

bits = np.random.randint(0, 2, 148)
sig = mod.process_mod(bits)
g_t = det.gmsk_filter()
q_t = det.generate_q_gmsk()


BT = 0.3
T = 3.69e-6
sps =  4
h =  0.5
gaus_duration =  4
rect_duration =  1
L = gaus_duration + rect_duration

oversampling = 100
sps_oversampling = sps * oversampling
dt = T / sps_oversampling

t_g = np.arange(g_t.size) * dt

s_1 = np.sin(np.pi / 2 * q_t)
t_1 = np.arange(s_1.size) * dt / T

s_2 = np.sin(np.pi / 2 - np.pi / 2 * (- q_t))
t_2 = (s_1.size + np.arange(s_2.size)) * dt / T

s = np.concatenate([s_1, s_2, np.zeros(2)])
t_s = np.arange(s.size) * dt

# s = []
# t_s = []
# for i in range (L):
#     s.append(np.concatenate([s_1, s_2]))
#     t_s.append(i + (  np.arange(s[0].size)) * dt / T)

c_0 = np.ones((L + 1) * sps_oversampling)
for i in range((L + 1) * sps_oversampling):
    for j in range(L):
        c_0[i] *= s[i + j * sps_oversampling]

c_0_sampled = c_0[::oversampling]

a = 0