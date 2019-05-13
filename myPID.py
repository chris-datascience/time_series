# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 22:45:57 2018

@author: Kris
"""

import numpy as np
import matplotlib.pyplot as plt

L = 100
setpoint = np.zeros(L,)
setpoint[10:] = 4
previous_error = 0
integral = 0
measured_value = [0.]
Kp = .3
Ki = .1
Kd = .0001
dt = 1
output = []
for i in range(L):
  error = setpoint[i] - measured_value[-1]
  integral +=  error * dt
  derivative = (error - previous_error) / dt
  output.append(Kp * error + Ki * integral + Kd * derivative)
  previous_error = error
  
  measured_value.append(measured_value[-1] + output[-1])

plt.figure(figsize=(8,6))
plt.plot(range(L), setpoint, 'r--')
plt.plot(range(L), measured_value[:-1], 'b-')
#plt.xlim(7,15)
plt.title('Process control', fontsize=15)

plt.figure(figsize=(8,4))
plt.plot(range(L), output, 'm-')
plt.title('PID Output', fontsize=15)