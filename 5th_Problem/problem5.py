# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:04:01 2020

@author: Ryohei Oura
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from numpy.random import randn
import csv

###Problem 5 to 9

###Problem 5
##HSIC1
def HSIC_1(x, y, k_x, k_y):
    n = len(x)
    #factor of first term
    S = 0
    for i in range(n):
        for j in range(n):
            S = S + k_x(x[i], x[j]) * k_y(y[i], y[j])
            
    #sefactor of cond term
    T = 0
    for i in range(n):
        T_1 = 0
        for j in range(n):
            T_1 = T_1 + k_x(x[i], x[j])
        T_2 = 0
        for j in range(n):
            T_2 = T_2 + k_y(y[i], y[j])
        T = T + T_1 * T_2
        
    #factors of third term
    U = 0
    for i in range(n):
        for j in range(n):
            U = U + k_x(x[i], x[j])
    V = 0
    for i in range(n):
        for j in range(n):
            V = V + k_y(y[i], y[j])
            
    #blank
    return S / n**2 - 2 * T / n**3 + U * V / n**4

##Gaussian kearnel
#for X
def k_x(x, y):
    S = np.exp(-(x - y)**2 / 2)
    return S

#for Y
def k_y(x, y):
    return k_x(x, y)

##Execute
n = 100
a_seq = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8])
for a in a_seq:
    x = np.random.randn(n)
    z = np.random.randn(n)
    y = a * x + np.sqrt(1 - a**2) * z
    print(HSIC_1(x, y, k_x, k_y))
 
###Problem 6
def lingam_1(x, y):
    a_hat = np.sum(x * y) / np.sum(x * x)
    e = y - a_hat * x
    h_1 = HSIC_1(x, e, k_x, k_y)
    b_hat = np.sum(x * y) / np.sum(y * y)
    f = x - b_hat * y
    #blank
    h_2 = HSIC_1(y, f, k_y, k_x)
    if h_1 < h_2:
        return 1
    else:
        return 2

##Execute
n = 100
a = 1
x = np.random.randn(n)**2 - np.random.randn(n)**2
y = a * x + np.random.randn(n)**2 - np.random.randn(n)**2
print(lingam_1(x, y))
txt_file = 'crime.txt'
crime = []  # crime rate
fund = []  # funding
with open(txt_file, 'r') as f:
    reader = csv.reader(f, delimiter = '\t')
    for row in reader:
        crime.append(int(row[0]))
        fund.append(int(row[2]))
# convert to numpy array
X = np.array(crime)
Y = np.array(fund)
print(lingam_1(X, Y))

###Problem 7
##HSIC2
def HSIC_2(x, y, z, k_x, k_y, k_z):
    n = len(x)
    #facor of first term
    S = 0
    for i in range(n):
        for j in range(n):
            S = S + k_x(x[i], x[j]) * k_y(y[i], y[j]) * k_z(z[i], z[j])
            
    #foctor of second term
    T = 0
    for i in range(n):
        T_1 = 0
        for j in range(n):
            T_1 = T_1 + k_x(x[i], x[j])
        T_2 = 0
        for j in range(n):
            T_2 = T_2 + k_y(y[i], y[j]) * k_z(z[i], z[j])
        T = T + T_1 * T_2
        
    #foctors of third term
    U = 0
    for i in range(n):
        for j in range(n):
            U = U + k_x(x[i], x[j])
    V = 0
    for i in range(n):
        for j in range(n):
            V = V + k_y(y[i], y[j]) * k_z(z[i], z[j])
    return S / n**2 - 2 * T / n**3 + U * V / n**4

##Gaussian kernel for Z
def k_z(x, y):
    return k_x(x, y)

##Execute
n = 200
a_seq = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8])
for a in a_seq:
    x = np.random.randn(n)
    u = np.random.randn(n)
    y = a * x + np.sqrt(1 - a**2) * u
    v = np.random.randn(n)
    z = a * y + np.sqrt(1 - a**2) * v
    print(HSIC_2(x, y, z, k_x, k_y, k_z))
    
###Problem 9
##Data
x = np.random.randn(n)**2 - np.random.randn(n)**2
y = 2 * x + np.random.randn(n)**2 - np.random.randn(n)**2
z = x + y + np.random.randn(n)**2 - np.random.randn(n)**2
x = x - np.mean(x)
y = y - np.mean(y)
z = z - np.mean(z)

##top
def cc(x, y):
    return sum(x * y) / len(x)

def f(u, v):
    return u - cc(u, v) / cc(v, v) * v

x_y = f(x, y)
y_z = f(y, z)
z_x = f(z, x)
x_z = f(x, z)
z_y = f(z, y)
y_x = f(y, x)
v1 = HSIC_2(x, y_x, z_x, k_x, k_y, k_z)
v2 = HSIC_2(y, z_y, x_y, k_y, k_z, k_x)
v3 = HSIC_2(z, x_z, y_z, k_z, k_x, k_y)
if v1 < v2:
    if v1 < v3:
        top = 1
    else:
        top = 3
else:
    ##blank(1)
    if v2 < v3:
        top = 2
    else:
        top = 3
        
##middle
x_yz = f(x_y, z_y)
y_zx = f(y_z, x_z)
z_xy = f(z_x, y_x)
if top == 1:
    v1 = HSIC_1(y_x, z_xy, k_y, k_z)
    v2 = HSIC_1(z_x, y_zx, k_z, k_y)
    if v1 < v2:
        middle = 2
        bottom = 3
    else:
        middle = 3
        bottom = 2
if top == 2:
    v1 = HSIC_1(z_y, x_yz, k_z, k_x)
    v2 = HSIC_1(x_y, z_xy, k_x, k_z)
    if v1 < v2:
        middle = 3
        bottom = 1
    else:
        middle = 1
        bottom = 3
if top == 3:
    ##blank(2) to (4)
    v1 = HSIC_1(x_z, y_zx, k_x, k_y)
    v2 = HSIC_1(y_z, x_yz, k_y, k_x)
    if v1 < v2:
        middle = 1
        bottom = 2
    else:
        middle = 2
        bottom = 1
print(top)
print(middle)
print(bottom)