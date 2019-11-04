#Estimation of Kamal and Sourour Models Parameters
#dalpha/dt=(K1+K2 alpha^m)(1-alpha)^n
#K1=A1exp(-Ea1/RT), K2=A2exp(-Ea2/RT)
#Estimates value of activation energies and
#pre-exponential factors given K1 and K2 values and isothermal temperatures

### Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate,stats

f=open('resultats.txt')
r=f.readlines()
print(r)
runs=[]
K1=[]
K2=[]
af=[]
b=[]
T=[]
r=r[1:]
for line in r:
    runs.append(line.split(' ')[0])
    K1.append(float(line.split(' ')[1]))
    K2.append(float(line.split(' ')[2]))
    af.append(float(line.split(' ')[3]))
    b.append(float(line.split(' ')[4]))
    T.append(float(line.split(' ')[5]))
f.close()

### Constants
R=8.314 #J/(K/mol)

### Entries
K1=np.array(K1)
K2=np.array(K2)
af=np.array(af)
b=np.array(b)
T=np.array(T)

### Analysis
lnK1=np.log(K1)
lnK2=np.log(K2)
invT=1/(T+273.15)
b1=stats.linregress(invT,lnK1)
b2=stats.linregress(invT,lnK2)
b3=stats.linregress(T+273.15,af)
b4=stats.linregress(T+273.15,b)
print("Estimated coefficients:\n A1 (intercept) = {}  \n Ea1 (slope*R) = {}".format(np.exp(b1[1]), b1[0]*R))
print('Standard error on Ea1 = {}'.format(b1[4]))
print('Correlation coefficient (r squared) on K1 plot = {}'.format(b1[2]**2))
print("Estimated coefficients:\n A2 (intercept) = {}  \n Ea2 (slope*R) = {}".format(np.exp(b2[1]), b2[0]*R))
print('Standard error on Ea2 = {}'.format(b2[4]))
print('Correlation coefficient (r squared) on K2 plot = {}'.format(b2[2]**2))
print("Estimated coefficients for af temp-dependancy :\n (intercept) = {}  \n (slope) = {}".format(b3[1], b3[0]))
print('Correlation coefficient (r squared) on af plot = {}'.format(b3[2]**2))
print("Estimated coefficients for b temp-dependancy :\n (intercept) = {}  \n (slope) = {}".format(b4[1], b4[0]))
print('Correlation coefficient (r squared) on b plot = {}'.format(b4[2]**2))

### Outputs
plt.plot(invT, lnK1,marker='+',linestyle='')
plt.plot(invT, b1[1]+b1[0]*invT)
plt.xlabel('$1/T$ (K)')
plt.ylabel('$\ln(K_1)$')
plt.figure()
plt.plot(invT, lnK2,marker='+',linestyle='')
plt.plot(invT, b2[1]+b2[0]*invT)
plt.xlabel('$1/T$ (K)')
plt.ylabel('$\ln(K_2)$')
plt.figure()
plt.plot(T+273.15, af,marker='+',linestyle='')
plt.plot(T+273.15, b3[1]+b3[0]*(T+273.15))
plt.xlabel('$T$ (K)')
plt.ylabel('$a_f$')
plt.figure()
plt.plot(T+273.15, b,marker='+',linestyle='')
plt.plot(T+273.15, b4[1]+b4[0]*(T+273.15))
plt.xlabel('$T$ (K)')
plt.ylabel('$b$')
plt.show()


