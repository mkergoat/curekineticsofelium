##n-order kinetics model solution via Euler and odeint method
#
#MK 20/06/2019
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
from scipy.optimize import fsolve
import matplotlib2tikz #This package creates a .tex file containing the
##tikz figure corresponding to the plot

N =10000 #number of time steps
dt = 1 #sec
T0 = 110+273.15
def s2hrs(t):
    return(t/3600)

#### constants
A = np.exp(5.06) #arrhenius pre-factor
n = 1.06 #order of reaction
Ea = 43.60*10**(3) #J/mol
R = 8.314 #Gas constant (J/mol K)
           
def dalpha(alpha,t):
    k=A*np.exp(-Ea/(R*T0))
    return(k*(1-alpha)**n)

#Euler discretisation: f'(x)\approx (f(x+\Delta x)-f(x))/(\Delta x) for \Delta x small

T = np.array([80,100,120,140,160,180,200])
markers = ['+','.','x','d','^','<','>','o']
k = 0
for Ti in T:
    alpha0 = 0
    alpha = [alpha0]
    t = [0]
    T0 = Ti+273.15
    for i in range(1,N):
        alpha.append(alpha[i-1]+dt*dalpha(alpha[i-1],t[i-1]+dt))
        t.append(t[i-1]+dt)
    t=np.array(t)
    #solution=sp.odeint(dalpha,alpha0,t)
    plt.plot(t,alpha,label=str(Ti)+' °C (Euler)',marker=markers[k],markevery=250)
    k+=1
    #plt.plot(t,solution,label=str(Ti)+' °C (odeint)', linestyle=':',marker='x',markevery=500)
plt.plot(t,[1.0]*N,color='b',linestyle=':')
plt.plot(t,[0.95]*N,color='b',linestyle=':')
plt.xlabel('Time (s)')
plt.ylabel('alpha (-)')
plt.legend()
plt.grid(True)
print('end')
plt.show() #uncomment to see picture
print('t max='+str(s2hrs(N)))

matplotlib2tikz.save("cure-times.tex") #comment to get TikZ code for plot
