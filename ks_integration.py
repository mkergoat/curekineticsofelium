###confirmation program
##compares alpha_exp to alpha_computed
#MK 21/08/2019
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
from scipy.optimize import fsolve
import matplotlib2tikz #This package creates a .tex file containing the
##tikz figure corresponding to the plot
N=2800
dt=1
A1=4.961501639474312e+21
A2=182.9868133108547
Ea1=159138.6725691667
Ea2=24525.422423966505
R=8.314
m=1
n=2
def dalpha(alpha,t):
    k1=A1*np.exp(-Ea1/(R*T0))
    k2=A2*np.exp(-Ea2/(R*T0))
    af= -1.2534950456164184  + 0.005807805044924184*T0
    b=-0.19333429200833865  +  0.0006832726804877584*T0
    fd=fd=2/(1+np.exp((alpha-af)/b))-1
    return((k1+k2*alpha**m)*(1-alpha)**n*fd)

#Euler discretisation: f'(x)\approx (f(x+\Delta x)-f(x))/(\Delta x) for \Delta x small

T = np.array([28.7,48.6,59.0,61.4])
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

#matplotlib2tikz.save("cure-times.tex") #comment to get TikZ code for plot
