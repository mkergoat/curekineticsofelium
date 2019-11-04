##Packages Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate,integrate,signal,stats
import os

###Functions
def traitement(data):
    '''
    Splits the .txt file created from the Excel file
    and extracts data
    '''
    data=open(data,'r')
    r=data.readlines()
    setup_data=[]
    el=r[0]
    sample_mass=r[5].split('\t')[1]
    count=0
    count2=0
    count3=0
    count4=0
    count5=0
    t=[]
    T=[]
    DSC=[]
    DDSC=[]
    info=[]
    prog=[]
    info.append(el.split('\t')[0]+": "+el.split('\t')[1])
    while el.split('\t')[0]!='min':
        count+=1
        el=r[count]
    count+=2
    while r[count2].split('\t')[0]!='Temperature Program':
        count2+=1
    while r[count3].split('\t')[1]!='Temperature Program Mode':
        count3+=1
##    while r[count4].split('\t')[0]!='Surface Area':
##        count4+=1
##        if count4>len(r):
##            count4=0
    while r[count5].split('\t')[0]!='Time':
        count5+=1
    info.append(r[count2].split('\t').pop(0)+' below'+', '+r[count3].split('\t').pop(1)+': '+r[count3].split('\t').pop(2))
    a=r[count2].split('\t')
    a.pop(0)
    a.pop(0)
    a.pop()
    b=['Sweep']
    b.extend(a)
    prog.append(b)
    for i in range(count2+1,count3):
        el=r[i].split('\t')
        el.pop(0)
        el.pop()
        prog.append(el)
    for i in range(count,len(r)):
        line=r[i].split('\t')
        t.append(float(line[0].replace(',','.')))
        T.append(float(line[1].replace(',','.')))
        DSC.append(float(line[2].replace(',','.')))
        DDSC.append(float(line[3].replace(',','.')))
##    if count4==0:
##        DeltaH=0
##    else:
##        DeltaH=r[count4+4].split('\t')[1]
##    data.close()
    return(info,prog,t,T,DSC,DDSC,sample_mass)

def estimate_coef(x, y):
    """
    Returns the coefficients of the linear regression y=b_0+b_1*x, estimated via a Least-Square Method
    """
    n = np.size(x)   
    m_x, m_y = np.mean(x), np.mean(y)   
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x  
    return(b_0, b_1) 

###Main Program
print('Cure Kinetics Evaluation Program')
print('MK, Summer 2019')
str1=input('Do you have a set-up-file ? if yes, enter the filename, if no, enter n:')
if str1=='n':
    filename=input('Enter filename?\n') #Asks for the filename
    path=input('Enter path to data?\n')           #Ask where to search the file
else:
    data=open(str1,'r')
    r=data.readlines()
    filename=str(r[1].rstrip())
    path=str(r[2].rstrip())
    preferency1=str(r[3].rstrip())
    preferency2=str(r[4].rstrip())
    starttemp=str(r[5].rstrip())
    stoptemp=str(r[6].rstrip())
    data.close()
os.chdir(path)                  #Changes path to get data
print('Path successfully changed to '+path)
info,prog,t,T,DSC,DDSC,sample_mass=traitement(filename) #Main call
print('File '+ filename+' successfully opened')
t=np.array(t)                   #Transforms lists into Numpy Arrays for easier use
T=np.array(T)
DSC=np.array(DSC)
DDSC=np.array(DDSC)

##Plot DSC + DDSC with 2 axis
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('temperature (°C)')
ax1.set_ylabel('DSC', color=color)
ax1.plot(T, DSC, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('d(DSC)/dt', color=color)  # we already handled the x-label with ax1
ax2.plot(T, DDSC, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.figure()

#DeltaH=float(DeltaH.replace(',','.'))
sample_mass=float(sample_mass.replace(',','.'))
if str1=='n':
    tx1=input('Want to see the sample information ? y = yes, n= no\n')
    if tx1=='y':
        print(info)                     #Useful infos to check
    tx2=input('Want to see the temperature program ? y = yes, n= no\n')
    if tx2=='y':
        for el in prog:
            print(el)
else:
    if preferency1=='y':
        print(info)                     #Useful infos to check
    if preferency2=='y':
        for el in prog:
            print(el)

peakind = signal.find_peaks_cwt(T, np.arange(1,2500))
#to identify the peaks in T command: useful only if various sweeps have been made

###Re-limitation around peak by user ###
if str1=='n':
    start=float(input('Starting point for DeltaH evaluation?\n'))
    stop=float(input('Stopping point for DeltaH evaluation?\n'))
else:
    start=float(starttemp)
    stop=float(stoptemp)
startIdx=(np.abs(T-start)).argmin()
stopIdx=(np.abs(T-stop)).argmin()
#### calculations on the first sweep ####
##b=peakind[0]-375 #Indice of End of First Heating Ramp
##if len(peakind)>1:
##    t1=np.array(t)[:b+1]*60
##    T1=np.array(T)[:b+1]
##    DSC1=np.array(DSC)[:b+1]
##    DDSC1=np.array(DSC)[:b+1]

t1=np.array(t)[startIdx:stopIdx+1]
T1=np.array(T)[startIdx:stopIdx+1]
DSC1=np.array(DSC)[startIdx:stopIdx+1]
DDSC1=np.array(DSC)[startIdx:stopIdx+1]

###Raw Plots ###
##Heating Program
plt.plot(t,T)
plt.xlabel('time (min)')
plt.ylabel('heating temperature (°C)')
plt.grid(True)
plt.figure()

### Baseline fitting ###
xx=[T1[startIdx-startIdx],T1[stopIdx-startIdx]]
yy=[DSC1[startIdx-startIdx],DSC1[stopIdx-startIdx]]
coefficients = np.polyfit(xx, yy, 1)
plt.plot(T1,DSC1,label='DSC signal')
plt.plot(T,coefficients[0]*T+coefficients[1],label='baseline')
plt.xlabel('Temperature (°C)')
plt.ylabel('DSC signal (mW)')
plt.title('Exothermic peak')
plt.figure()
### Calculations of total heat of reaction and partial heats ###
DeltaH2=integrate.trapz(DSC1-coefficients[0]*T1+coefficients[1],T1)
H=integrate.cumtrapz(DSC1-coefficients[0]*T1+coefficients[1],T1,initial=0) #heat of reaction at time t - H(t) - computed via Trapezoidal method
alpha=H/(DeltaH2)                  #computation of the degree of cure
print(DeltaH2)
###Plot alpha vs t ###
plt.plot(t1,alpha)
plt.plot(t1,len(alpha)*[1])
plt.xlabel('time (min)')
plt.ylabel('degree of cure alpha (-)')
plt.grid(True)
plt.figure()
### Calculation of parameters ###
criterion1=0.1*DeltaH2
criterion2=0.9*DeltaH2
left_bndryIdx=(np.abs(H-criterion1)).argmin()
right_bndryIdx=(np.abs(H-criterion2)).argmin()
Tleft=T1[left_bndryIdx]
Tright=T1[right_bndryIdx]
Tinter=T1[left_bndryIdx:right_bndryIdx+1]
m=len(Tinter)
Tcalc=Tinter[np.linspace(0,len(Tinter)-1,150).astype(int)]
indexes=[left_bndryIdx]
indexes.append(m+left_bndryIdx)
alphaS=alpha[np.linspace(left_bndryIdx,len(Tinter)+left_bndryIdx-1,150).astype(int)]
op='n'
while op=='n':
    n=float(input('Order of reaction n ?\n'))
    int1=n*np.log(1-alphaS)
    lnr=np.log((DSC1+abs(min(DSC1))+0.001)/abs(DeltaH2))[np.linspace(left_bndryIdx,len(Tinter)+left_bndryIdx-1,150).astype(int)] #Logarithm of r=d(alpha/dt)
    lnK=lnr-int1
    invT=1/(np.array(Tcalc)+273.15)
    plt.plot(invT,lnK,label='Experimental data',linestyle=':')
    #b = estimate_coef(invT,lnK)
    b = stats.linregress(invT,lnK)
    print('n = '+str(n))
    print("Estimated coefficients:\n intercept = {}  \n slope = {}".format(b[1], b[0]))
    print('Estimated value of Activation Energy = {} J.mol-1, {} kcal.mol-1'.format(-b[0]*8.314, -b[0]*8.314/4184))
    print('Estimated value of Frequency Factor = {} s-1'.format(np.exp(b[1])))
    print('Standard error = {}'.format(b[4]))
    print('Correlation coefficient (r squared) = {}'.format(b[2]**2))
    y_pred = b[1] + b[0]*invT
    plt.plot(invT, y_pred,label='Linear regression with n='+str(n))
    plt.xlabel('1/T (1/K)')
    plt.ylabel('ln(d(\u03B1)/dt)-nln(1-\u03B1)')
    plt.legend()
    plt.show()
    op=input('Results ok? y=yes, n=no \n')

