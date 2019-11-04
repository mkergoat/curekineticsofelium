##Packages Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate,integrate,signal,stats,optimize
import os
import matplotlib2tikz

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
    SampleName=r[4].split('\t')[1]
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
    data.close()
    return(info,prog,t,T,DSC,DDSC,sample_mass,SampleName)

###Main Program
print('Cure Kinetics Evaluation Program')
print('Diffusion Controlled Model Version')
print('MK, Summer 2019')
str1=input('Do you have a set-up-file ? if yes, enter the filename, if no, enter n:')
if str1=='n':
    filename=input('Enter filename?\n') #Asks for the filename
    path=input('Enter path to data?\n')           #Ask where to search the file
if str1=='*':
    data=open('sui.txt','r')
    r=data.readlines()
    filename=str(r[1].rstrip())
    path=str(r[2].rstrip())
    preferency1=str(r[3].rstrip())
    preferency2=str(r[4].rstrip())
    starttime=str(r[5].rstrip())
    stopttime=str(r[6].rstrip())
    DeltaHDynamic=str(r[7].rstrip())
    data.close()
else:
    data=open(str1,'r')
    r=data.readlines()
    filename=str(r[1].rstrip())
    path=str(r[2].rstrip())
    preferency1=str(r[3].rstrip())
    preferency2=str(r[4].rstrip())
    starttime=str(r[5].rstrip())
    stopttime=str(r[6].rstrip())
    DeltaHDynamic=str(r[7].rstrip())
    data.close()
os.chdir(path)                  #Changes path to get data
print('Path successfully changed to '+path)
info,prog,t,T,DSC,DDSC,sample_mass,SampleName=traitement(filename) #Main call
print('File '+ filename+' successfully opened')
t=np.array(t)                   #Transforms lists into Numpy Arrays for easier use
T=np.array(T)
DSC=np.array(DSC)
DDSC=np.array(DDSC)

##Plot DSC + DDSC with 2 axis
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (min)')
ax1.set_ylabel('DSC', color=color)
ax1.plot(t, DSC, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('d(DSC)/dt', color=color)  # we already handled the x-label with ax1
ax2.plot(t, DDSC, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.figure()

sampling_time=0.025/3 #min
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

###Re-limitation around peak by user ###
if str1=='n':
    start=float(input('Starting point for DeltaH Static evaluation?\n'))
    stop=float(input('Stopping point for DeltaH Static evaluation?\n'))
    DeltaHDynamic=float(input('Delta H evaluated in dynamic mode ?'))
else:
    start=float(starttime)
    start2=start+5
    stop=float(stopttime)
    DeltaHDynamic=float(DeltaHDynamic)
startIdx=(np.abs(t-start)).argmin()
stopIdx=(np.abs(t-stop)).argmin()
t1=np.array(t)[startIdx:stopIdx+1]
T1=np.array(T)[startIdx:stopIdx+1]
DSC1=np.array(DSC)[startIdx:stopIdx+1]
DDSC1=np.array(DSC)[startIdx:stopIdx+1]
###Raw Plots ###
##Heating Program
##plt.plot(t,T)
##plt.xlabel('time (min)')
##plt.ylabel('Cell temp (°C)')
##plt.grid(True)
##plt.figure()

### Baseline fitting ###
xx=[t1[startIdx-startIdx],t1[stopIdx-startIdx]]
yy=[DSC1[startIdx-startIdx],DSC1[stopIdx-startIdx]]
coefficients = np.polyfit(xx, yy, 1)
##plt.plot(t1,DSC1,label='DSC signal')
##plt.plot(t,coefficients[0]*t+coefficients[1],label='baseline')
##plt.plot(t1,DSC1)
##plt.xlabel('Time (s)')
##plt.ylabel('DSC signal (microW)')
##plt.title('Exothermic Peak')
##plt.figure()
baseline=coefficients[0]*t1+coefficients[1]

### Calculations of total heat of reaction and partial heats ###
t1Integration=t1*60
DeltaHStatic=integrate.trapz(DSC1-baseline,t1Integration)
DeltaHStaticNorm=(DeltaHStatic*10**(-3))/(sample_mass)#normalized
dBeta=(DSC1-baseline)/DeltaHStatic
alpha=(DeltaHStaticNorm/DeltaHDynamic)*integrate.cumtrapz(dBeta,t1Integration,initial=0)
###Plot alpha vs t ###
plt.plot(t1Integration,alpha)
plt.plot(t1Integration,len(alpha)*[1])
plt.xlabel('time (s)')
plt.ylabel('degree of cure \u03B1 (-)')
plt.grid(True)
plt.figure()
plt.show()
DSC1Norm=(DSC1*10**(-3))/(sample_mass)
N=[0.5,1,1.5,2,2.5]
op='n'
while op!='y':
    m=float(input('Order of reaction m ?\n'))
    x=alpha**m
    for n in N:
        y=(DSC1Norm/(DeltaHDynamic))*(1-alpha)**(-n)
        plt.plot(x,y,label='n='+str(n),linestyle=':')
        plt.legend()
    plt.xlabel('\u03B1^m (-)')
    plt.ylabel('(d\u03B1/dt)/(1-\u03B1)^n')
    plt.show()
    op=str(input('Results ok ? y/n'))
n=float(input('n value retained\n'))

#### Diffusion model complete

def model(alpha,K1,K2,alphaf,b):
    fd=2/(1+np.exp((alpha-alphaf)/b))-1

    return((K1+K2*alpha**m)*fd)
x=alpha**m
DSC2Norm=DSC1Norm
y=(DSC2Norm/DeltaHDynamic)*1/(1-x)**(n)
ydata=y
xdata=x
bounds=([0,0,0,0],[np.inf,np.inf,np.inf,np.inf])
OptimalParameters,ParametersCovariance=optimize.curve_fit(model,xdata,ydata,method='lm')
StandardDeviations=np.sqrt(np.diag(ParametersCovariance))
###Following lines to be uncommented only if strange results happens with constants values
### (the code will modify the start and end value to search for positives values for k_1 at least)
##while (OptimalParameters[0]<0):
##    start=start+0.1
##    print('t start ='+str(start))
##    startIdx=(np.abs(t-start)).argmin()
##    stopIdx=(np.abs(t-stop)).argmin()
##    print('idx  t start ='+str(startIdx))
##    stopIdx=(np.abs(t-stop)).argmin()
##    t1=np.array(t)[startIdx:stopIdx+1]
##    DSC1=np.array(DSC)[startIdx:stopIdx+1]
##    DSC1Norm=(DSC1*10**(-3))/(sample_mass)
##    xx=[t1[startIdx-startIdx],t1[stopIdx-startIdx]]
##    yy=[DSC1[startIdx-startIdx],DSC1[stopIdx-startIdx]]
##    coefficients = np.polyfit(xx, yy, 1)
##    baseline=coefficients[0]*t1+coefficients[1]
##    t1Integration=t1*60
##    DeltaHStatic=integrate.trapz(DSC1-baseline,t1Integration)
##    DeltaHStaticNorm=(DeltaHStatic*10**(-3))/(sample_mass)#normalized
##    dBeta=(DSC1-baseline)/DeltaHStatic
##    alpha=(DeltaHStaticNorm/DeltaHDynamic)*integrate.cumtrapz(dBeta,t1Integration,initial=0)
##    x=alpha**m
##    y=(DSC1Norm/DeltaHDynamic)*1/(1-alpha)**(n)
##    ydata=y
##    xdata=x
##    OptimalParameters,ParametersCovariance=optimize.curve_fit(model,xdata,ydata,method='lm')
##    print('calculation done')
##    StandardDeviations=np.sqrt(np.diag(ParametersCovariance))
plt.plot(t1,DSC1,label='DSC signal')
plt.plot(t,coefficients[0]*t+coefficients[1],label='baseline')
plt.plot(t1,DSC1)
plt.xlabel('Time (s)')
plt.ylabel('DSC signal (microW)')
plt.title('Exothermic Peak')
plt.figure()
print('Full diffusion-controlled model')
print("K1,K2,alphaf,b")
print(OptimalParameters)
print('Standard deviations on each parameters')
print(StandardDeviations)
K1,K2,alphaf,b=OptimalParameters
plt.plot(xdata,ydata,label='experimental data',linestyle=':')
plt.plot(xdata,model(xdata,K1,K2,alphaf,b),label='model prediction')
plt.xlabel('alpha^'+str(m)+' (-)')
plt.ylabel('(d\u03B1/dt)/(1-\u03B1)^n')
plt.legend()
plt.figure()
F=(DSC2Norm/DeltaHDynamic)/((K1+K2*alpha**m)*(1-alpha**n))
plt.plot(alpha,F)
plt.xlabel('alpha^'+str(m)+' (-)')
plt.ylabel('(d\u03B1/dt)/((K_1+K_2\u03B1^m)(1-\u03B1)^n)')
plt.figure()
plt.show()
f=open('res.txt','a')
f.write(SampleName+'\n')
f.write(str(m)+'\n')
f.write(str(n)+'\n')
f.write('Optimal Paramaters \n')
f.write(str(OptimalParameters)+'\n')
f.write(str(StandardDeviations)+'\n')
f.close()
