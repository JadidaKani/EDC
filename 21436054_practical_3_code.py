import matplotlib.pyplot as plt
import numpy as np

#2a
Io = 1
wave = 1
L = 1*wave
beta = 2*np.pi/wave

# Values of z for each length
z1 = np.linspace((L*0.5)/2,-(L*0.5)/2) #creates a sequence of values that are evenly spaced within a specified range
z2 = np.linspace(L/2,-L/2)
z3 = np.linspace((L*1.25)/2,-(L*1.25)/2)
z4 = np.linspace((L*1.5)/2,-(L*1.5)/2)

# current at each length
y1 = Io*np.sin(beta*((L*0.5)/2-np.abs(z1)))
y2 = Io*np.sin(beta*(L/2-np.abs(z2)))
y3 = Io*np.sin(beta*((L*1.25)/2-np.abs(z3)))
y4 = Io*np.sin(beta*((L*1.5)/2-np.abs(z4)))

#subplots
fig,axes =plt.subplots(nrows=2,ncols=2) # plots subplots 2 rows and 2 columns

axes[0, 0].set_ylim(-1, 1)
axes[0, 1].set_ylim(-1, 1)
axes[1, 0].set_ylim(-1, 1)
axes[1, 1].set_ylim(-1, 1)

axes[0,0].plot(z1,y1)
axes[0,0].set_title('L=0.5\u03BB')

axes[0,1].plot(z2,y2)
axes[0,1].set_title('L=1\u03BB')

axes[1,0].plot(z3,y3)
axes[1,0].set_title('L=1.25\u03BB')

axes[1,1].plot(z4,y4)
axes[1,1].set_title('L=1.5\u03BB')

axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('Current I(z)')

axes[1, 0].set_xlabel('z')
axes[1, 0].set_ylabel('Current I(z)')

axes[1,1].set_xlabel('z')

plt.tight_layout()
plt.show()

#===========================================================================================
#2b

plt.axes(projection='polar')

Io = 1
wave = 1
L = 1.5*wave # change the variable according to the change in length
beta = 2*np.pi/wave
t = np.arange(0, 2*np.pi, 0.01) # plot graphs using radians

for rad in t:
    Es = abs(((np.cos((beta * L * np.cos(t)) / 2) - np.cos((beta * L) / 2) )/ np.sin(t)))
    plt.polar(t, Es, 'g.')

plt.title("L=1.5\u03BB")

plt.show()

#=====================================================================
# 2c
t = np.arange(0, 2*np.pi, 0.01)
print(np.max(1/np.sin(t)))
Io = 1
wave = 1
L = 1.5*wave # change the variable according to the change in length
beta = 2*np.pi/wave
t = np.arange(0, 2*np.pi, 0.01)

E = abs(((np.cos((beta * L * np.cos(t)) / 2) - np.cos((beta * L) / 2)) /(np.sin(t))))
divide = (E/abs(np.nanmax(E)))
E_db = 20*np.log10(divide)

plt.plot(t,E_db)

plt.xlabel("Angle in degrees")
plt.ylabel("Electric field pattern (dB)")
plt.title("L=1.5\u03BB")

plt.ylim(-30,0)
plt.show()

#=================================================================
#2d

Io = 1
wave = 1
L = 1.5*wave # change the variable according to the change in length
beta = 2*np.pi/wave
t = np.arange(0, 2*np.pi, 0.01)
P = ((1/2)*pow((np.cos((beta * L * np.cos(t)) / 2) - np.cos((beta * L) / 2)) / np.sin(t),2))
divide = 10 * np.log10(P/np.nanmax(P))
plt.plot(t,divide)

plt.xlabel("Angle in degrees")
plt.ylabel("Time average power density in dB")
plt.title("L=1.5\u03BB")

plt.show()
