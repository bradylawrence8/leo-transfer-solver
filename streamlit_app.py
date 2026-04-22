import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from orbittools import *

st.set_page_config(layout="wide")

st.subheader("Inputs:", divider="blue")
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Starting Orbit (Red):")
    ina = st.slider("Size (SMA):", 6500, 50000, value=7000, key=1)
    ine = st.slider("Shape (ECC): ", 0.0, 0.99, step=0.001, value=0.0, key=2)
    inw = st.slider("Orientation of Orbit (AOP): ", 0.0, math.pi*2, step=0.01, value=0.0, key=3)
    intheta = st.slider("Position in Orbit (TA):", 0.0, math.pi*2, step=0.01, value=0.0, key=4)
    

with col2:
    st.subheader("Ending Orbit (Blue):")
    ina2 = st.slider("Size (SMA):", 6500, 50000, value=42164, key=5)
    ine2 = st.slider("Shape (ECC): ", 0.0, 0.99, step=0.001, value=0.0, key=6)
    inw2 = st.slider("Orientation of Orbit (AOP): ", 0.0, math.pi*2, step=0.01, key=7)
    intheta2 = st.slider("Position in Orbit (TA):", 0.0, math.pi*2, step=0.01, value=math.pi/2, key=8)
        
mu = math.pi**2 * 4

col3, col4, col5 = st.columns([1, 1, 1])
with col3:
    tof_def = 15.0
    tof = st.slider("Time of Flight:", 0.1, 48.0, key=9, value=tof_def)
with col4:
    dvmax = st.slider("Maximum Delta-V:", 0.5, 10.0, 4.0, key=10)
with col5:
    searchres = st.slider("Solution Search Resolution (10^-n hr):", -3, 2, -1, key=11)

col6, col7 = st.columns([1, 1])
with col6:
    starttime = st.slider("Departure Time (hr):", 0.0, tof-0.1, 0.0, key=12)
with col7:
    endtime = st.slider("Arrival Time (hr):", 0.1-tof, 0.0, 0.0, key=13)


mu_earth = 398600
LU = 10000
TU = 2*math.pi*LU**1.5*mu_earth**-0.5
hrtotu = 3600/TU

r1, v1 = COE2State(ina/LU, ine, 0, 0, inw, intheta, mu)
r2, v2 = COE2State(ina2/LU, ine2, 0, 0, inw2, intheta2, mu)

plottype = st.checkbox("Only show transfer trajectory", value=True)
showearth = st.checkbox("Show Earth surface", value=False)
    

data = pd.DataFrame(columns=["Revolutions", "Orbit Type", "SMA (km)", "Eccentricity", "Delta-V (km/s)", "Closest Approach (km)", "Departure Time (hr)"])
res = 200
times = np.arange(starttime, tof+endtime+10**searchres, 10**searchres)
numsols = 0
deptimes = []
dvs = []

fig = plt.figure()
for j in times:
    theta = intheta + EAtoTA(KTE(mu_earth, ina, ine, j*3600, invKTE(mu_earth, ina, ine, TAtoEA(intheta, ine)), 1e-8), ine)
    r1, v1 = COE2State(ina/LU, ine, 0, 0, inw, theta, mu)

    try:
        mlist, v1list, v2list = mrlambert(r1, r2, mu, (tof-j)*hrtotu)
    except (ValueError, UnboundLocalError):
        continue
    else:
        size = np.size(mlist)
        mmax = mlist[size-1]

        for i in range(size):
            dvm = dvmax
            bestsol = []
            v1t = v1list[i]*LU/TU
            v2t = v2list[i]*LU/TU
            energy = np.dot(v1t, v1t)/2-mu/np.linalg.norm(r1)
            a = -mu/(2*energy)
            h = np.cross(r1, v1t)
            p = np.dot(h, h)/mu
            e = math.sqrt(1+2*p*energy/mu)
            if mlist[i] > 0:
                rp = p/(1+e)
            else:
                rp = min([np.linalg.norm(r1), np.linalg.norm(r2)])
            dv1 = v1t-v1
            dv2 = v2t-v2
            dv = (np.linalg.norm(dv1)+np.linalg.norm(dv2))*LU/TU
            #print(dv)
            if dv<dvmax:
                numsols += 1
                if dv<dvm:
                    dvm = dv
                    bestsol = [j, dv]
                if energy < 0:
                    otype = "Elliptic"
                else:
                    otype = "Hyperbolic"
                tablerow = pd.DataFrame([{"Revolutions":mlist[i], "Orbit Type":otype, "SMA (km)":round(a*LU), "Eccentricity":e, "Delta-V (km/s)":dv, "Closest Approach (km)":round(rp*LU), "Departure Time (hr)":j}])
                data = pd.concat([data, tablerow], ignore_index=True)
                if mmax > 0:
                    if plottype:
                        mrplotSolution(r1*LU, r2*LU, v1list[i]*LU/TU, mu_earth, res, (mlist[i]/(mmax), 0, 1-mlist[i]/mmax), mlist[i])
                    else:
                        plotOrbit(r1*LU, v1list[i]*LU/TU, mu_earth, res, (mlist[i]/mmax, 0, 1-mlist[i]/mmax), "")
                else:
                    if plottype:
                        plotSolution(r1*LU, r2*LU, v1list[0]*LU/TU, mu_earth, res, (1, 0, 1))
                    else:
                        plotOrbit(r1, v1list[0]*LU/TU, mu, res, (1, 0, 1), "")
                #plt.plot(np.array([0, r1[0]*LU]), np.array([0, r1[1]*LU]), color=(1, 0, 0), linestyle='dashed')
            if not bestsol == []:
                deptimes.append(bestsol[0])
                dvs.append(bestsol[1])

plotOrbit(r1*LU, v1*LU/TU, mu_earth, 500, (1, 0, 0), "Dotted")
plotOrbit(r2*LU, v2*LU/TU, mu_earth, 500, (0, 0, 1), "Dotted")
plt.plot(np.array([0, r2[0]*LU]), np.array([0, r2[1]*LU]), color=(0, 0, 1), linestyle='dashed')
if showearth:
    R_Earth = 6378
    x_earth = R_Earth*np.cos(np.linspace(0, math.pi*2+0.01, 100))
    y_earth = R_Earth*np.sin(np.linspace(0, math.pi*2+0.01, 100))
    plt.fill(x_earth, y_earth, 'g')
plt.plot(0, 0, 'gx')
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')

#plt.show()


#print(data)
st.subheader("Outputs:", divider="red")
st.write("Number of found solutions: ", numsols)
st.pyplot(fig)
if numsols > 1:
    fig2 = plt.figure()
    plt.scatter(deptimes, dvs)
    plt.xlabel("Departure Time (hr)")
    plt.ylabel("Delta-V (km/s)")
    st.pyplot(fig2)
st.table(data)

