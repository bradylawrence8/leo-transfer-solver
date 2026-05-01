import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from orbittools import *

st.set_page_config(layout="wide")

tab1, tab2 = st.tabs(["Tools", "Help"])

with tab1:

    st.subheader("Inputs:", divider="blue")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Starting Orbit (Red):")
        ina = st.slider("Semimajor Axis (km):", 6500, 50000, value=7000, key=1)
        ine = st.slider("Eccentricity: ", 0.0, 0.99, step=0.001, value=0.0, key=2)
        inw = st.slider("Argument of Periapsis (rad): ", 0.0, math.pi*2, step=0.01, value=0.0, key=3)
        ini = st.slider("Inclination (rad):", 0.0, math.pi, step=0.01, value=0.0, key=4)
        ino = st.slider ("RAAN (rad):", 0.0, math.pi*2, step=0.01, value=0.0, key=5)
        intheta = st.slider("True Anomaly (rad):", 0.0, math.pi*2, step=0.01, value=0.0, key=6)
        

    with col2:
        st.subheader("Ending Orbit (Blue):")
        ina2 = st.slider("Semimajor Axis (km):", 6500, 50000, value=42164, key=7)
        ine2 = st.slider("Eccentricity: ", 0.0, 0.99, step=0.001, value=0.0, key=8)
        inw2 = st.slider("Argument of Periapsis (rad): ", 0.0, math.pi*2, step=0.01, key=9)
        ini2 = st.slider("Inclination (rad):", 0.5, math.pi, step=0.01, value=0.0, key=10)
        ino2 = st.slider ("RAAN (rad):", 0.0, math.pi*2, step=0.01, value=0.0, key=11)
        intheta2 = st.slider("True Anomaly (rad):", 0.0, math.pi*2, step=0.01, value=math.pi/2, key=12)
            
    mu = math.pi**2 * 4

    col3, col4, col5 = st.columns([1, 1, 1])
    with col3:
        tof_def = 15.0
        tof = st.slider("Time of Flight:", 0.1, 48.0, key=13, value=tof_def)
    with col4:
        dvmax = st.slider("Maximum Delta-V:", 0.5, 10.0, 4.0, key=14)
    with col5:
        searchres = st.slider("Solution Search Resolution (10^-n hr):", -6, 2, -1, key=15)

    col6, col7 = st.columns([1, 1])
    with col6:
        starttime = st.slider("Departure Time (hr):", 0.0, tof-0.1, 0.0, key=16)
    with col7:
        endtime = st.slider("Arrival Time (hr):", 0.1-tof, 0.0, 0.0, key=17)


    mu_earth = 398600
    LU = 10000
    TU = 2*math.pi*LU**1.5*mu_earth**-0.5
    hrtotu = 3600/TU

    r1, v1 = COE2State(ina/LU, ine, ini, ino, inw, intheta, mu)
    r2, v2 = COE2State(ina2/LU, ine2, ini2, ino2, inw2, intheta2, mu)

    plottype = st.checkbox("Only show transfer trajectory", value=True)
    # showearth = st.checkbox("Show Earth surface", value=False)
    sorting = st.checkbox("Sort by Delta-V", value=True)
        

    data = pd.DataFrame(columns=["Revolutions", "Orbit Type", "SMA (km)", "Eccentricity", "Delta-V (km/s)", "Closest Approach (km)", "Departure Time (hr)"])
    fulldata = pd.DataFrame(columns=["Revolutions", "Orbit Type", "SMA (km)", "Eccentricity", "Delta-V (km/s)", "Closest Approach (km)", "Departure Time (hr)", "Initial Position (km)", "Initial Velocity (km/s)", "Final Position (km)", "Final Velocity (km/s)"])
    #data_uf = pd.DataFrame(columns=["Revolutions", "Orbit Type", "SMA (km)", "Eccentricity", "Delta-V (km/s)", "Closest Approach (km)", "Departure Time (hr)"])
    res = 200
    times = np.arange(starttime, tof+endtime+10**searchres, 10**searchres)
    numsols = 0
    deptimes = []
    dvs = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in times:
        theta = intheta + EAtoTA(KTE(mu_earth, ina, ine, j*3600, invKTE(mu_earth, ina, ine, TAtoEA(intheta, ine)), 1e-8), ine)
        r1, v1 = COE2State(ina/LU, ine, ini, ino, inw, theta, mu)

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
                if energy < 0:
                    otype = "Elliptic"
                else:
                    otype = "Hyperbolic"
                if dv<dvmax:
                    if round(rp*LU) > 6378:
                        numsols += 1
                        if dv<dvm:
                            dvm = dv
                            bestsol = [j, dv]
                        tablerow = pd.DataFrame([{"Revolutions":mlist[i], "Orbit Type":otype, "SMA (km)":round(a*LU), "Eccentricity":e, "Delta-V (km/s)":dv, "Closest Approach (km)":round(rp*LU), "Departure Time (hr)":j}])
                        outputrow = pd.DataFrame([{"Revolutions":mlist[i], "Orbit Type":otype, "SMA (km)":(a*LU), "Eccentricity":e, "Delta-V (km/s)":dv, "Closest Approach (km)":(rp*LU), "Departure Time (hr)":j, "Initial Position (km)":(r1*LU), "Initial Velocity (km/s)":v1t, "Final Position (km)":(r2*LU), "Final Velocity (km/s)":v2t}])
                        data = pd.concat([data, tablerow], ignore_index=True)
                        fulldata = pd.concat([fulldata, outputrow], ignore_index=True)
                        if plottype:
                            threedimsolutionplot(r1*LU, r2*LU, v1list[i]*LU/TU, mlist[i], mu_earth, res, (0.4, 0, 0.8))
                        else:
                            threedimorbitplot(r1*LU, v1list[i]*LU/TU, mu_earth, res, (0.4, 0, 0.8), "")
                        #plt.plot(np.array([0, r1[0]*LU]), np.array([0, r1[1]*LU]), color=(1, 0, 0), linestyle='dashed')
                    # else:
                    #     tablerow = pd.DataFrame([{"Revolutions":mlist[i], "Orbit Type":otype, "SMA (km)":round(a*LU), "Eccentricity":e, "Delta-V (km/s)":dv, "Closest Approach (km)":round(rp*LU), "Departure Time (hr)":j}])
                    #     data_uf = pd.concat([data, tablerow], ignore_index=True)
                if not bestsol == []:
                    deptimes.append(bestsol[0])
                    dvs.append(bestsol[1])

    threedimorbitplot(r1*LU, v1*LU/TU, mu_earth, 500, (1, 0, 0), "Dotted")
    threedimorbitplot(r2*LU, v2*LU/TU, mu_earth, 500, (0, 0, 1), "Dotted")
    plt.plot(np.array([0, r2[0]*LU]), np.array([0, r2[1]*LU]), np.array([0, r2[2]*LU]), color=(0, 0, 1), linestyle='dashed')
    # if showearth:
    #     R_Earth = 6378
    #     x_earth = R_Earth*np.multiply(np.cos(np.linspace(0, math.pi*2+0.01, 100)), np.sin(np.linspace(0, math.pi+0.01, 100)))
    #     y_earth = R_Earth*np.multiply(np.sin(np.linspace(0, math.pi*2+0.01, 100)), np.sin(np.linspace(0, math.pi+0.01, 100)))
    #     z_earth = R_Earth*np.cos(np.linspace(0, math.pi+0.01, 100))
    #     plt.fill(x_earth, y_earth, 'g')
    plt.plot(0, 0, 0, 'gx')
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.zlabel("Z")
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
    st.subheader("List of solutions:")
    data_sorted = data.sort_values(by="Delta-V (km/s)")
    #data_uf_sorted = data_uf.sort_values(by="Delta-V (km/s)")
    if sorting:
        st.table(data_sorted)
        # st.subheader("Solutions that intersect Earth:")
        # st.table(data_uf_sorted)
    else:
        st.table(data)
        # st.subheader("Solutions that intersect Earth:")
        # st.table(data_uf)
    csv = fulldata.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", data=csv, file_name="transfers.csv", mime="text/csv")

with tab2:
    st.header("User Guide", divider="blue")
    st.subheader("Coming soon, for now use the following link:")
    st.page_link("https://docs.google.com/document/d/1EUUm4tyJBIBv5B3OnrY04V0w0eEBr8cujETP0FZFKns/edit?usp=sharing", "Documentation")

