import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as an
import math

# base Lambert's problem solver, uses second order Householder method (Izzo 2014). returns velocity vector at initial and final positions
def lambert(r1v, r2v, mu, TOF):
    def F(a, b, c, x):
        return 1 + (a*b/c)*x + (a*(a+1)*b*(b+1)/(c*(c+1)))*(x**2/2) + (a*(a+1)*(a+2)*b*(b+1)*(b+2)/(c*(c+1)*(c+2)))*(x**3/6)

    def Y(x, l):
        return math.sqrt(1-l*l*(1-x*x))

    def N(x, l):
        return Y(x, l) - l*x

    def S(x, l):
        return (1-l-x*N(x, l))/2

    def Q(x, l):
        return 4*F(3, 1, 5/2, S(x, l))/3

    def T(x, l):
        if (x > 0.9) & (x < 1.1):
            return (N(x, l)**3*Q(x, l)+4*l*N(x, l))/2
        else:
            y = Y(x, l)
            if x < 1:
                psi = math.acos(x*y+l*(1-x*x))
            else:
                psi = math.acosh(x*y-l*(x*x-1))
            return (1/(1-x*x))*(psi/math.sqrt(abs(1-x*x))-x+l*y)
    
    def dT(x, l, T):
        return (3*T*x-2+2*l**3*x/Y(x, l))/(1-x**2)

    def d2T(x, l, T, dT):
        return (3*T+5*x*dT+2*(1-l**2)*l**3/(Y(x, l)**3))/(1-x**2)

    def d3T(x, l, T, dT, d2T):
        return (7*x*d2T+8*dT-6*(1-l**2)*l**5*x/(Y(x, l)**5))/(1-x**2)     

    if np.linalg.norm(np.cross(r1v, r2v)) == 0:
        a = (np.linalg.norm(r1v)+np.linalg.norm(r2v))/2
        r1 = np.linalg.norm(r1v)
        r2 = np.linalg.norm(r2v)
        ra = max(r1, r2)
        e = ra/a-1
        p = a*(1-e**2)
        h = math.sqrt(mu*p)
        e_h = np.array([0, 0, 1])
        v1v = np.cross(e_h, r1v/r1)
        v2v = np.cross(e_h, r2v/r2)
        v1 = (h/r1)*v1v
        v2 = (h/r2)*v2v
    else:
        r1 = np.linalg.norm(r1v)
        r2 = np.linalg.norm(r2v)
        c = math.sqrt(r1**2+r2**2-2*np.dot(r1v, r2v))
        s = (r1+r2+c)/2

        g = math.sqrt(s*mu/2)
        p = (r1-r2)/c
        o = math.sqrt(1-p**2)

        e_r1 = r1v/r1
        e_r2 = r2v/r2
        e_h = np.cross(e_r1, e_r2)/(np.linalg.norm(np.cross(e_r1, e_r2)))
        if (r1v[0]*r2v[1]-r1v[1]*r2v[0]) < 0:
            e_t1 = np.cross(e_r1, e_h)
            e_t2 = np.cross(e_r2, e_h)
            l = -math.sqrt(1-c/s)
        else:
            e_t1 = np.cross(e_h, e_r1)
            e_t2 = np.cross(e_h, e_r2)
            l = math.sqrt(1-c/s)

        Td = math.sqrt(2*mu/s**3)*TOF
        T1 = 2*(1-l**3)/3
        T0 = math.acos(l) + l*math.sqrt(1-l**2)

        if Td < T1:
            x0 = (T1/Td)-1
        elif Td < T0:
            x0 = pow(T0/Td, math.log2(T1/T0))-1
        else:
            x0 = pow(T0/Td, 2/3)-1

        x_ite = np.linspace(0, 0, 10)

        for i in range(3):
            Ts = T(x0, l)
            dTs = dT(x0, l, Ts)
            d2Ts = d2T(x0, l, Ts, dTs)
            Ts = Ts - Td
            x1 = x0 - (Ts*dTs)/(dTs**2-Ts*d2Ts/2)
            x_ite[i] = x1
            x0 = x1

        y1 = Y(x1, l)

        v_r1 = g*((l*y1-x1)-p*(l*y1+x1))/r1
        v_r2 = -g*((l*y1-x1)+p*(l*y1+x1))/r2
        v_t1 = g*o*(y1+l*x1)/r1
        v_t2 = g*o*(y1+l*x1)/r2

        v1 = v_r1*e_r1 + v_t1*e_t1
        v2 = v_r2*e_r2 + v_t2*e_t2

    return v1, v2

# base orbit plotting function, requires one state and gravitational parameter. res input is plot resolution, c is a color in (r,g,b) form on a 0-1 scale
def plotOrbit(r1, v1, mu, res, c, style):
    hv = np.cross(r1, v1)
    E = np.linalg.norm(v1)**2/2-mu/np.linalg.norm(r1)
    a = -mu/(2*E)
    ev = np.cross(v1, hv)/mu - r1/np.linalg.norm(r1)
    if np.linalg.norm(np.cross(ev, r1)) == 0:
        p = np.linalg.norm(hv)**2/mu
        e = math.sqrt(1-p/a)
        if np.linalg.norm(r1) < a:
            periapsis = r1
            w = 0
        else:
            periapsis = -r1 * (a*(1-e)/np.linalg.norm(r1))
            w = math.pi
    else:
        e = np.linalg.norm(ev)
        p = a*(1-e**2)
        periapsis = ev/e * (p/(1+e))
        w = math.acos(periapsis[0]/np.linalg.norm(periapsis)) * -np.sign(periapsis[1])
    if e > 0.5:
        res = 100*res
    elif e > 0.9:
        res = 10000*res
    n = res
    theta = np.linspace(0, 2*math.pi+0.01, num=n)
    x = np.linspace(0, 0, num=n)
    y = np.linspace(0, 0, num=n)
    for i, t in np.ndenumerate(theta):
        r = p/(1+e*math.cos(t))
        x[i] = r*math.cos(t)*math.cos(w)+r*math.sin(t)*math.sin(w)
        y[i] =-r*math.cos(t)*math.sin(w)+r*math.sin(t)*math.cos(w)
    if style=="Dashed":
        plt.plot(x, y, color=c, linestyle="dashed")
    elif style=="Dotted":
        plt.plot(x, y, color=c, linestyle="dotted")
    else:
        plt.plot(x, y, color=c)

# modified version of plotOrbit, only plots the actual trajectory, not the full orbit. requires an extra input, r2 (final position)
def plotSolution(r1, r2, v1, mu, res, c):
    hv = np.cross(r1, v1)
    E = np.linalg.norm(v1)**2/2-mu/np.linalg.norm(r1)
    a = -mu/(2*E)
    ev = np.cross(v1, hv)/mu - r1/np.linalg.norm(r1)
    if np.linalg.norm(np.cross(ev, r1)) == 0:
        p = np.linalg.norm(hv)**2/mu
        e = math.sqrt(1-p/a)
        if np.linalg.norm(r1) < a:
            periapsis = r1
            w = 0
        else:
            periapsis = -r1 * (a*(1-e)/np.linalg.norm(r1))
            w = math.pi
    else:
        e = np.linalg.norm(ev)
        p = a*(1-e**2)
        periapsis = ev/e * (p/(1+e))
        w = math.acos(periapsis[0]/np.linalg.norm(periapsis)) * (-periapsis[1]/abs(periapsis[1]))
    if e > 0.5:
        res = 100*res
    elif e > 0.9:
        res = 10000*res
    n = res
    if np.linalg.norm(ev) < 1e-12:
        w = 0
        t1 = 0
        t2 = math.acos(np.dot(r1,r2)/(np.linalg.norm(r1)**2))
    else:
        t1 = math.acos((p-np.linalg.norm(r1))/(np.linalg.norm(r1)*e))
        if np.dot(np.cross(r1,periapsis), np.cross(r1, v1))>0:
            t1 = 2*math.pi - t1
        t2 = math.acos((p-np.linalg.norm(r2))/(np.linalg.norm(r2)*e))
        if np.dot(np.cross(r2,periapsis), np.cross(r1, v1))>0:
            t2 = 2*math.pi - t2
        if t1 > t2:
            t1 = t1 - 2*math.pi
    theta = np.linspace(t1, t2, num=n)
    x = np.linspace(0, 0, num=n)
    y = np.linspace(0, 0, num=n)
    for i, t in np.ndenumerate(theta):
        r = p/(1+e*math.cos(t))
        x[i] = r*math.cos(t)*math.cos(w)+r*math.sin(t)*math.sin(w)
        y[i] =-r*math.cos(t)*math.sin(w)+r*math.sin(t)*math.cos(w)
    plt.plot(x, y, color=c)

# creates short animation of the entire solution orbit, for a full period
def animateOrbit(r1, v1, mu, res):
    fig, ax = plt.subplots()
    plotOrbit(r1, v1, mu, res, (0, 1, 1))
    plt.scatter(0, 0, marker='x', c=(0, 0, 0))
    hv = np.cross(r1, v1)
    E = np.linalg.norm(v1)**2/2-mu/np.linalg.norm(r1)
    a = -mu/(2*E)
    ev = np.cross(v1, hv)/mu - r1/np.linalg.norm(r1)
    if np.linalg.norm(np.cross(ev, r1)) == 0:
        p = np.linalg.norm(hv)**2/mu
        e = math.sqrt(1-p/a)
        if np.linalg.norm(r1) < a:
            periapsis = r1
            w = 0
        else:
            periapsis = -r1 * (a*(1-e)/np.linalg.norm(r1))
            w = math.pi
    else:
        e = np.linalg.norm(ev)
        p = a*(1-e**2)
        periapsis = ev/e * (p/(1+e))
        w = math.acos(periapsis[0]/np.linalg.norm(periapsis)) * (-periapsis[1]/abs(periapsis[1]))
    n = res
    theta = np.linspace(0, 2*math.pi+0.01, num=n)
    xc = np.linspace(0, 0, num=n)
    yc = np.linspace(0, 0, num=n)
    for i, t in np.ndenumerate(theta):
        r = p/(1+e*math.cos(t))
        xc[i] = r*math.cos(t)*math.cos(w)+r*math.sin(t)*math.sin(w)
        yc[i] =-r*math.cos(t)*math.sin(w)+r*math.sin(t)*math.cos(w)
    sc = ax.scatter(xc, yc)

    def update(frame):
        x = xc[frame]
        y = yc[frame]
        data = np.stack([x,y]).T
        sc.set_offsets(data)

    anim = an.FuncAnimation(fig=fig, func=update, frames=n, interval=10000/n)
    plt.show()

# animates just the true trajectory. uses time of flight input to save on computing time
def animateSolution(r1, r2, v1, mu, tof, res):
    hv = np.cross(r1, v1) # km^2/s
    ev = np.cross(v1, hv)/mu-r1/np.linalg.norm(r1)
    e = np.linalg.norm(ev)
    energy = np.dot(v1,v1)/2-mu/np.linalg.norm(r1)
    a = -mu/(2*energy)
    period = 2*math.pi*math.sqrt(a**3/mu)
    if np.linalg.norm(np.cross(ev, r1)) == 0:
        p = np.linalg.norm(hv)**2/mu
        e = math.sqrt(1-p/a)
        if np.linalg.norm(r1) < a:
            periapsis = r1
            w = 0
        else:
            periapsis = -r1 * (a*(1-e)/np.linalg.norm(r1))
            w = math.pi
    else:
        e = np.linalg.norm(ev)
        p = a*(1-e**2)
        periapsis = ev/e * (p/(1+e))
        w = math.acos(periapsis[0]/np.linalg.norm(periapsis)) * (-periapsis[1]/abs(periapsis[1]))
    n = res
    if np.linalg.norm(ev) < 1e-12:
        w = 0
        t1 = 0
        t2 = math.acos(np.dot(r1,r2)/(np.linalg.norm(r1)**2))
    else:
        t1 = math.acos((p-np.linalg.norm(r1))/(np.linalg.norm(r1)*e))
        if np.dot(np.cross(r1,periapsis), np.cross(r1, v1))>0:
            t1 = 2*math.pi - t1
        t2 = math.acos((p-np.linalg.norm(r2))/(np.linalg.norm(r2)*e))
        if np.dot(np.cross(r2,periapsis), np.cross(r1, v1))>0:
            t2 = 2*math.pi - t2
        if t1 > t2:
            t1 = t1 - 2*math.pi

    E1 = TAtoEA(t1, e)
    time1 = invKTE(mu, a, e, E1)
    time2 = time1 + tof
    
    t = np.linspace(time1, time2, n)

    EA = np.zeros(n)
    TA = np.zeros(n)
    r = np.zeros(n)
    for i, time in np.ndenumerate(t):
        EA[i] = KTE(mu, a, e, time, 0, 10**-5)
        TA[i] = EAtoTA(EA[i], e)
        r[i] = ((np.linalg.norm(hv) ** 2)/mu)/(1+e*math.cos(TA[i]))

    xc = np.multiply(r,np.cos(TA))
    yc = np.multiply(r,np.sin(TA))
    for i, ts in np.ndenumerate(t):
        r = p/(1+e*math.cos(TA[i]))
        xc[i] = r*math.cos(TA[i])*math.cos(w)+r*math.sin(TA[i])*math.sin(w)
        yc[i] =-r*math.cos(TA[i])*math.sin(w)+r*math.sin(TA[i])*math.cos(w)
    return xc, yc

# multi-revolution lambert's problem solver. returns three lists, being first a list of each orbit's number of revolutions (mlist), then velocity vectors.
def mrlambert(r1v, r2v, mu, TOF):
    def F(a, b, c, x):
        return 1 + (a*b/c)*x + (a*(a+1)*b*(b+1)/(c*(c+1)))*(x**2/2) + (a*(a+1)*(a+2)*b*(b+1)*(b+2)/(c*(c+1)*(c+2)))*(x**3/6)

    def Y(x, l):
        return math.sqrt(1-l*l*(1-x*x))

    def N(x, l):
        return Y(x, l) - l*x

    def S(x, l):
        return (1-l-x*N(x, l))/2

    def Q(x, l):
        return 4*F(3, 1, 5/2, S(x, l))/3

    def T(x, l, M):
        if (x > 0.9) & (x < 1.1):
            return (N(x, l)**3*Q(x, l)+4*l*N(x, l))/2
        else:
            y = Y(x, l)
            if x < 1:
                psi = math.acos(x*y+l*(1-x*x))
            else:
                psi = math.acosh(x*y-l*(x*x-1))
            return (1/(1-x*x))*((psi + M*math.pi)/math.sqrt(abs(1-x*x))-x+l*y)
    
    def dT(x, l, T):
        return (3*T*x-2+2*l**3*x/Y(x, l))/(1-x**2)

    def d2T(x, l, T, dT):
        return (3*T+5*x*dT+2*(1-l**2)*l**3/(Y(x, l)**3))/(1-x**2)

    def d3T(x, l, T, dT, d2T):
        return (7*x*d2T+8*dT-6*(1-l**2)*l**5*x/(Y(x, l)**5))/(1-x**2)     
    
    def halley(x, l, t, M):
        x0 = x
        for i in range(3):
            Ts = T(x0, l, M)
            dTs = dT(x0, l, Ts)
            d2Ts = d2T(x0, l, Ts, dTs)
            Ts = Ts - t
            x1 = x0 - (Ts*dTs)/(dTs**2-Ts*d2Ts/2)
            x0 = x1
        return x1
    
    def halleyderiv(x, l, t):
        x0 = x
        for i in range(3):
            f = dT(x0, l, t)
            df = d2T(x0, l, t, f)
            d2f = d3T(x0, l, t, f, df)
            x1 = x0 - (f*df)/(df**2-f*d2f/2)
            x0 = x1
        return x1
    
    def findxy(l, t):
        M = math.floor(t/math.pi)
        T00 = math.acos(l)+l*math.sqrt(1-l**2)
        T0 = T00 + M*math.pi
        if (t < T0) and (M > 0):
            x1 = halleyderiv(0, l, T0)
            Ts = T(x1, l, M)
            if t < Ts:
                M -= 1
        T1 = 2*(1-l**3)/3

        if t < T1:
            x0 = (T1/t)-1
        elif t < T0:
            x0 = pow(T0/t, math.log2(t/T0))-1
        else:
            x0 = pow(T0/t, 2/3)-1

        xlist = []
        ylist = []
        mlist = []

        while M > 0:
            x0l = ((math.pi*(M+1)/(8*t))**(2/3)-1)/((math.pi*(M+1)/(8*t))**(2/3)+1)
            x0r = ((8*t/(M*math.pi))**(2/3)-1)/((8*t/(M*math.pi))**(2/3)+1)
            xl = halley(x0l, l, t, M)
            xr = halley(x0r, l, t, M)
            xlist.append(xl)
            ylist.append(Y(xl, l))
            xlist.append(xr)
            ylist.append(Y(xr, l))
            mlist.append(M)
            mlist.append(M)
            M = M - 1

        x = halley(x0, l, t, 0)
        xlist.append(x)
        ylist.append(Y(x, l))
        mlist.append(0)

        xlist.reverse()
        ylist.reverse()
        mlist.reverse()

        return mlist, xlist, ylist

    if np.linalg.norm(np.cross(r1v, r2v)) == 0:
        a = (np.linalg.norm(r1v)+np.linalg.norm(r2v))/2
        r1 = np.linalg.norm(r1v)
        r2 = np.linalg.norm(r2v)
        ra = max(r1, r2)
        e = ra/a-1
        p = a*(1-e**2)
        h = math.sqrt(mu*p)
        e_h = np.array([0, 0, 1])
        v1v = np.cross(e_h, r1v/r1)
        v2v = np.cross(e_h, r2v/r2)
        v1 = (h/r1)*v1v
        v2 = (h/r2)*v2v
    else:
        r1 = np.linalg.norm(r1v)
        r2 = np.linalg.norm(r2v)
        c = math.sqrt(r1**2+r2**2-2*np.dot(r1v, r2v))
        s = (r1+r2+c)/2

        g = math.sqrt(s*mu/2)
        p = (r1-r2)/c
        o = math.sqrt(1-p**2)

        e_r1 = r1v/r1
        e_r2 = r2v/r2
        e_h = np.cross(e_r1, e_r2)/(np.linalg.norm(np.cross(e_r1, e_r2)))
        if (r1v[0]*r2v[1]-r1v[1]*r2v[0]) < 0:
            e_t1 = np.cross(e_r1, e_h)
            e_t2 = np.cross(e_r2, e_h)
            l = -math.sqrt(1-c/s)
        else:
            e_t1 = np.cross(e_h, e_r1)
            e_t2 = np.cross(e_h, e_r2)
            l = math.sqrt(1-c/s)

        Td = math.sqrt(2*mu/s**3)*TOF
        
        mlist, xlist, ylist = findxy(l, Td)

        v1list = []
        v2list = []

        for x1, y1 in zip(xlist, ylist):
            v_r1 = g*((l*y1-x1)-p*(l*y1+x1))/r1
            v_r2 = -g*((l*y1-x1)+p*(l*y1+x1))/r2
            v_t1 = g*o*(y1+l*x1)/r1
            v_t2 = g*o*(y1+l*x1)/r2

            v1 = v_r1*e_r1 + v_t1*e_t1
            v2 = v_r2*e_r2 + v_t2*e_t2

            v1list.append(v1)
            v2list.append(v2)

    return mlist, v1list, v2list

# plots a multi-rev solution trajectory, showing the full orbit if M > 0. this could be simplified to if M == 0: plotSolution, else plotOrbit
def mrplotSolution(r1, r2, v1, mu, res, c, M):
    hv = np.cross(r1, v1)
    E = np.linalg.norm(v1)**2/2-mu/np.linalg.norm(r1)
    a = -mu/(2*E)
    ev = np.cross(v1, hv)/mu - r1/np.linalg.norm(r1)
    if np.linalg.norm(np.cross(ev, r1)) == 0:
        p = np.linalg.norm(hv)**2/mu
        e = math.sqrt(1-p/a)
        if np.linalg.norm(r1) < a:
            periapsis = r1
            w = 0
        else:
            periapsis = -r1 * (a*(1-e)/np.linalg.norm(r1))
            w = math.pi
    else:
        e = np.linalg.norm(ev)
        p = a*(1-e**2)
        periapsis = ev/e * (p/(1+e))
        w = math.acos(periapsis[0]/np.linalg.norm(periapsis)) * (-periapsis[1]/abs(periapsis[1]))
    if e > 0.5:
        res = 100*res
    elif e > 0.9:
        res = 10000*res
    n = res*(M+1)
    if np.linalg.norm(ev) < 1e-12:
        w = 0
        t1 = 0
        t2 = math.acos(np.dot(r1,r2)/(np.linalg.norm(r1)**2))
    else:
        t1 = math.acos((p-np.linalg.norm(r1))/(np.linalg.norm(r1)*e))
        if np.dot(np.cross(r1,periapsis), np.cross(r1, v1))>0:
            t1 = 2*math.pi - t1
        t2 = math.acos((p-np.linalg.norm(r2))/(np.linalg.norm(r2)*e))
        if np.dot(np.cross(r2,periapsis), np.cross(r1, v1))>0:
            t2 = 2*math.pi - t2
        if t1 > t2:
            t1 = t1 - 2*math.pi
    theta = np.linspace(t1, t2 + 2*M*math.pi, num=n)
    x = np.linspace(0, 0, num=n)
    y = np.linspace(0, 0, num=n)
    for i, t in np.ndenumerate(theta):
        r = p/(1+e*math.cos(t))
        x[i] = r*math.cos(t)*math.cos(w)+r*math.sin(t)*math.sin(w)
        y[i] =-r*math.cos(t)*math.sin(w)+r*math.sin(t)*math.cos(w)
    plt.plot(x, y, color=c, label=f'M = {M}')

# multi-rev animated solution plotter. this takes in the outputs from mrlambert and animates every single solution for a given case. input m = mlist
def mranimateSolution(r1, r2, v1, mu, tof, m, res):
    size = np.size(m)
    coords = np.zeros((2*size, res))
    scs = []
    fig, ax = plt.subplots()
    plt.scatter(0, 0, marker='x', color=(0, 0, 0))
    ax.set_aspect('equal')

    for i in range(0, size):
        mrplotSolution(r1, r2, v1[i], mu, res, (i/size, 0, 1-i/size), m[i])
        xc, yc = animateSolution(r1, r2, v1[i], mu, tof, res)
        coords[2*i, :] = xc
        coords[2*i+1, :] = yc
        scs.append(ax.scatter(xc, yc, color=(i/size, 0, 1-i/size)))
    ax.legend()

    def update(frame):
        for i in range(0, size):
            x = coords[2*i, frame]
            y = coords[2*i+1, frame]
            data = np.stack([x,y]).T
            scs[i].set_offsets(data)

    anim = an.FuncAnimation(fig=fig, func=update, frames=res, interval=10, repeat=False)
    plt.show()

# plot orbit from state vectors in a 3-d window
def threedimorbitplot(r1, v1, mu, res, c, style):
    if np.linalg.norm(np.cross([0, 0, 1], np.cross(r1, v1)))==0:
        plotOrbit(r1, v1, mu, 500, c, style)
    else:
        E = np.dot(v1, v1)/2-mu/np.linalg.norm(r1)
        a = -mu/(2*E)
        h = np.cross(r1, v1)
        ev = np.cross(v1, h)/mu-r1/np.linalg.norm(r1)
        e = np.linalg.norm(ev)
        p = a*(1-e**2)
        periapsis = p/(1+e)*ev/e
        I = np.array([1, 0, 0])
        J = np.array([0, 1, 0])
        K = np.array([0, 0, 1])
        N = np.cross(K, h)/np.linalg.norm(np.cross(K, h))
        if np.dot(ev, K) > 0:
            w = math.acos(np.dot(N, ev)/e)
        else:
            w = 2*math.pi-math.acos(np.dot(N, ev)/e)
        i = math.acos(np.dot(h, K)/np.linalg.norm(h))
        if np.dot(I, N) >= 0:
            o = math.atan(np.dot(J, N)/np.dot(I, N))
        else:
            o = math.pi + math.atan(np.dot(J, N)/np.dot(I, N))
        DCM = np.transpose(np.matmul(rotz(w), np.matmul(rotx(i), rotz(o))))
        t = np.linspace(0, 2*math.pi+0.01, num=res)
        coords = np.zeros([res, 3])
        for z in range(res):
            r = p/(1+e*math.cos(t[z]))
            perifocal = np.array([r*math.cos(t[z]), r*math.sin(t[z]), 0])
            coords[z, :] = np.transpose(np.matmul(DCM, np.transpose(perifocal)))
        if style == "Dotted":
            plt.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=c, linestyle="dotted")
        elif style == "Dashed":
            plt.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=c, linestyle="dashed")
        else:
            plt.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=c)

# plot Lambert solution trajectory in a 3-d window
def threedimsolutionplot(r1, r2, v1, m, mu, res, c):
    if np.linalg.norm(np.cross([0, 0, 1], np.cross(r1, v1)))==0:
        if m > 0:
            plotOrbit(r1, v1, mu, 500, c, "")
        else:
            plotSolution(r1, r2, v1, mu, res, c)
    else:
        if m > 0:
            threedimorbitplot(r1, v1, mu, res, c, "")
        else:
            E = np.dot(v1, v1)/2-mu/np.linalg.norm(r1)
            a = -mu/(2*E)
            h = np.cross(r1, v1)
            ev = np.cross(v1, h)/mu-r1/np.linalg.norm(r1)
            e = np.linalg.norm(ev)
            p = a*(1-e**2)
            periapsis = p/(1+e)*ev/e
            I = np.array([1, 0, 0])
            J = np.array([0, 1, 0])
            K = np.array([0, 0, 1])
            N = np.cross(K, h)/np.linalg.norm(np.cross(K, h))
            if np.dot(ev, K) > 0:
                w = math.acos(np.dot(N, ev)/e)
            else:
                w = 2*math.pi-math.acos(np.dot(N, ev)/e)
            i = math.acos(np.dot(h, K)/np.linalg.norm(h))
            if np.dot(I, N) >= 0:
                o = math.atan(np.dot(J, N)/np.dot(I, N))
            else:
                o = math.pi + math.atan(np.dot(J, N)/np.dot(I, N))
            DCM = np.transpose(np.matmul(rotz(w), np.matmul(rotx(i), rotz(o))))
            if np.linalg.norm(ev) < 1e-12:
                w = 0
                t1 = 0
                t2 = math.acos(np.dot(r1,r2)/(np.linalg.norm(r1)**2))
            else:
                t1 = math.acos((p-np.linalg.norm(r1))/(np.linalg.norm(r1)*e))
                if np.dot(np.cross(r1,periapsis), np.cross(r1, v1))>0:
                    t1 = 2*math.pi - t1
                t2 = math.acos((p-np.linalg.norm(r2))/(np.linalg.norm(r2)*e))
                if np.dot(np.cross(r2,periapsis), np.cross(r1, v1))>0:
                    t2 = 2*math.pi - t2
                if t1 > t2:
                    t1 = t1 - 2*math.pi
            t = np.linspace(t1, t2, num=res)
            coords = np.zeros([res, 3])
            for z in range(res):
                r = p/(1+e*math.cos(t[z]))
                perifocal = np.array([r*math.cos(t[z]), r*math.sin(t[z]), 0])
                coords[z, :] = np.transpose(np.matmul(DCM, np.transpose(perifocal)))
            plt.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=c)

# returns direction cosine matrix to convert from perifocal coordinates to ECI
def Perifocal2ECI(r1, v1, mu):
    E = np.dot(v1, v1)/2-mu/np.linalg.norm(r1)
    a = -mu/(2*E)
    h = np.cross(r1, v1)
    ev = np.cross(v1, h)/mu-r1/np.linalg.norm(r1)
    e = np.linalg.norm(ev)
    p = a*(1-e**2)
    periapsis = p/(1+e)*ev/e
    I = np.array([1, 0, 0])
    J = np.array([0, 1, 0])
    K = np.array([0, 0, 1])
    N = np.cross(K, h)/np.linalg.norm(np.cross(K, h))
    if np.dot(ev, K) > 0:
        w = math.acos(np.dot(N, ev)/e)
    else:
        w = 2*math.pi-math.acos(np.dot(N, ev)/e)
    i = math.acos(np.dot(h, K)/np.linalg.norm(h))
    if np.dot(I, N) >= 0:
        o = math.atan(np.dot(J, N)/np.dot(I, N))
    else:
        o = math.pi + math.atan(np.dot(J, N)/np.dot(I, N))
    DCM = np.transpose(np.matmul(rotz(w), np.matmul(rotx(i), rotz(o))))
    return DCM

def COE2State(a, e, i, o, w, theta, mu):
    p = a*(1-e**2)
    r = p/(1+e*math.cos(theta))
    perifocal = np.array([r*math.cos(theta), r*math.sin(theta), 0])
    h = math.sqrt(p*mu)
    v = math.sqrt(mu*(2/r-1/a))
    if (h/(r*v)) > 1:
        fpa = math.acos(1)
    else:
        fpa = math.acos(h/(r*v))
    if theta > math.pi:
        fpa = -fpa
    trv = math.pi/2-fpa
    vpf = np.array([v*math.cos(theta+trv), v*math.sin(theta+trv), 0])
    DCM = np.transpose(np.matmul(rotz(w), np.matmul(rotx(i), rotz(o))))
    rv = np.transpose(np.matmul(DCM, np.transpose(perifocal)))
    vv = np.transpose(np.matmul(DCM, np.transpose(vpf)))
    return rv, vv

# solves kepler's time equation to find eccentric anomaly. if t is already formatted as since periapsis, set t_p=0
def KTE(mu, a, e, t, t_p, tol):
    dt = t-t_p
    M = math.sqrt(mu/(a**3))*dt
    E0 = 0
    E1 = M
    while abs(E1-E0) > tol:
        E0 = E1
        E1 = E1-(E1-e*math.sin(E1)-M)/(1-e*math.cos(E1))
    return E1

# inverts kepler's time equation to find time elapsed since periapsis for a given eccentric anomaly. required for animated transfer trajectories
def invKTE(mu, a, e, EA):
    t = (EA - e*math.sin(EA))/math.sqrt(mu/a**3)
    return t

# converts from eccentric anomaly to true anomaly
def EAtoTA(E, e):
    T = 2*math.atan(math.sqrt((1+e)/(1-e))*math.tan(E/2))
    return T

# converts from true anomaly to eccentric anomaly
def TAtoEA(T, e):
    E = 2*math.atan(math.sqrt((1-e)/(1+e))*math.tan(T/2))
    return E

# rotation matrix about x axis
def rotx(t):
    return np.array([[1, 0, 0], [0, math.cos(t), math.sin(t)], [0, -math.sin(t), math.cos(t)]])

# rotation matrix about y axis
def roty(t):
    return np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])

# rotation matrix about z axis
def rotz(t):
    return np.array([[math.cos(t), math.sin(t), 0], [-math.sin(t), math.cos(t), 0], [0, 0, 1]])

