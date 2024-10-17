

import numpy as np

from scipy.special import kv, jv, sinc
import scipy.constants as cte


#todo: passar para dentro das funcoes, isso nao e' uma boa pratica
# fundamental constants
_hbar = cte.hbar # reduced planck constant [J.s]
_c = cte.c # speed of light [m/s]
_e = cte.e # fundamental charge [C]
_me = cte.electron_mass # electron rest mass [kg]
_E0 = _me*(_c**2)/_e # electron rest energy [eV]
_eps0 = cte.epsilon_0 # electric vacuum permittivity [F/m]

# accelerator constants
_I = 0.1 # current [A]
_E = 3e9 # energy [eV]
_gamma = _E/_E0 # lorentz factor [adim]
_beta = 1-(1/(2*(_gamma**2))) # velocity parameter [adim]




# ------------------------ bending magnet radiation ------------------------ #


#todo: unificar as duas funcoes passando argumento de escolher adimensional or dimensional em angulo ou area

def bm_d2IdWdw(w_wc,gtheta):
    G = 0.5*w_wc*((1+gtheta**2)**(3/2))
    Q = (3*(_e**2)*(_gamma**2))/(16*(np.pi**3)*_eps0*_c)
    return Q * (w_wc**2)*((1+gtheta**2)**2) * (kv(2/3,G)**2 + (gtheta**2/(1+gtheta**2))*kv(1/3,G)**2)

# exact vertical bending magnet spectrum
#todo: polarizacao pi e sigma
def bm(B,energy,gtheta,dep='ang',d=None):
    """
    Calculates bending magnet spectrum.

    Args:
        B [T], energy [eV], gtheta.
        dep: dependence type of the spectrum. It can be 'adim', which gives __ or
        'ang', ph/s/mrad2/0.1%bw, or 'area', ph/s/mm2/0.1%bw

    Returns:
        Flux Density [__] or [ph/s/mrad2/0.1%bw] or [ph/s/mm2/0.1%bw].
    """

    if isinstance(energy,list):
        energy = np.array(energy)
  
    # magnetic radius
    rho = _E*_beta/(B*_c)
    # critical frequency
    wc = (3*_c*_gamma**3)/(2*rho)

    w = energy*_e/_hbar
    w_wc = w/wc

    if dep=='adim':
        spec = bm_d2IdWdw(w_wc,gtheta)
    elif dep=='ang':
        spec = 1e-9*(_I/(_hbar*_e))*bm_d2IdWdw(w_wc,gtheta)
    elif dep=='area':
        if d != None:
            spec = 1e-9*(_I/(_hbar*_e*(d**2)))*bm_d2IdWdw(w_wc,gtheta)
        else:
            raise ValueError("Dependence (dep) not identified!")
        
    return spec
    

# ----------------------- planar undulator radiation ----------------------- #

# functions #

# grating function
def L(wbar,gt,K,N):
    x = np.pi*wbar*(1+((gt**2)/(1+(K**2)/2)))
    grating = (sinc(N*x/np.pi)/sinc(x/np.pi))**2
    isHarmonic = np.isclose(np.sin(x),0)
    return np.where(isHarmonic,1.0,grating)


def und_aux(wbar,gtx,gty,m,n,K,lambda_u):

    # undulator parameters
    beta_med = _beta*(1-((K/(2*_gamma))**2)) # mean velocity parameter in the undulator [adim]
    w0 = 2*np.pi*beta_med*_c/lambda_u # magnetic frequency [rad/s]
    w1 = 4*np.pi*_c*(_gamma**2)/(lambda_u*(1+((K**2)/2))) # first harmonic frequency [rad/s]

    # calc parameters
    q_u = (K*w1)/((_gamma**2)*w0)
    q_v = ((K**2)*w1)/(8*(_gamma**2)*w0)

    gt2 = gtx**2 + gty**2 # gt=gamma*theta; gtx=gt*cos(phi); gty=gt*sin(phi)

    R_w = wbar*(1+(gt2/(1+(K**2)/2)))-m+2*n
    temp = jv(n,q_v*wbar)*jv(m,q_u*gtx*wbar)*(2*gtx*sinc(R_w)-K*(sinc(R_w+1)+sinc(R_w-1)))

    return temp

# undulator spectrum (still to improve, k=2,6)
#todo: even harmonics k=4,6,10,... not symmetric, but should be
def und(energy,gtx,gty,K,lambda_u,N,I_b):

    # undulator parameters
    beta_med = _beta*(1-((K/(2*_gamma))**2)) # mean velocity parameter in the undulator [adim]
    w0 = 2*np.pi*beta_med*_c/lambda_u # magnetic frequency [rad/s]
    w1 = 4*np.pi*_c*(_gamma**2)/(lambda_u*(1+((K**2)/2))) # first harmonic frequency [rad/s]

    # calc parameters
    q_u = (K*w1)/((_gamma**2)*w0)
    q_v = ((K**2)*w1)/(8*(_gamma**2)*w0)
    Q = (((_e*w1)/(4*_gamma*w0))**2)/(np.pi*_eps0*_c)

    w = energy*_e/_hbar
    wbar = w/w1
    gt2 = gtx**2 + gty**2 # gt=gamma*theta; gtx=gt*cos(phi); gty=gt*sin(phi)
    gt = np.sqrt(gt2)

    # j, sc, r, temp = [], [], [], []
    # sc1, sc2 = [], []

    h, v = 0.0, 0.0
    for m in range(-10,10+1):
        for n in range(-20,20+1):
            R_w = wbar*(1+(gt2/(1+(K**2)/2)))-m+2*n
            htemp = jv(n,q_v*wbar)*jv(m,q_u*gtx*wbar)*(2*gtx*sinc(R_w)-K*(sinc(R_w+1)+sinc(R_w-1)))
            vtemp = jv(n,q_v*wbar)*jv(m,q_u*gtx*wbar)*2*gty*sinc(R_w)
            
            # r.append(np.round(R_w,4))
            # j.append(np.round(jv(m,q_u*gtx*wbar),4))
            # sc.append(np.round(2*gtx*sinc(R_w)-K*(sinc(R_w+1)+sinc(R_w-1)),4))
            # temp.append(np.round(htemp,4))
            # sc1.append(np.round(2*gtx*sinc(R_w),4))
            # sc2.append(np.round(K*(sinc(R_w+1)+sinc(R_w-1)),4))

            h += htemp
            v += vtemp

    # np.savetxt('j.dat',j) # j nao e' a origem da falta de simeteria
    # np.savetxt('sc.dat',sc) # nao simetrico
    # np.savetxt('sc1.dat',sc1) # antissimetrico
    # np.savetxt('sc2.dat',sc2) # simetrico
    # np.savetxt('r.dat',r) # r e' simetrico tambem, entao descartamos
    # np.savetxt('temp.dat',temp) # como j nao e', da' pra descartar esse tambem
    

    return 1e-9*(I_b/(_e*_hbar)) * (N**2)*Q*(wbar**2)*(h**2 + v**2)*L(wbar,gt,K,N)


# undulator spectrum. valid only for harmonics (i.e. w/w1 = k, k non-negative
# integer. not to be confused with K, deflection parameter)
def und_k(k,gtx,gty,K,lambda_u,N,I_b):

    # undulator parameters
    beta_med = _beta*(1-((K/(2*_gamma))**2)) # mean velocity parameter in the undulator [adim]
    w0 = 2*np.pi*beta_med*_c/lambda_u # magnetic frequency [rad/s]
    w1 = 4*np.pi*_c*(_gamma**2)/(lambda_u*(1+((K**2)/2))) # first harmonic frequency [rad/s]

    # calc parameters
    #todo: redefinir q_u e q_v para cortar lambda_u, se nao me engano
    q_u = (K*w1)/((_gamma**2)*w0)
    q_v = ((K**2)*w1)/(8*(_gamma**2)*w0)
    Q = (((_e*w1)/(4*_gamma*w0))**2)/(np.pi*_eps0*_c)

    gt2 = gtx**2 + gty**2 # gt=gamma*theta; gtx=gt*cos(phi); gty=gt*sin(phi)
    wbar = k/(1+gt2/(1+(K**2)/2)) # w/(w1(theta)); adimensional frequency of harmonic k

    h, v = 0.0, 0.0
    for n in range(-10,10+1):
        R_w = k+2*n
        h += jv(n,q_v*wbar)*(2*gtx*jv(R_w,q_u*gtx*wbar)-K*(jv(R_w+1,q_u*gtx*wbar)+jv(R_w-1,q_u*gtx*wbar)))
        v += jv(n,q_v*wbar)*2*gty*jv(R_w,q_u*gtx*wbar)

    return 1e-9*(I_b/(_e*_hbar)) * (N**2)*Q*(wbar**2)*(h**2 + v**2)


def F_0(wbar,K,lambda_u):

    # undulator parameters
    beta_med = _beta*(1-((K/(2*_gamma))**2)) # mean velocity parameter in the undulator [adim]
    w0 = 2*np.pi*beta_med*_c/lambda_u # magnetic frequency [rad/s]
    w1 = 4*np.pi*_c*(_gamma**2)/(lambda_u*(1+((K**2)/2))) # first harmonic frequency [rad/s]

    # calc parameters
    q_v = ((K**2)*w1)/(8*(_gamma**2)*w0)
    Q = (((_e*w1)/(4*_gamma*w0))**2)/(np.pi*_eps0*_c)

    t = 0.0
    for n in range(-50,50+1):
        R_w = wbar+2*n
        t += jv(n,q_v*wbar)*(sinc(R_w+1)+sinc(R_w-1))

    return Q*(wbar**2)*(t**2)

# undulator spectrum. valid only on-axis (i.e. gtx=gty=0)
def und_0(energy,deflec_param,period_length,nr_periods,I_b):
    K, lambda_u, N = deflec_param, period_length, nr_periods

    w1 = 4*np.pi*_c*(_gamma**2)/(lambda_u*(1+((K**2)/2)))
    w = energy*_e/_hbar
    wbar = w/w1

    spec = 1e-9*(I_b/(_e*_hbar)) * (N**2)*(K**2)*F_0(wbar,K,lambda_u)*L(wbar,0,K,N)

    return spec


def F_k_0(k,K,lambda_u):

    # undulator parameters
    beta_med = _beta*(1-((K/(2*_gamma))**2)) # mean velocity parameter in the undulator [adim]
    w0 = 2*np.pi*beta_med*_c/lambda_u # magnetic frequency [rad/s]
    w1 = 4*np.pi*_c*(_gamma**2)/(lambda_u*(1+((K**2)/2)))
    
    # calc parameters
    q_v = ((K**2)*w1)/(8*(_gamma**2)*w0)
    Q = (((_e*w1)/(4*_gamma*w0))**2)/(np.pi*_eps0*_c)

    t = jv((k+1)/2,q_v*k) - jv((k-1)/2,q_v*k)

    return Q*(k**2)*(t**2)

# undulator spectrum. valid only for harmonics (i.e. w/w1=k, k non-negative
# integer) and on-axis (i.e. gtx=gty=0)
def und_k_0(k,deflec_param,period_length,nr_periods,I_b):
    K, lambda_u, N = deflec_param, period_length, nr_periods

    return 1e-9*(I_b/(_e*_hbar)) * (N**2)*(K**2)* F_k_0(k,K,lambda_u)


