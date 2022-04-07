#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt

# use bigger font size for plots
plt.rcParams.update({'font.size': 16})
plt.style.use("seaborn-whitegrid")

def chi2(y_measure,y_predict,errors):
    """Calculate the chi squared value given a measurement with errors and 
    prediction"""
    return np.sum( np.power(y_measure - y_predict, 2) / np.power(errors, 2) )

def chi2reduced(y_measure, y_predict, errors, number_of_parameters):
    """Calculate the reduced chi squared value given a measurement with errors
    and prediction, and knowing the number of parameters in the model."""
    return chi2(y_measure, y_predict, errors)/ \
            (y_measure.size - number_of_parameters)

def read_data(filename, skiprows=1, usecols=(0,1), delimiter=","):
    """Load give\n file as csv with given parameters, 
    returns the unpacked values"""
    return np.loadtxt(filename,
                      skiprows=skiprows, usecols=usecols, 
                      delimiter=delimiter,
                      unpack=True)

def fit_data(model_func, xdata, ydata, yerrors, guess=None):
    """Utility function to call curve_fit given x and y data with errors"""
    popt, pcov = optim.curve_fit(model_func, 
                                xdata, ydata, absolute_sigma=True, 
                                sigma=yerrors,
                                p0=guess)

    pstd = np.sqrt(np.diag(pcov))
    return popt, pstd

chromium_files = [
        # "../data/cr-1.csv",
        # "../data/cr-2.csv",
        "../data/cr-4.csv",
        "../data/cr-5.csv",
        "../data/cr-3.csv",
    ]

silver_files = [
        # "../data/ag-1.csv",
        "../data/ag-2.csv",
        "../data/ag-3.csv",
        "../data/ag-4.csv",
        # "../data/ag-5.csv",
    ]

constants = {
        "cr": {
                "R_v1v2": 60.38,
                "dR": 0.01,
                "width": 3.10*10**-3,
                "length": 16.30*10**-3,
                "fringestep": 0.21*10**-3,
                "fringeseparation": 7.0*10**-3,
                "rh_theory": 3.80*10**-10, 
                "dw": 0.01*10**-3,
                "dl": 0.01*10**-3,
                "df": 0.1*10**-3,
                "da": 0.0,
                "RH": [],
                "dRH": [],
                "n": [],
                "dn": [],
                "mu": [],
                "dmu": [],
                "vd": [],
                "dvd": [],
            },
        "ag": {
                "R_v1v2": 2.06,
                "dR": 0.01,
                "width": 2.8*10**-3,
                "length": 21.10*10**-3,
                "fringestep": 1.5*10**-3,
                "fringeseparation": 11.12*10**-3,
                "rh_theory": 1.07*10**-10, 
                "dw": 0.1*10**-3,
                "dl": 0.1*10**-3,
                "df": 1*10**-3,
                "da": 0.0,
                "RH": [],
                "dRH": [],
                "n": [],
                "dn": [],
                "mu": [],
                "dmu": [],
                "vd": [],
                "dvd": [],
            }
    }

e = 1.602*10**-19
V_err = 0.001 / 1000
I_err = 0.01 / 1000
distance_err =  0.01 * 10 ** -3
dB = 0.001

def model_function(J_x, RB):
    return RB * J_x

def sqr(x):
    return x*x

def plot_ey_vs_j(name, filename, csnts):
    print("analyzing file", filename)
    file = open(filename, 'r')

    line = file.readline()
    B = float(line.split(",")[1])
    
    file.close()
    
    measured_I, measured_V = read_data(filename,
                                       usecols=(0,1),
                                       skiprows=2)
    
    global V_err
    scale = 5
    if "Silver" in name:
        V_err = 0.001 / 1000
        measured_V = -measured_V / scale
    else:
        measured_V = measured_V * scale
        V_err = 0.001 / 1000

    measured_I = measured_I * 10**-3
    measured_V = measured_V * 10**-3
    
    # print("V={}, I={}".format(measured_V[0], measured_I[0]))
    
    width = csnts["width"]
    thickness = csnts["thickness"]
    resistivity = csnts["resistivity"]
    area = width * thickness
        
    E_y = measured_V / width
    J_x = measured_I / area
    
    # print("E_y={}, J_x={}".format(E_y[0], J_x[0]))

    dE = E_y * np.sqrt( (V_err / measured_V) **  2 +
                               (distance_err / width) ** 2)
    
    dJ = J_x * np.sqrt( (I_err / measured_I) **  2 +
                               (csnts["da"] / area) ** 2)
        
    popt, pstd = fit_data(model_function, 
                      J_x, 
                      E_y, 
                      dE, guess=(csnts["rh_theory"] * B))
    
    # print("Slope={}".format(popt[0]))

    RH = popt[0]
    
    # print("dB={:e}, B={:e}, dE={:e}, E={:e}, dJ={:e}, J={:e}".format(
    #                 np.mean(dB), np.mean(B), 
    #                 np.mean(dE), np.mean(E_y), np.mean(dJ), 
    #                 np.mean(J_x)))
        
    dRH = RH * np.sqrt( (dB / B) **  2 + (dE / E_y) ** 2 + (dJ / J_x) ** 2)
                               
    csnts["RH"].append(RH)
    csnts["dRH"].append(np.mean(dRH))
    
    print("  B = {:.3f} +/- {:e}".format(B, dB))
    print("  Hall Constant = {:e} +/- {:e}".format(RH, np.mean(dRH)))
    plt.errorbar(J_x, E_y,
             yerr=dE,
             capsize=2,
             ls="",
             marker="o", 
             markersize=3)

    xdata = np.linspace(J_x[0], J_x[-1], 1000)
    predicted_E_y = model_function(xdata, popt[0])

    chi2r_curve_fit = chi2reduced(E_y, model_function(J_x, popt[0]), 
                                  dE, 1)
    
    # plot the decay chart
    plt.plot(xdata, predicted_E_y, 
             label="B=%.2f T, $R_H$=%.e $m^3/C$, $\chi^2_{red}$=%.2f" % \
                 (B, RH,
                  chi2r_curve_fit))
       
    # plt.ticklabel_format(style='plain')
    
    print("  chi2r_curve_fit = %.2f" % chi2r_curve_fit)
    v_d = E_y/B
    v_d_err = np.std(v_d)
    v_d_mean = np.mean(v_d)
    

    print("  drift velocity = %.2f +/- %.2f" % (v_d_mean, v_d_err))
    n = 1/e/RH
    dn = 1/e/sqr(RH)*dRH
    print("  density of charge carriers = {:e} +/- {:e}".format(n, 
                                                                np.mean(dn)))
    mu = RH / resistivity
    # print(constants)
    dmu = mu * np.sqrt(sqr(csnts["dcond"]/
                                              csnts["conductivity"]) 
                                          + sqr(dRH/RH))
    
    csnts["vd"].append(v_d_mean)
    csnts["n"].append(np.mean(n))
    csnts["mu"].append(np.mean(mu))

    csnts["dvd"].append(np.mean(v_d_err))
    csnts["dn"].append(np.mean(dn))
    csnts["dmu"].append(np.mean(dmu))

    print("  electric mobility = {:e} +/- {:e}".format(mu, np.mean(dmu)))
    
    plt.legend()

def analyze_metal(name, files, csnts):
    plt.figure(figsize=(16, 7))
    plt.title("Hall Field vs Current Density for %s Probe" % name)
    if "Silver" in name:
        plt.xlabel("Current Density ($10^8 A/m^2$)")
    else:
        plt.xlabel("Current Density ($10^9 A/m^2$)")
        
    plt.ylabel("Hall Field (V/m)")
    
    for f in files:
        plot_ey_vs_j(name, f, csnts)
        
    plt.savefig("{}.png".format(name), bbox_inches='tight')
        
def compute_thickness(fringestep, fringeseparation):
    return fringestep / fringeseparation * 2945 * 10**-10

constants["cr"]["thickness"] = \
    compute_thickness(constants["cr"]["fringestep"],
                      constants["cr"]["fringeseparation"])
    
constants["ag"]["thickness"] = \
    compute_thickness(constants["ag"]["fringestep"],
                      constants["ag"]["fringeseparation"])
        
area_ag = constants["ag"]["width"] * constants["ag"]["thickness"]
area_cr = constants["cr"]["width"] * constants["cr"]["thickness"]

constants["cr"]["dt"] = constants["cr"]["thickness"] * \
    np.sqrt(sqr(constants["cr"]["df"]/constants["cr"]["fringestep"]) 
            + sqr(constants["cr"]["df"]/constants["cr"]["fringeseparation"]))

constants["ag"]["dt"] = constants["ag"]["thickness"] * \
    np.sqrt(sqr(constants["ag"]["df"]/constants["ag"]["fringestep"]) 
            + sqr(constants["ag"]["df"]/constants["ag"]["fringeseparation"]))
    
constants["cr"]["da"] = area_cr * np.sqrt(sqr(constants["cr"]["dw"]/
                                              constants["cr"]["width"]) 
                                          + sqr(constants["cr"]["dt"]/
                                            constants["cr"]["thickness"]))

constants["ag"]["da"] = area_ag * np.sqrt(sqr(constants["ag"]["dw"]/
                                              constants["ag"]["width"]) 
                                          + sqr(constants["ag"]["dt"]/
                                            constants["ag"]["thickness"]))

def compute_resistivity(R, w, l, t):
    area = w * t
    return R * area / l
        
constants["cr"]["resistivity"] = \
    compute_resistivity(constants["cr"]["R_v1v2"],
                        constants["cr"]["width"],
                        constants["cr"]["length"],
                        constants["cr"]["thickness"])

constants["ag"]["resistivity"] = \
    compute_resistivity(constants["ag"]["R_v1v2"],
                        constants["ag"]["width"],
                        constants["ag"]["length"],
                        constants["ag"]["thickness"])
    
constants["cr"]["conductivity"] = 1/ constants["cr"]["resistivity"]
constants["ag"]["conductivity"] = 1/ constants["ag"]["resistivity"]

dRes_cr = constants["cr"]["resistivity"] * np.sqrt(
    sqr(constants["cr"]["dR"]/ constants["cr"]["R_v1v2"]) +
    sqr(constants["cr"]["da"]/ area_cr) +    
    sqr(constants["cr"]["dl"]/ constants["cr"]["length"])
    )

dRes_ag = constants["ag"]["resistivity"] * np.sqrt(
    sqr(constants["ag"]["dR"]/ constants["cr"]["R_v1v2"]) +
    sqr(constants["ag"]["da"]/ area_ag) +    
    sqr(constants["ag"]["dl"]/ constants["cr"]["length"])
    )

constants["cr"]["dcond"] = dRes_cr / sqr(constants["cr"]["resistivity"])
    
constants["ag"]["dcond"] = dRes_ag / sqr(constants["ag"]["resistivity"])

analyze_metal("Chromium (Cr)", chromium_files, constants["cr"])
analyze_metal("Silver (Ag)", silver_files, constants["ag"])

print("Conductivity Cr = {:e} +/- {:e}".format(
    constants["cr"]["conductivity"], constants["cr"]["dcond"]))
print("Conductivity Ag = {:e} +/-{:e}".format(
    constants["ag"]["conductivity"], constants["ag"]["dcond"]))

print("Resistivity Cr = {:e} +/- {:e}".format(
    constants["cr"]["resistivity"], dRes_cr))
print("Resistivity Ag = {:e} +/-{:e}".format(
    constants["ag"]["resistivity"], dRes_ag))

RH = np.mean(constants["cr"]["RH"])
dRH = np.mean(constants["cr"]["dRH"])
print("Hall Constant Cr = {:e} +/- {:e}".format(
    RH, dRH))
RH = np.mean(constants["ag"]["RH"])
dRH = np.mean(constants["ag"]["dRH"])
print("Hall Constant Ag = {:e} +/- {:e}".format(
    RH, dRH))


n = np.mean(constants["cr"]["n"])
dn = np.mean(constants["cr"]["dn"])
print("n Cr = {:e} +/- {:e}".format(
    n, dn))
n = np.mean(constants["ag"]["n"])
dn = np.mean(constants["ag"]["dn"])
print("n Ag = {:e} +/- {:e}".format(
    n, dn))


mu = np.mean(constants["cr"]["mu"])
dmu = np.mean(constants["cr"]["dmu"])
print("mu Cr = {:e} +/- {:e}".format(
    mu, dmu))
mu = np.mean(constants["ag"]["mu"])
dmu = np.mean(constants["ag"]["dmu"])
print("mu Ag = {:e} +/- {:e}".format(
    mu, dmu))


vd = np.mean(constants["cr"]["vd"])
dvd = np.mean(constants["cr"]["dvd"])
print("vd Cr = {:e} +/- {:e}".format(
    vd, dvd))
vd = np.mean(constants["ag"]["vd"])
dvd = np.mean(constants["ag"]["dvd"])
print("vd Ag = {:e} +/- {:e}".format(
    vd, dvd))
