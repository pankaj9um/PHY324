#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import math
import statslab as utils
import matplotlib.pyplot as plt
import numpy as np
import glob

data_folder_path = "../data"

# estimated period 10 seconds

# uncertainty in mass = 1 gram
mass_error = 0.001

# uncertainty in length = 1 cm
length_error = 0.05

# uncertainty in angle = 1 deg
angle_error = math.radians(1)

# array to hold the estimated values
estimates = []

def model_function_only_decay(time, initial_angle, 
                              period, decay_constant, init_phase):
    return initial_angle * np.power(np.e, - time / decay_constant)

def model_function(time, initial_angle, period, decay_constant, init_phase):
    return initial_angle * np.power(np.e, - time / decay_constant) \
        * np.cos(2 * math.pi * time / period + init_phase)

def get_constants(filename):
    file = open(filename, 'r')
    # read weight, convert to SI
    line = file.readline()
    mass = float(line.split(",")[1]) /1000.0
    
    # read length
    line = file.readline()
    length = float(line.split(",")[1])

    file.close()

    return mass, length

def analyze_file(filename):
    mass, length = get_constants(filename)
        
    # read the data
    measured_times, measured_angles_degrees = utils.read_data(filename,
                                                usecols=(0,1),
                                                skiprows=3)
    
    measured_times = measured_times - measured_times[0]
    sample_measured_times = measured_times
    
    # convert the angles to radians
    measured_angles_radians = np.radians(measured_angles_degrees)
    
    angle_errors = np.ones_like(measured_angles_radians) * angle_error
    
    # lets do fitting only first 1000 sample point
    fit_samples = 1000
    popt, pstd = utils.fit_data(model_function, 
                          measured_times[:fit_samples], 
                          measured_angles_radians[:fit_samples], 
                          angle_errors[:fit_samples])

    init_angle_radians = popt[0]
    period = popt[1]
    decay_constant = popt[2] 
    init_phase_radians = popt[3]
    init_angle_degrees = np.degrees(init_phase_radians)

    init_angle_error_radians = pstd[0]
    period_error = pstd[1]
    decay_constant_error = pstd[2] 
    init_phase_error_radians = pstd[3]
                   
    print("Analysing file :=", filename)
    print("  Mass = %.2f \u00b1 0.001 kg" % mass)
    print("  Length = %.2f \u00b1 0.01 m" % length)

    print("  Estimated Initial Angle = (%.2f \u00b1 %.3f) s" % \
          (init_angle_radians, init_angle_error_radians))
    print("  Estimated Period = (%.2f \u00b1 %.5f) s" % \
          (period, period_error))
    print("  Estimated Decay Constant = (%.2f \u00b1 %.3f) s" % \
          (decay_constant, decay_constant_error))
    print("  Estimated Initial Phase = (%.2f \u00b1 %.3f) rad" % \
          (init_phase_radians, init_phase_error_radians))

    plt.figure(figsize=(16, 8))
    
    # plot the error bar chart
    plt.errorbar(measured_times, measured_angles_radians,
                 yerr= angle_errors,
                 capsize=2,
                 ls="",
                 marker="o", 
                 markersize=1,
                 label="measured angle (rad)")

    # compute the decay curve      
    predicted_decay_angles = model_function_only_decay(measured_times, 
                      init_angle_radians, 
                      period,
                      decay_constant, 
                      init_phase_radians)
    
    # plot the decay chart
    plt.plot(measured_times, predicted_decay_angles, 
             label="exponential decay envelop",
             color="red")
    plt.plot(measured_times, -predicted_decay_angles, 
             color="red")

    if "decay" not in filename:
        predicted_angles = model_function(measured_times, 
                                  init_angle_radians, 
                                  period,
                                  decay_constant, 
                                  init_phase_radians)
    
        
        plt.plot(measured_times, predicted_angles, 
                 label="curve fit data points")
        
    
    # calculate chi2r using the fit_samples
    chi2r_curve_fit = utils.chi2reduced(measured_angles_radians[:fit_samples], 
                      model_function(measured_times[:fit_samples], 
                                     init_angle_radians,
                                     period,  
                                     decay_constant,
                                     init_phase_radians),
                      angle_errors[:fit_samples], 
                      4)

    print("  chi2r_curve_fit = %.2f" % chi2r_curve_fit)

    plt.xlabel("time (s)")
    plt.ylabel("$\Theta$ (rad)")
    plt.title("Angle vs Time (Mass = %.f \u00b1 1 grams, Length = %.1f \
\u00b1 0.5 cm)" % \
            (mass * 1000, length*100))
    plt.legend(loc=1)
    
    if "decay" not in filename:
        ylim = np.max(measured_angles_radians[:100])
        plt.ylim([-ylim * 1.5, ylim * 2.0])
    
    plt.savefig("%s.png" % filename[:-4])

    # store the estimates
    estimates.append([init_angle_radians, period, period_error, 
                      decay_constant, decay_constant_error])
    
plt.style.use("seaborn-whitegrid")

files = [
    "../data/26-10.csv",
    "../data/26-20.csv",
    "../data/26-30.csv",
    "../data/26-40.csv",
    "../data/26-50.csv",
    "../data/60-30.csv",
    "../data/95-30.csv",
    "../data/165-30.csv",
    ]

for file in files:
    analyze_file(file)
    