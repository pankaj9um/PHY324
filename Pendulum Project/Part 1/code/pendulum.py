#!/usr/bin/env python3# -*- coding: utf-8 -*-"""@author: Pankaj Patil"""import mathimport statslab as utilsimport matplotlib.pyplot as pltimport numpy as npimport globdata_folder_path = "../data/data2"# estimated period 10 seconds# uncertainty in mass = 1 grammass_error = 0.001# uncertainty in length = 1 cmlength_error = 0.05# uncertainty in angle = 1 degangle_error = math.radians(1)# array to hold the estimated valuesestimates = []def model_function_only_decay(time, initial_angle,                               period, decay_constant, init_phase):    return initial_angle * np.power(np.e, - time / decay_constant)def model_function(time, initial_angle, period, decay_constant, init_phase):    return initial_angle * np.power(np.e, - time / decay_constant) \        * np.cos(2 * math.pi * time / period + init_phase)def get_constants(filename):    file = open(filename, 'r')    # read weight, convert to SI    line = file.readline()    mass = float(line.split(",")[1]) /1000.0        # read length    line = file.readline()    length = float(line.split(",")[1])    line = file.readline()    angle = float(line.split(",")[1])    file.close()    return mass, length, angledef analyze_file(filename):    mass, length, iangle = get_constants(filename)            # read the data    measured_times, measured_angles_degrees = utils.read_data(filename,                                                usecols=(0,1),                                                skiprows=4)        sample_measured_times = measured_times        # convert the angles to radians    measured_angles_radians = np.radians(measured_angles_degrees)        angle_errors = np.ones_like(measured_angles_radians) * angle_error        # lets do fitting only first 1000 sample point    fit_samples = 1000    popt, pstd = utils.fit_data(model_function,                           measured_times[:fit_samples],                           measured_angles_radians[:fit_samples],                           angle_errors[:fit_samples])    init_angle_radians = popt[0]    period = popt[1]    decay_constant = popt[2]     init_phase_radians = popt[3]    init_angle_degrees = np.degrees(init_phase_radians)        if "decay" in filename:        decay_constant = 95    init_angle_error_radians = pstd[0]    period_error = pstd[1]    decay_constant_error = pstd[2]     init_phase_error_radians = pstd[3]                       print("Analysing file :=", filename)    print("  Mass = %.2f \u00b1 0.001 kg" % mass)    print("  Length = %.2f \u00b1 0.01 m" % length)    print("  Estimated Initial Angle = (%.2f \u00b1 %.3f) s" % \          (init_angle_radians, init_angle_error_radians))    print("  Estimated Period = (%.2f \u00b1 %.5f) s" % \          (period, period_error))    print("  Estimated Decay Constant = (%.2f \u00b1 %.3f) s" % \          (decay_constant, decay_constant_error))    print("  Estimated Initial Phase = (%.2f \u00b1 %.3f) rad" % \          (init_phase_radians, init_phase_error_radians))    fig = plt.figure(figsize=(16, 8))    (a0, a1) = fig.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [3, 1]})            # plot the error bar chart    a0.errorbar(measured_times, measured_angles_radians,                 yerr= angle_errors,                 capsize=2,                 ls="",                 marker="o",                  markersize=1,                 label=r"Measured $\theta$ (rad)")    # sort the measured angles    reverse_measured_angles_radians = np.flip(measured_angles_radians)    current_peak = np.abs(reverse_measured_angles_radians[0])    reverse_measured_decay_angles = [current_peak]    for a in reverse_measured_angles_radians[1:]:        if a > current_peak:            current_peak = a                reverse_measured_decay_angles.append(current_peak)                # compute the decay curve          predicted_decay_angles = model_function_only_decay(measured_times,                       init_angle_radians,                       period,                      decay_constant,                       init_phase_radians)        measured_decay_angles = np.flip(reverse_measured_decay_angles)        residue = None            # calculate chi2r using the fit_samples    chi2r_curve_fit = utils.chi2reduced(measured_angles_radians[:fit_samples],                       model_function(measured_times[:fit_samples],                                      init_angle_radians,                                     period,                                       decay_constant,                                     init_phase_radians),                      angle_errors[:fit_samples],                       4)    # plot the decay chart    if "decay" in filename:        chi2r_curve_fit_decay = utils.chi2reduced(measured_decay_angles[:fit_samples],                   model_function_only_decay(measured_times[:fit_samples],                                  init_angle_radians,                                 period,                                   decay_constant,                                 init_phase_radians),                  angle_errors[:fit_samples],                   4)        a0.plot(measured_times, predicted_decay_angles,               label=r"$\theta(t) = \theta_0 e^{-t/\tau},\ \chi^2_{red}=%.2f$"\                  % chi2r_curve_fit_decay,              color="red")        residue = (measured_decay_angles- predicted_decay_angles)    if "decay" not in filename:        predicted_angles = model_function(measured_times,                                   init_angle_radians,                                   period,                                  decay_constant,                                   init_phase_radians)            residue = (measured_angles_radians- predicted_angles)        label = r"$\theta(t) = \theta_0 e^{-t/\tau}\cos(\frac{t}{T}),\ \chi^2_{red}=%.2f$" % chi2r_curve_fit                    a0.plot(measured_times, predicted_angles,               label=label)            print("  chi2r_curve_fit = %.2f" % chi2r_curve_fit)    a1.set_xlabel("Time (s)")    a0.set_ylabel(r"$\theta$ (rad)")    a0.set_title("Angle vs Time \(M = %.f \u00b1 1 grams, L = %.1f \\u00b1 0.5 cm, %s = %.2f rad = %.f deg)" % \            (mass*1000, length*100, r"$\theta_0$", math.radians(iangle),              iangle))    a0.legend(loc=1)        if "decay" not in filename:        ylim = np.max(measured_angles_radians[:100])        a0.set_ylim([-ylim * 1.2, ylim * 2])            # a0.set_ylim(-0.8,1.1)    # store the estimates    estimates.append([init_angle_radians, period, period_error,                       decay_constant, decay_constant_error])            # new residue graph    a1.set_ylabel(r"Residuals $\Delta\theta$")    a1.plot(measured_times, residue,           label="exponential decay envelop")        plt.savefig("%s.png" % filename[:-4], bbox_inches='tight')    plt.style.use("seaborn-whitegrid")files = [    "../data/30.csv",    "../data/47.csv",    "../data/58.csv",    "../data/72.csv",    "../data/85.csv",    "../data/decay.csv"    ]for file in files:    analyze_file(file)    