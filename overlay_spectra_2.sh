#!/bin/bash
cd "/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/RES_BLADE_9_25/rotor_airframe_data/"
overlay_spectra.py -c 10_6/kde_t0_bl_d05_m90_r1 10_6/motor_t1550_m90_r1 10_6/kde_t1775_iso_m90_r2 10_5/kde_t1775_bl_dn1_m90_r2 -l "Background Noise" "Motor Only" "Isolated Rotor" "With Airframe" -df 1 -ovr 0.5 -win hann -m 0 4 8
