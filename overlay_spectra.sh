#!/bin/bash
cd "/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/RES_BLADE_9_25/rotor_airframe_data/10_4/"
overlay_spectra.py -c s_t1450_bl_d1_m70_r1 motor_t1220_m90_r1 s_t1450_iso_d1_m90_r1 s_t1450_bl_d1_m90_r2 -l Baseline SDOF MDOF SPLIT --tonal_separation -df 1 -ovr 0.5 -win hann -m 0 4 8 --align --filter_harmonics 1 100 --filter_shaft_order
