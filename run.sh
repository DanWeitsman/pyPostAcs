#!/bin/bash
cd "/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/RES_BLADE_9_25/rotor_airframe_data/10_4/"

cd s_t1450_bl_d1_m60_r1
pyPostAcs.py --tonal_separation -df 1 -ovr 0.5 -win hann -m 0 4 8 --align --filter_harmonics 1 100 --filter_shaft_order
cd ../s_t1450_bl_d1_m70_r1
pyPostAcs.py --tonal_separation -df 1 -ovr 0.5 -win hann -m 0 4 8 --align --filter_harmonics 1 100 --filter_shaft_order
cd ../s_t1450_bl_d1_m80_r1
pyPostAcs.py --tonal_separation -df 1 -ovr 0.5 -win hann -m 0 4 8 --align --filter_harmonics 1 100 --filter_shaft_order
cd ../s_t1450_bl_d1_m90_r2
pyPostAcs.py --tonal_separation -df 1 -ovr 0.5 -win hann -m 0 4 8 --align --filter_harmonics 1 100 --filter_shaft_order
cd ../s_t1450_bl_d1_m100_r1
pyPostAcs.py --tonal_separation -df 1 -ovr 0.5 -win hann -m 0 4 8 --align --filter_harmonics 1 100 --filter_shaft_order
cd ../s_t1450_bl_d1_m110_r1
pyPostAcs.py --tonal_separation -df 1 -ovr 0.5 -win hann -m 0 4 8 --align --filter_harmonics 1 100 --filter_shaft_order
cd ../s_t1450_bl_d1_m120_r1
pyPostAcs.py --tonal_separation -df 1 -ovr 0.5 -win hann -m 0 4 8 --align --filter_harmonics 1 100 --filter_shaft_order
