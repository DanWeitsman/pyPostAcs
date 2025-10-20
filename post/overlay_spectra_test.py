import overlay_spectra
import os

# Set working directory
os.chdir("/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/RES_BLADE_9_25/data/9_11/ng2")
args=["-c","bgd_g20","bgd_g30","bgd_g40","bgd_g50","bgd_g60","bgd_g70","bgd_g80","bgd_g90","bgd_g100","-df", "1","-ovr", "0.5", "-win" , "hann","-m","5","1","8"]
overlay_spectra.main(args)
