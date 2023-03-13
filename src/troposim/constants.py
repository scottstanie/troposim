from math import pi

SENTINEL_WAVELENGTH = 5.5465763  # cm
PHASE_TO_CM_S1 = SENTINEL_WAVELENGTH / (4 * pi)
P2MM_S1 = PHASE_TO_CM_S1 * 10 * 365  # (cm / day) -> (mm / yr)
