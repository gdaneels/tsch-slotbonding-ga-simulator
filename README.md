Contains the __6tisch simulator and GA to do 6TiSCH simulations with slotbonding__. The GA for the simulator-only simulations is (ga-heuristic-slots-simulation.py). These simulations use the feasibility heuristic that use the OFDM modulations and tries to schedule in the same spectrum.

Additionally, also contains the GA implementation (ga-heuristic-slots.py) for calculating the PDRs with monitoring (link reliability) results from the __imec Wireless OfficeLab__. This GA uses the heuristic that uses two spectra for the 50 and 1000 kbps PHY seperatly (heuristic2phys.phy).

Developed by Glenn Daneels (glenn.daneels@uantwerpen.be).

