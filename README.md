# FCCee Collider IBS

FCC-ee studies of intrabeam scattering (IBS) and synchrotron-radiation damping using Xsuite.

## Contents

- `ibs_sr_damping_times.ipynb`: SR damping times and IBS growth rates.
- `fccee_z.json`: lattice used by the IBS notebook.

## Running

Use a Python environment with Xsuite (`xtrack`, `xfields`, and their dependencies), NumPy, SciPy, Matplotlib, pandas, and Jupyter. Open the main IBS notebook from the repository root so its lattice path resolves.

## Review notes

The files preserve the source calculations and saved outputs. In the IBS notebook, the energy-loss printout divides eV by 1e6, so its displayed unit should be MeV rather than GeV. The Nagaitsev calculation warns about nonzero vertical dispersion; compare against Bjorken-Mtingwa before interpreting those rates. The final percentage compares instantaneous IBS growth and SR damping rates, rather than solving for equilibrium emittance.
