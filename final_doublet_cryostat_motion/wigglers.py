"""Vertical wigglers for the LCC 108 Z lattice, following FCCeePerformanceStudies.add_fccee_wigglers.

Differences from the repository function (see the notebook):
* the bends are tilted by pi/2 (vertical wigglers). The repository code creates untilted bends, which act horizontally
  and cannot generate vertical emittance;
* one group of 6 poles per straight (A, D, G, J) in the 17 m drift between qf9m_r.k and qd8m_r.k of the 108 lattice,
  instead of the two groups per straight of the LCC 106 model (there is no second free 17 m drift on that side).
Geometry/field: arXiv:1909.12245 (B+ = 0.7 T, L+ = 0.43 m, L- = 3 L+, 0.25 m gaps).
"""
import numpy as np
import xtrack as xt

TARGET_EY_M = 2.419172e-12

def add_vertical_wigglers(env, line):
    brho = line.particle_ref.p0c[0]/line.particle_ref.q0/0.299792458*1e-9
    B_plus, L_plus, r_asym, L_dis, L_qa = 0.7, 0.430, 6, 0.250, 2.9
    L_minus = L_plus*r_asym/2.0
    line.vars['on_wiggler_v'] = 0
    env.new('MWIM', xt.Bend, k0_from_h=True, length=L_minus, rot_s_rad=np.pi/2,
            angle=f'(-0.5 * (on_wiggler_v * {B_plus} * {L_plus} / {brho}))')
    env.new('MWIP', xt.Bend, k0_from_h=True, length=L_plus, rot_s_rad=np.pi/2,
            angle=f'(on_wiggler_v * {B_plus} * {L_plus} / {brho})')
    offsets1 = [0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.5, 5.0, 5.5, 6.5, 7.0, 7.5]
    offsets2 = [0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0]
    table = line.get_table(); names = np.asarray(table.name, dtype=str); s = np.asarray(table.s, dtype=float)
    inserts = []
    for k, straight in enumerate(('A', 'D', 'G', 'J')):
        s_start = float(s[names == f'qf9m_r.{k}'][0]) + L_qa
        for e, element in enumerate(('A', 'B', 'C', 'D', 'E', 'F')):
            name = f'MWI.{element}5R{straight}'
            env.new(name=name, parent='MWIP' if element in ('B', 'E') else 'MWIM')
            inserts.append(env.place(name=name, at=s_start + (e + 1)*L_dis + offsets1[e]*L_minus + offsets2[e]*L_plus))
    line.insert(inserts)

def equilibrium(line):
    # call once per freshly built line: compensate_radiation_energy_loss cannot be repeated on a compensated line
    line.configure_radiation(model='mean')
    line.compensate_radiation_energy_loss()
    tw = line.twiss(radiation_analysis=True)
    return float(tw.eq_gemitt_x), float(tw.eq_gemitt_y), tw

def make_line(lattice, kvsei1, on_wiggler_v):
    env = xt.Environment.from_json(lattice); line = env[list(env.lines)[0]]
    if kvsei1 is not None:
        env['kvsei1'] = kvsei1
    add_vertical_wigglers(env, line)
    line.vars['on_wiggler_v'] = on_wiggler_v
    return env, line

def match_on_wiggler_v(lattice, kvsei1, target_ey=TARGET_EY_M, tolerance=2e-4, start=0.27, exponent=5.0, verbose=True):
    """Tune on_wiggler_v so that eq_gemitt_y (6D radiation analysis, energy loss compensated) equals target_ey.

    eps_y grows like v^5 (curly-H ~ v^2 times 1/rho^3 ~ v^3), so the fixed-point update uses exponent 5."""
    v = start
    for iteration in range(8):
        env, line = make_line(lattice, kvsei1, v)
        ex, ey, _ = equilibrium(line)
        if verbose:
            print(f'  iter {iteration}: on_wiggler_v = {v:.6f}  eps_y = {ey:.6e}  eps_x = {ex:.6e}', flush=True)
        if abs(ey/target_ey - 1) < tolerance:
            return v, ex, ey
        v *= (target_ey/ey)**(1.0/exponent)
    raise RuntimeError('wiggler matching did not converge')
