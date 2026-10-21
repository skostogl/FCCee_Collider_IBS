"""Equilibrium vertical emittance and closed orbit with final-doublet cryostat motion, VSEI bends OFF and ON (beam 1).

Radiation model 'mean', energy loss compensated by the cavity on the *nominal* lattice (kept fixed when the doublets move).
Each (state, seed, temporal phase) is one task: shift the 16 doublet cryostats (rigid, one random phase each), 6D twiss
with radiation_analysis (equilibrium emittances), then restore.
"""
import os, re, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

LATTICE = Path('/Users/skostogl/Documents/workspace/workspace_2026/FCC/10_2026/tapering/acc-models-fcc-ee/108/acc-models-fcc-ee/lattices/z/fccee_z_beam_1_positrons.json')
OUT = Path(__file__).resolve().parent/'results'/'emittance'
IP_NAMES = ('ipa', 'ipd', 'ipg', 'ipj')
CRYOSTATS = [(ip, side, magnet) for ip in IP_NAMES for side in ('L', 'R') for magnet in ('QF1', 'QD0')]
MAGNET_RE = re.compile(r'^(qd0|qf1)([a-d])([lr])(\.\d+)?$')
# state -> (kvsei1, with_wigglers); kvsei1 None = lattice value. Wiggler strengths come from match_wigglers.py
STATE_DEFS = {'off': (0.0, False), 'on': (None, False), 'wig_off': (0.0, True), 'wig_on': (None, True)}
STATES = os.environ.get('STATES', 'off,on').split(',')
SUFFIX = os.environ.get('TAG_SUFFIX', '')
KVSEI1 = {state: STATE_DEFS[state][0] for state in STATES}
AMPLITUDE_RMS_NM = float(os.environ.get('AMPLITUDE_RMS_NM', 15.0))
SEEDS = np.arange(42, 82)
THETAS = np.linspace(0.0, 2.0*np.pi, 5, endpoint=False)

def random_phases(seed):
    return np.random.default_rng(int(seed)).uniform(0.0, 2*np.pi, len(CRYOSTATS))

LINES, MEMBERS, REF = {}, {}, {}

def twiss_radiation(line):
    return line.twiss(radiation_analysis=True)

def init_worker():
    import xtrack as xt
    for state, kvsei in KVSEI1.items():
        if STATE_DEFS[state][1]:
            import json, wigglers
            v = json.loads((OUT/'wiggler_match.json').read_text())[state]['on_wiggler_v']
            env, line = wigglers.make_line(LATTICE, kvsei, v)
        else:
            env = xt.Environment.from_json(LATTICE)
            line = env[list(env.lines)[0]]
            if kvsei is not None:
                env['kvsei1'] = kvsei
        line.configure_radiation(model='mean')
        line.compensate_radiation_energy_loss()
        table = line.get_table()
        names = np.asarray(table.name, dtype=str); s = np.asarray(table.s, dtype=float); circumference = s[-1]
        ip_s = {ip: float(s[names == ip][0]) for ip in IP_NAMES}
        members = {c: [] for c in CRYOSTATS}
        for name, s_magnet in zip(names, s):
            match = MAGNET_RE.match(name)
            if not match:
                continue
            distance = {ip: (s_magnet - s_ip + circumference/2) % circumference - circumference/2 for ip, s_ip in ip_s.items()}
            ip = min(distance, key=lambda key: abs(distance[key]))
            if abs(distance[ip]) > 20.0:
                continue
            members[(ip, 'R' if distance[ip] > 0 else 'L', 'QD0' if match.group(1) == 'qd0' else 'QF1')].append(name)
        tw = twiss_radiation(line)
        LINES[state], MEMBERS[state] = line, members
        REF[state] = {'y': np.asarray(tw.y, float), 'dy': np.asarray(tw.dy, float), 's': np.asarray(tw.s, float),
                      'ip_y': np.asarray(tw.rows[list(IP_NAMES)].y, float), 'ex': float(tw.eq_gemitt_x), 'ey': float(tw.eq_gemitt_y)}

def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))

def run_task(task):
    state, seed, k, amplitude_nm = task
    line, members, ref = LINES[state], MEMBERS[state], REF[state]
    theta = THETAS[k]
    shifts = np.sqrt(2.0)*amplitude_nm*1e-9*np.cos(theta + random_phases(seed))
    applied = []
    for cryostat, value in zip(CRYOSTATS, shifts):
        for name in members[cryostat]:
            line[name].shift_y = float(value); applied.append(value)
    try:
        tw = twiss_radiation(line)
        dy_orbit = np.asarray(tw.y, float) - ref['y']
        result = {'state': state, 'seed': int(seed), 'theta_index': k, 'amplitude_rms_nm': amplitude_nm, 'converged': True,
                  'input_rms_nm': 1e9*rms(applied), 'eq_gemitt_x_m': float(tw.eq_gemitt_x), 'eq_gemitt_y_m': float(tw.eq_gemitt_y),
                  'orbit_y_rms_um': 1e6*rms(dy_orbit), 'orbit_y_max_um': 1e6*float(np.max(np.abs(dy_orbit))),
                  'delta_dy_rms_mm': 1e3*rms(np.asarray(tw.dy, float) - ref['dy']),
                  'delta_dy_max_mm': 1e3*float(np.max(np.abs(np.asarray(tw.dy, float) - ref['dy'])))}
        for ip, value in zip(IP_NAMES, np.asarray(tw.rows[list(IP_NAMES)].y, float) - ref['ip_y']):
            result[f'{ip}_offset_nm'] = 1e9*float(value)
        orbit = (np.asarray(tw.s, float), 1e6*dy_orbit) if (seed == SEEDS[0] and k == 0) else None
    except Exception as error:
        result = {'state': state, 'seed': int(seed), 'theta_index': k, 'amplitude_rms_nm': amplitude_nm, 'converged': False}
        orbit = None
    for cryostat in CRYOSTATS:
        for name in members[cryostat]:
            line[name].shift_y = 0.0
    return result, orbit

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tasks = [(state, seed, k, AMPLITUDE_RMS_NM) for state in KVSEI1 for seed in SEEDS for k in range(len(THETAS))]
    start = time.time()
    with Pool(n_proc, initializer=init_worker) as pool:
        out = list(pool.imap_unordered(run_task, tasks, chunksize=2))
    frame = pd.DataFrame([r for r, _ in out])
    tag = f'{AMPLITUDE_RMS_NM:g}nm{SUFFIX}'
    frame.to_csv(OUT/f'emittance_scan_{tag}.csv', index=False)
    orbits = {f'{r["state"]}_{key}': value for (r, orbit) in out if orbit is not None for key, value in zip(('s', 'y_um'), orbit)}
    np.savez(OUT/f'orbit_seed{SEEDS[0]}_theta0_{tag}.npz', **orbits)
    init_worker()
    pd.DataFrame([{'state': s, 'eq_gemitt_x_m': REF[s]['ex'], 'eq_gemitt_y_m': REF[s]['ey']} for s in REF]).to_csv(OUT/f'nominal_emittance{SUFFIX}.csv', index=False)
    print(f'done {len(tasks)} tasks in {time.time() - start:.0f} s', flush=True)
