"""Direct closed-orbit scan (no linear approximation) of the b1/b2 IP offsets for the final-doublet cryostat motion.

For every (case, moved IP, amplitude, seed) the 5 temporal phases are applied to both rings, the closed orbit of
each ring is computed with Twiss (only_orbit) and the maximum over the temporal phases of |y_b1|, |y_b2| and
|y_b1 - y_b2| is stored at the observed IPs. Results are written per amplitude and the script can be resumed.
"""
import os, re, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

LATTICE_DIR = Path('/Users/skostogl/Documents/workspace/workspace_2026/FCC/10_2026/tapering/acc-models-fcc-ee/108/acc-models-fcc-ee/lattices/z')
LINE_FILES = {'b1': LATTICE_DIR/'fccee_z_beam_1_positrons.json', 'b2': LATTICE_DIR/'fccee_z_beam_2_electrons.json'}
OUT = Path(__file__).resolve().parent/'results'/'direct_scan'
IP_NAMES = ('ipa', 'ipd', 'ipg', 'ipj')
CRYOSTATS = [(ip, side, magnet) for ip in IP_NAMES for side in ('L', 'R') for magnet in ('QF1', 'QD0')]
MAGNET_RE = re.compile(r'^(qd0|qf1)([a-d])([lr])(\.\d+)?$')
SEEDS = np.arange(42, 82)
AMPLITUDES_NM = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
THETAS = np.linspace(0.0, 2.0*np.pi, 5, endpoint=False)

def random_phases(seed):
    return np.random.default_rng(int(seed)).uniform(0.0, 2*np.pi, len(CRYOSTATS))

def active_mask(case, moved_ip):
    if case == 'all_doublets':
        return np.ones(len(CRYOSTATS), bool)
    if case == 'both_sides':
        return np.array([c[0] == moved_ip for c in CRYOSTATS])
    return np.array([c[0] == moved_ip and c[1] == case[-1] for c in CRYOSTATS])

TASK_CASES = [('one_side_L', ip) for ip in IP_NAMES] + [('one_side_R', ip) for ip in IP_NAMES] + \
             [('both_sides', ip) for ip in IP_NAMES] + [('all_doublets', 'all')]

LINES, MEMBERS = {}, {}

def init_worker():
    import xtrack as xt
    for beam, path in LINE_FILES.items():
        env = xt.Environment.from_json(path)
        line = env[list(env.lines)[0]]
        line.configure_radiation(model=None)
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
            downstream = distance[ip] > 0
            side = ('R' if downstream else 'L') if beam == 'b1' else ('L' if downstream else 'R')
            members[(ip, side, 'QD0' if match.group(1) == 'qd0' else 'QF1')].append(name)
        LINES[beam], MEMBERS[beam] = line, members
        line.twiss(method='4d', only_orbit=True)            # builds the tracker / warms up the kernel

def ip_y(beam):
    twiss = LINES[beam].twiss(method='4d', only_orbit=True)
    return np.asarray(twiss.rows[list(IP_NAMES)].y, float)

def run_task(task):
    case, moved_ip, amplitude, seed = task
    mask = active_mask(case, None if case == 'all_doublets' else moved_ip)
    observed = list(IP_NAMES) if case == 'all_doublets' else [moved_ip]
    phases = random_phases(seed)
    peak = {beam: np.zeros(len(IP_NAMES)) for beam in ('b1', 'b2', 'diff')}
    try:
        for theta in THETAS:
            shifts = np.sqrt(2.0)*amplitude*1e-9*np.cos(theta + phases)
            for beam in LINES:
                for cryostat, value, on in zip(CRYOSTATS, shifts, mask):
                    if on:
                        for name in MEMBERS[beam][cryostat]:
                            LINES[beam][name].shift_y = float(value)
            y1, y2 = ip_y('b1'), ip_y('b2')
            for beam in LINES:
                for cryostat, on in zip(CRYOSTATS, mask):
                    if on:
                        for name in MEMBERS[beam][cryostat]:
                            LINES[beam][name].shift_y = 0.0
            peak['b1'] = np.maximum(peak['b1'], np.abs(y1)); peak['b2'] = np.maximum(peak['b2'], np.abs(y2))
            peak['diff'] = np.maximum(peak['diff'], np.abs(y1 - y2))
        ok = True
    except Exception as error:                                # closed orbit not found
        for beam in LINES:
            for cryostat in CRYOSTATS:
                for name in MEMBERS[beam][cryostat]:
                    LINES[beam][name].shift_y = 0.0
        peak = {beam: np.full(len(IP_NAMES), np.nan) for beam in peak}
        ok = False
    return [{'case': case, 'moved_ip': moved_ip, 'observed_ip': ip, 'amplitude_rms_nm': amplitude, 'seed': int(seed), 'converged': ok,
             'max_abs_b1_nm': 1e9*peak['b1'][IP_NAMES.index(ip)], 'max_abs_b2_nm': 1e9*peak['b2'][IP_NAMES.index(ip)],
             'max_abs_relative_nm': 1e9*peak['diff'][IP_NAMES.index(ip)]} for ip in observed]

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    init_worker()                                              # warm the compiled kernel once before forking workers
    with Pool(n_proc, initializer=init_worker) as pool:
        for amplitude in AMPLITUDES_NM:
            path = OUT/f'amplitude_{amplitude:g}nm.csv'
            if path.exists():
                print('skip', path.name, flush=True); continue
            tasks = [(case, ip, amplitude, seed) for case, ip in TASK_CASES for seed in SEEDS]
            start = time.time()
            rows = [row for result in pool.imap_unordered(run_task, tasks, chunksize=4) for row in result]
            pd.DataFrame(rows).to_csv(path.with_suffix('.tmp'), index=False); path.with_suffix('.tmp').replace(path)
            print(f'A = {amplitude:g} nm done in {time.time() - start:.0f} s ({len(tasks)} tasks)', flush=True)
