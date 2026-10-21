"""Match the vertical wigglers to eps_y = 2.419172 pm for VSEI OFF and ON; write results/emittance/wiggler_match.json."""
import json
from pathlib import Path
import xtrack as xt
import wigglers

LATTICE = Path('/Users/skostogl/Documents/workspace/workspace_2026/FCC/10_2026/tapering/acc-models-fcc-ee/108/acc-models-fcc-ee/lattices/z/fccee_z_beam_1_positrons.json')
out = {}
for state, kvsei in (('wig_off', 0.0), ('wig_on', None)):
    print(state, flush=True)
    v, ex, ey = wigglers.match_on_wiggler_v(LATTICE, kvsei)
    out[state] = {'on_wiggler_v': v, 'eq_gemitt_x_m': ex, 'eq_gemitt_y_m': ey, 'target_ey_m': wigglers.TARGET_EY_M}
Path('results/emittance').mkdir(parents=True, exist_ok=True)
Path('results/emittance/wiggler_match.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
