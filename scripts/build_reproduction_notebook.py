#!/usr/bin/env python3
"""Build the master interactive reproduction Jupyter Notebook."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

nb = {
    'cells': [],
    'metadata': {
        'language_info': {
            'name': 'python',
            'version': '3.10'
        },
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

def add_md(text):
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

def add_code(code):
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in code.strip().split('\n')]
    })

# Cell 0: Header
add_md("""# Late-Campaign Opinion Volatility and Algorithmic Counter-Mobilization in Polarized Elections
## Interactive Computational Reproduction Notebook & Simulation Suite

**Author:** César Rennó-Costa$^{1,*}$  
$^{1}$*Brain Institute, Federal University of Rio Grande do Norte (UFRN), Natal, RN, Brazil*  
$^*$Corresponding author: `rennocosta@neuro.ufrn.br`  
**Target Journal:** *Nature Human Behaviour* (2026)

---

### Abstract & Theoretical Overview
Democratic elections worldwide increasingly exhibit sudden, late-campaign volatility that produces persistent polling discrepancies despite the documented stability of voters' foundational political convictions. This study investigates the double polling error puzzle using an empirical digital twin of the 2026 Brazilian presidential election—a high-stakes quasi-natural experiment featuring a preliminary conservative surge followed by a historic runoff reversal.

This interactive notebook reproduces all six publication figures of the manuscript:
1. **Figure 1:** Empirical Data and Polling Dynamics of the 2026 Brazilian Presidential Election
2. **Figure 2:** Structure and Dynamics of the Urban Interpersonal Attitude Network
3. **Figure 3:** Structure and Gatekeeping Dynamics of the Traditional Mass Media Ecosystem
4. **Figure 4:** Algorithmic Curation, Influencer Hierarchies, and Multi-Stage Cascade Dynamics in Social Media
5. **Figure 5:** Longitudinal Electoral Trajectories and Parametric Sensitivity of the First Round
6. **Figure 6:** Strategic Evaluation of Algorithmic Counter-Tactics in the 2026 Runoff
""")

# Cell 1: Environment Setup
add_code("""import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display, Image

# Set project root
PROJECT_ROOT = Path(os.path.abspath('..'))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import replication modules
from replication.sim_figure1_empirical_data import generate_figure1
from replication.sim_figure2_urban_network import generate_figure2, run_urban_network_simulation
from replication.sim_figure3_traditional_media import generate_figure3, run_traditional_media_simulation
from replication.sim_figure4_social_media import generate_figure4, run_social_media_simulation
from replication.sim_figure5_first_round import generate_figure5, run_first_round_simulation
from replication.sim_figure6_second_round import generate_figure6, run_second_round_simulation

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print("OmniOpinion replication suite loaded successfully.")
""")

# Cell 2: Section 1 - Figure 1
add_md("""---
## Section 1: Figure 1 — Empirical Data and Polling Dynamics (2026 Presidential Election)

This module compiles and plots the multi-source empirical datasets from official and market sources:
- **Panels a–c:** Official TSE ballot results (Round 1: 47.03% vs 45.16%; Round 2: 51.79% vs 48.21%), Datafolha longitudinal polling trajectories, and candidate rejection dynamics.
- **Panels d–f:** Statutory free broadcast airtime (HGPE daily blocks shifting from +27% Lula advantage to strict parity), FEFC party fund allocations (R$ 4.96B pool), and ticket receipts.
- **Panels g–i:** Quaest Digital Popularity Index (+16.0 pt net turnaround swing), Bites daily interaction volume surges, and the legislative surrogate bench reach.
""")

add_code("""# Generate Figure 1: Empirical Observational Master Synthesis
pdf_fig1, png_fig1 = generate_figure1(output_dir=FIGURES_DIR)
display(Image(filename=str(png_fig1), width=950))
""")

# Cell 3: Section 2 - Figure 2
add_md("""---
## Section 2: Figure 2 — Urban Interpersonal Attitude Network & Two-Timescale Dynamics

Simulates the five-layer urban multiplex contact network (households, workplaces/schools, religious congregations, small-world friendship circles, and transit weak ties) and evaluates the two-timescale opinion-conviction elastic tether:

$$\\mathbf{x}_i(t+1) = \\mathrm{clip}_{[-1, 1]^D}\\!\\left(\\mathbf{s}_i(t) + \\frac{g_i^{\\mathrm{eff}}(t)}{g_i(t) + 10^{-12}}\\big(\\tilde{\\mathbf{x}}_i(t) - \\mathbf{s}_i(t)\\big)\\right)$$

where fast expressed opinions temporarily depart from anchored convictions under acute external shocks before elastically relaxing back to baseline.
""")

add_code("""# Run urban network simulation and render Figure 2
pdf_fig2, png_fig2 = generate_figure2(output_dir=FIGURES_DIR, mode="fast")
display(Image(filename=str(png_fig2), width=950))
""")

# Cell 4: Section 3 - Figure 3
add_md("""---
## Section 3: Figure 3 — Traditional Mass Media Gatekeeping & Market Polarization

Models the traditional media ecosystem across five modalities (TV, radio, press, outdoor signage, and rallies) and evaluates the impact of editorial tendentiousness across five market configurations:

$$\\mathbf{y}_{k,g,v}(t) = (1 - \\eta_{g,v})\,\\mathbf{f}_k(t) + \\eta_{g,v}\\,\\mathbf{m}_g + \\boldsymbol{\\xi}_{k,g,v}$$

Demonstrating that broadcast pulses operate on a fast decay timescale while interpersonal network deliberation maintains prolonged societal resonance.
""")

add_code("""# Run traditional media simulation and render Figure 3
pdf_fig3, png_fig3 = generate_figure3(output_dir=FIGURES_DIR)
display(Image(filename=str(png_fig3), width=950))
""")

# Cell 5: Section 4 - Figure 4
add_md("""---
## Section 4: Figure 4 — Algorithmic Curation, Influencer Hierarchies & Dark-Social Cascades

Simulates the open feed algorithmic curation engine with power-law influencer in-degrees and dual-valence recommendation scoring along the diagonal of controversy:

$$R_m = \\mathrm{Likes}_m + \\omega \\cdot \\mathrm{Dislikes}_m, \\qquad \\omega = 1.0$$

Capturing the three-stage cross-platform cascade lifecycle: open visual feeds (Stage 1), microblogging commentary debate (Stage 2), and dark-social encrypted micro-group forwarding (Stage 3, WhatsApp trust multiplier 0.85, 2.8x persuasion efficiency).
""")

add_code("""# Run social media simulation and render Figure 4
pdf_fig4, png_fig4 = generate_figure4(output_dir=FIGURES_DIR)
display(Image(filename=str(png_fig4), width=950))
""")

# Cell 6: Section 5 - Figure 5
add_md("""---
## Section 5: Figure 5 — First-Round Calibration (RMSE = 0.85%) & 5 Janones Mechanisms

Calibrates the full multi-layer agent-based model against longitudinal Datafolha tracking surveys and official TSE ballot tallies ($R^2 = 0.9955$, RMSE = 0.85 percentage points). Evaluates the five structural mechanisms driving late-campaign conservative momentum:
1. Church conviction priming (+1.80%, +2.13M votes)
2. Traditional media center-right editorial lean (+1.20%, +1.42M votes)
3. Coordinated bot astroturfing (+1.40%, +1.66M votes)
4. Asymmetric candidate rejection accumulation (+1.60%, +1.90M votes)
5. Acute 48-hour pre-election blitz (+2.20%, +2.61M votes)

Includes parametric sensitivity sweeps and error bars across Monte Carlo realizations.
""")

add_code("""# Run First-Round calibration and render Figure 5
pdf_fig5, png_fig5 = generate_figure5(output_dir=FIGURES_DIR, mode="fast")
display(Image(filename=str(png_fig5), width=950))
""")

# Cell 7: Section 6 - Figure 6
add_md("""---
## Section 6: Figure 6 — Second-Round Runoff & Janonismo Counter-Tactics (RMSE = 0.56%)

Simulates the second-round runoff under statutory 50/50 television parity, evaluating the six tactical pillars of *Janonismo Cultural* that overturned Flávio's initial lead to secure Lula's historic victory (51.79% vs 48.21%; $R^2 = 0.9128$, RMSE = 0.56 percentage points):
1. **Outrage-Starvation Embargo (T1):** Signal denial collapsing opposition algorithmic relevance by -48%.
2. **Monothematic Bandwidth Concentration (T2):** Concentrating surrogate output to quadruple virality via quadratic convexity.
3. **Loss-Aversion Framing (T3):** Mobilizing economically vulnerable centrists via prospect theory.
4. **Molecular Encrypted Cascades (T4):** Bypassing open-feed algorithms through trusted closed WhatsApp networks.
5. **Physical Turnout Locking (T5):** Mass rallies converting attendee turnout to certainty.
6. **Attention-Ceiling Saturation (T6):** Continuous 24-hour livestream sprint compressing rival blitz discovery by 65%.
""")

add_code("""# Run Second-Round runoff simulation and render Figure 6
pdf_fig6, png_fig6 = generate_figure6(output_dir=FIGURES_DIR, mode="fast")
display(Image(filename=str(png_fig6), width=950))
""")

# Cell 8: Conclusion & Citation Block
add_md("""---
## Citation & Academic Attribution

If you use this replication package, digital twin models, or simulation kernels, please cite:

```bibtex
@article{rennocosta2026latecampaign,
  title={Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections},
  author={Renn{\'o}-Costa, C{\'e}sar},
  journal={Nature Human Behaviour},
  year={2026},
  publisher={Nature Publishing Group},
  institution={Brain Institute, Federal University of Rio Grande do Norte (UFRN), Natal, RN, Brazil}
}
```
""")

# Save files
p1 = PROJECT_ROOT / 'notebooks' / 'paper_reproduction_figures.ipynb'
p2 = PROJECT_ROOT / 'replication' / 'paper_reproduction_figures.ipynb'

with open(p1, 'w') as f:
    json.dump(nb, f, indent=2)

with open(p2, 'w') as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated:\n - {p1}\n - {p2}")
