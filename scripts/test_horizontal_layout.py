import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

fig = plt.figure(figsize=(26, 7.8), dpi=300)
gs = GridSpec(1, 4, width_ratios=[1.1, 1.1, 1.0, 1.45], wspace=0.25)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

# Column 4 nested grid
gs_col4 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 3], height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.25)
ax_d = fig.add_subplot(gs_col4[0, :])
ax_e = fig.add_subplot(gs_col4[1, 0])
ax_f = fig.add_subplot(gs_col4[1, 1])

ax_a.set_title("a  Electoral Outcomes")
ax_b.set_title("b  Vote Intentions (71 Days)")
ax_c.set_title("c  Candidate Rejection")
ax_d.set_title("d  Quaest IPD (Survey Waves)")
ax_e.set_title("e  Gross Interactions")
ax_f.set_title("f  Follower Footprint")

plt.savefig("figures/test_layout.png", bbox_inches="tight")
plt.close()
print("Layout test saved successfully!")
