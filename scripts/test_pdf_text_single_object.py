import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.kerning_factor": 0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
})

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0, 1], [0, 1])
ax.set_title("Test Single Text Object")
ax.set_xlabel("Campaign Day (Aug 16 - Oct 25, 2026)")
ax.set_ylabel("Valid Vote Intentions / Share (%)")
ax.text(0.5, 0.5, "Single Editable Text String in Illustrator", ha="center", va="center")

fig.savefig("figures/test_text.pdf")
fig.savefig("figures/test_text.svg")
plt.close()
print("Test PDF and SVG generated successfully.")
