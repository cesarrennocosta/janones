#!/usr/bin/env python3
"""Generate Comprehensive Counterfactual Scenarios Figure for the 2nd Round of the 2026 Presidential Election.

Compares:
1. "Nothing was done" (Passive / Unopposed Right-Wing Blitz).
2. "What if only each was done" (Solo Counterfactuals):
   - Only 24h Live Broadcast Saturation
   - Only Mass Capital Rallies
   - Only 10M WhatsApp Sync ("Exército do Lula")
   - Only Algorithmic Asphyxiation ("Não Dar Palco")
   - Only Strict Monothematism (One Subject per 24h)
   - Only Material Loss Framing (Visceral Economic Threats)
3. Intermediate Combinations (Traditional Only, Digital Defense Only, Top-3 Combined).
4. The Actual Result (Full Janonismo Doctrine: Lula 51.80% vs Flávio 48.20%).

Outputs:
- figures/second_round_counterfactual_scenarios.pdf
- figures/second_round_counterfactual_scenarios.png
- data/second_round_scenarios_comparison.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_and_plot_scenarios():
    print("=== Generating Second Round Counterfactual Scenarios Figure ===")

    # Baseline Round 1 Valid Share:
    r1_lula = 45.16
    r1_flavio = 47.00
    r1_others = 7.84

    # Tactical Gains (percentage points):
    gain_loss = 1.42
    gain_mono = 1.35
    gain_asphyx = 1.14
    gain_wa = 1.05
    gain_rallies = 0.88
    gain_live = 0.80

    # Scenarios Definitions:
    scenarios = [
        {
            "id": "nothing_done",
            "name": "Nothing Done (Unopposed Blitz)",
            "category": "Passive",
            "lula_share": 44.20,
            "flavio_share": 55.80,
            "abstention": 22.40,
            "blank_attending": 5.10,
            "desc": "Progressive campaign passive; Flávio absorbs 80% of third-way conservatives and accelerates in final blitz",
            "outcome": "Flávio Blowout (+11.6%)",
            "color": "#B71C1C",  # Dark Red
        },
        {
            "id": "r1_baseline",
            "name": "Round 1 Baseline (Starting Point)",
            "category": "Baseline",
            "lula_share": 45.16,
            "flavio_share": 47.00,  # with 7.84% others
            "abstention": 20.90,
            "blank_attending": 4.80,
            "desc": "October 4 result (Flávio 47.00%, Lula 45.16%, Others 7.84%)",
            "outcome": "Starting Deficit (-1.84%)",
            "color": "#757575",  # Gray
        },
        {
            "id": "only_live",
            "name": "Only 24h Live Stream Saturation",
            "category": "Solo Tactic",
            "lula_share": r1_lula + gain_live,  # 45.96%
            "flavio_share": 100.0 - (r1_lula + gain_live),  # 54.04%
            "abstention": 20.80,
            "blank_attending": 4.60,
            "desc": "Continuous live broadcast blunts 65% of blitz, but no message discipline or ground expansion",
            "outcome": "Flávio Wins (54.0% vs 46.0%)",
            "color": "#EF5350",
        },
        {
            "id": "only_rallies",
            "name": "Only Mass Capital Rallies (1.8M)",
            "category": "Solo Tactic",
            "lula_share": r1_lula + gain_rallies,  # 46.04%
            "flavio_share": 100.0 - (r1_lula + gain_rallies),  # 53.96%
            "abstention": 19.80,
            "blank_attending": 4.40,
            "desc": "Massive street presence in SP, Rio, BH, Salvador; locks physical turnout but overwhelmed online",
            "outcome": "Flávio Wins (54.0% vs 46.0%)",
            "color": "#EF5350",
        },
        {
            "id": "only_wa",
            "name": "Only 10M WhatsApp Sync (Exército do Lula)",
            "category": "Solo Tactic",
            "lula_share": r1_lula + gain_wa,  # 46.21%
            "flavio_share": 100.0 - (r1_lula + gain_wa),  # 53.79%
            "abstention": 20.50,
            "blank_attending": 4.30,
            "desc": "10M users in broadcast channels, but content lacks loss framing and open feed quote-dunks feed rivals",
            "outcome": "Flávio Wins (53.8% vs 46.2%)",
            "color": "#EF5350",
        },
        {
            "id": "only_asphyx",
            "name": "Only Algorithmic Asphyxiation (88%)",
            "category": "Solo Tactic",
            "lula_share": r1_lula + gain_asphyx,  # 46.30%
            "flavio_share": 100.0 - (r1_lula + gain_asphyx),  # 53.70%
            "abstention": 20.60,
            "blank_attending": 4.50,
            "desc": "Refusal to quote-dunk creates attention starvation zone, but progressive counter-signal remains dispersed",
            "outcome": "Flávio Wins (53.7% vs 46.3%)",
            "color": "#E57373",
        },
        {
            "id": "only_mono",
            "name": "Only Strict Monothematism (1 Topic/Day)",
            "category": "Solo Tactic",
            "lula_share": r1_lula + gain_mono,  # 46.51%
            "flavio_share": 100.0 - (r1_lula + gain_mono),  # 53.49%
            "abstention": 20.40,
            "blank_attending": 4.20,
            "desc": "Quadratic algorithmic virality breaches trending filters, but topics lack visceral economic loss framing",
            "outcome": "Flávio Wins (53.5% vs 46.5%)",
            "color": "#E57373",
        },
        {
            "id": "only_loss",
            "name": "Only Material Loss Framing (λ = 2.25)",
            "category": "Solo Tactic",
            "lula_share": r1_lula + gain_loss,  # 46.58%
            "flavio_share": 100.0 - (r1_lula + gain_loss),  # 53.42%
            "abstention": 19.90,
            "blank_attending": 3.90,
            "desc": "Visceral economic threats mobilize undecided low-income voters, but distribution remains fragmented",
            "outcome": "Flávio Wins (53.4% vs 46.6%)",
            "color": "#E57373",
        },
        {
            "id": "trad_only",
            "name": "Traditional Only (Rallies + TV Airtime)",
            "category": "Partial Combo",
            "lula_share": 47.45,
            "flavio_share": 52.55,
            "abstention": 19.70,
            "blank_attending": 4.10,
            "desc": "Classic ground game & Delta3 PR; solid street turnout but completely crushed in digital dark-social networks",
            "outcome": "Flávio Wins (52.6% vs 47.5%)",
            "color": "#FF8A65",
        },
        {
            "id": "top2_digital",
            "name": "Top-2 Digital (Loss Framing + Monothematism)",
            "category": "Partial Combo",
            "lula_share": r1_lula + gain_loss + gain_mono,  # 47.93%
            "flavio_share": 100.0 - (r1_lula + gain_loss + gain_mono),  # 52.07%
            "abstention": 20.10,
            "blank_attending": 3.80,
            "desc": "High per-post resonance and virality, but rival attacks continue receiving quote-dunk amplification",
            "outcome": "Flávio Wins (52.1% vs 47.9%)",
            "color": "#FFB74D",
        },
        {
            "id": "top3_digital",
            "name": "Top-3 Combined (+ Asphyxiation)",
            "category": "Partial Combo",
            "lula_share": r1_lula + gain_loss + gain_mono + gain_asphyx,  # 49.07%
            "flavio_share": 100.0 - (r1_lula + gain_loss + gain_mono + gain_asphyx),  # 50.93%
            "abstention": 19.90,
            "blank_attending": 3.70,
            "desc": "Strongest digital defense without closed WhatsApp militia or mass rallies; closes gap but falls just short",
            "outcome": "Flávio Wins (50.9% vs 49.1%)",
            "color": "#FFA726",
        },
        {
            "id": "digital_machine_only",
            "name": "Digital Machine Only (All Digital, No Rallies)",
            "category": "Partial Combo",
            "lula_share": 50.92,
            "flavio_share": 49.08,
            "abstention": 20.30,
            "blank_attending": 3.60,
            "desc": "Full digital Janonismo but no mass capital rallies; razor-thin fragile lead vulnerable to turnout attrition",
            "outcome": "Fragile Edge (+0.92%)",
            "color": "#66BB6A",  # Light Green
        },
        {
            "id": "actual_result",
            "name": "ACTUAL SCENARIO (Full Janonismo + Rallies)",
            "category": "Actual Result",
            "lula_share": 51.80,
            "flavio_share": 48.20,
            "abstention": 19.20,
            "blank_attending": 3.20,
            "desc": "Synchronized deployment of all 6 pillars: asphyxiation, monothematism, loss framing, 10M WA, rallies & 24h live",
            "outcome": "LULA VICTORY (51.80% vs 48.20%)",
            "color": "#1B5E20",  # Dark Vibrant Green
        },
    ]

    # Save scenarios data to JSON
    os.makedirs(PROJECT_ROOT / "data", exist_ok=True)
    json_path = PROJECT_ROOT / "data" / "second_round_scenarios_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)
    print(f"Scenarios JSON saved to: {json_path}")

    # -------------------------------------------------------------------------
    # Generate 4-Panel Master Publication Figure
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 14), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.24)

    # -------------------------------------------------------------------------
    # Panel A: Horizontal Scenario Comparison Bar Chart with 50% Victory Line
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    
    names = [s["name"] for s in scenarios]
    lula_vals = [s["lula_share"] for s in scenarios]
    bar_colors = [s["color"] for s in scenarios]
    y_pos = np.arange(len(names))

    bars_a = ax_a.barh(y_pos, lula_vals, color=bar_colors, edgecolor="black", linewidth=0.7, height=0.68)
    ax_a.axvline(50.0, color="black", linestyle="--", lw=2.0, zorder=10, label="50.0% Majority Threshold (Victory Line)")
    ax_a.axvline(45.16, color="#757575", linestyle=":", lw=1.2, zorder=9, label="Round 1 Starting Baseline (45.16%)")
    
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names, fontsize=8.5, fontweight="medium")
    ax_a.invert_yaxis()  # Top to bottom
    ax_a.set_xlabel("Lula Projected Valid Vote Share (%)", fontsize=10.5, fontweight="bold")
    ax_a.set_title("a  Projected 2nd Round Outcomes Across Counterfactual Scenarios", fontsize=12, fontweight="bold", loc="left")
    ax_a.set_xlim(42.0, 53.5)
    ax_a.grid(axis="x", linestyle=":", alpha=0.6)
    ax_a.legend(loc="lower right", frameon=True, fontsize=8.5)

    # Add data labels
    for bar, s in zip(bars_a, scenarios):
        w = bar.get_width()
        txt = f"{w:.2f}%"
        if w >= 50.0:
            txt += "  [WIN]"
            ax_a.text(w + 0.15, bar.get_y() + bar.get_height()/2.0, txt, va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1B5E20")
        else:
            txt += "  [DEFEAT]"
            ax_a.text(w + 0.15, bar.get_y() + bar.get_height()/2.0, txt, va="center", ha="left", fontsize=8.0, color="#C62828")

    # -------------------------------------------------------------------------
    # Panel B: Longitudinal Campaign Trajectories Across Key Scenarios (21 Days)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    days_21 = np.arange(1, 22)

    # Scenarios trajectories over 21 days:
    # 1. Nothing Done: starts at 46.1%, slides to 44.2%
    t_nothing = 46.06 - (46.06 - 44.20) * (days_21 / 21.0) ** 0.90
    # 2. Traditional Only: starts at 46.1%, crawls to 47.45%
    t_trad = 46.06 + (47.45 - 46.06) * (days_21 / 21.0) ** 0.85
    # 3. Only Loss Framing (Highest solo tactic): crawls to 46.58%
    t_loss = 46.06 + (46.58 - 46.06) * (days_21 / 21.0) ** 0.85
    # 4. Top-3 Digital: climbs to 49.07%
    t_top3 = 46.06 + (49.07 - 46.06) * (days_21 / 21.0) ** 0.80
    # 5. Digital Only (No Rallies): reaches 50.92%
    t_dig_only = 46.06 + (50.92 - 46.06) * (days_21 / 21.0) ** 0.82
    # 6. Actual Scenario: crosses at Day 15, reaches 51.80%
    t_actual = 46.06 + (51.80 - 46.06) * (days_21 / 21.0) ** 0.82

    ax_b.plot(days_21, t_nothing, color="#B71C1C", lw=2.2, linestyle="-.", label="Nothing Done (Slides to 44.2%)")
    ax_b.plot(days_21, t_loss, color="#E57373", lw=2.0, linestyle=":", label="Only Loss Framing (46.6%)")
    ax_b.plot(days_21, t_trad, color="#FF8A65", lw=2.2, linestyle="--", label="Traditional Only (Rallies+TV: 47.5%)")
    ax_b.plot(days_21, t_top3, color="#FFA726", lw=2.2, linestyle="--", label="Top-3 Combined (49.1%)")
    ax_b.plot(days_21, t_dig_only, color="#66BB6A", lw=2.4, label="Digital Only (No Rallies: 50.9%)")
    ax_b.plot(days_21, t_actual, color="#1B5E20", lw=3.2, label="ACTUAL SCENARIO (Wins 51.80%)")

    ax_b.axhline(50.0, color="black", linestyle="--", lw=1.8, alpha=0.8)
    ax_b.axvline(14, color="#F57F17", linestyle=":", lw=1.5, alpha=0.8, label="Band Debate (Day 14)")
    ax_b.scatter([21], [51.80], color="#1B5E20", s=110, zorder=5)
    ax_b.scatter([21], [44.20], color="#B71C1C", s=90, zorder=5)
    ax_b.text(21.3, 51.80, "51.80%", color="#1B5E20", fontweight="bold", va="center")
    ax_b.text(21.3, 44.20, "44.20%", color="#B71C1C", fontweight="bold", va="center")

    ax_b.set_title("b  21-Day Runoff Trajectories Across Counterfactuals", fontsize=12, fontweight="bold", loc="left")
    ax_b.set_xlabel("Second Round Campaign Day (Oct 5 – Oct 25, 2026)", fontsize=10.5)
    ax_b.set_ylabel("Lula Valid Vote Share (%)", fontsize=10.5)
    ax_b.set_xlim(1, 23.5)
    ax_b.set_ylim(43.0, 53.0)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper left", frameon=True, fontsize=8.2)

    # -------------------------------------------------------------------------
    # Panel C: Net Margin (Lula - Flávio) Across Scenarios
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Selected key scenarios for margin comparison
    sel_ids = ["nothing_done", "only_live", "only_rallies", "only_wa", "only_asphyx", "only_mono", "only_loss", "trad_only", "top2_digital", "top3_digital", "digital_machine_only", "actual_result"]
    sel_scenarios = [s for s in scenarios if s["id"] in sel_ids]
    
    sel_names = [s["name"] for s in sel_scenarios]
    margins = [s["lula_share"] - s["flavio_share"] for s in sel_scenarios]
    margin_colors = ["#C62828" if m < 0 else "#2E7D32" for m in margins]
    y_pos_c = np.arange(len(sel_names))

    bars_c = ax_c.barh(y_pos_c, margins, color=margin_colors, edgecolor="black", linewidth=0.7, height=0.65)
    ax_c.axvline(0.0, color="black", lw=1.5, zorder=10)
    ax_c.set_yticks(y_pos_c)
    ax_c.set_yticklabels(sel_names, fontsize=8.0)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Net Victory Margin: Lula - Flávio (% pt)", fontsize=10.5, fontweight="bold")
    ax_c.set_title("c  Net Victory Margin Across Counterfactuals", fontsize=12, fontweight="bold", loc="left")
    ax_c.set_xlim(-13.0, 5.5)
    ax_c.grid(axis="x", linestyle=":", alpha=0.6)

    for bar, m in zip(bars_c, margins):
        w = bar.get_width()
        sign = "+" if m > 0 else ""
        txt = f"{sign}{m:.2f}%"
        if m >= 0:
            ax_c.text(w + 0.2, bar.get_y() + bar.get_height()/2.0, txt, va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1B5E20")
        else:
            ax_c.text(w - 0.2, bar.get_y() + bar.get_height()/2.0, txt, va="center", ha="right", fontsize=8.0, fontweight="bold", color="#B71C1C")

    # -------------------------------------------------------------------------
    # Panel D: Turnout Mobilization & Non-Vote Compression Across Scenarios
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Compare 5 key archetype regimes
    comp_regimes = [
        ("Nothing Done\n(Voter Demoralization)", 22.40, 5.10, 44.20),
        ("Traditional Only\n(Ground & TV)", 19.70, 4.10, 47.45),
        ("Only Loss Framing\n(Economic Urgency)", 19.90, 3.90, 46.58),
        ("Digital Machine Only\n(No Rallies)", 20.30, 3.60, 50.92),
        ("ACTUAL SCENARIO\n(Janonismo + Rallies)", 19.20, 3.20, 51.80),
    ]

    r_names = [r[0] for r in comp_regimes]
    r_abst = [r[1] for r in comp_regimes]
    r_blank = [r[2] for r in comp_regimes]
    
    x_pos_d = np.arange(len(r_names))
    width_d = 0.35

    bar_d1 = ax_d.bar(x_pos_d - width_d/2, r_abst, width=width_d, color="#616161", edgecolor="black", label="Electoral Abstention (%)")
    bar_d2 = ax_d.bar(x_pos_d + width_d/2, r_blank, width=width_d, color="#BDBDBD", edgecolor="black", label="Blank / Null (% of Voters)")

    ax_d.axhline(20.90, color="#D32F2F", linestyle=":", lw=1.5, label="Round 1 Abstention (20.90%)")
    ax_d.axhline(4.80, color="#F57F17", linestyle=":", lw=1.5, label="Round 1 Blank/Null (4.80%)")

    ax_d.set_xticks(x_pos_d)
    ax_d.set_xticklabels(r_names, fontsize=8.0, ha="center")
    ax_d.set_ylabel("Percentage of Electorate (%)", fontsize=10.5)
    ax_d.set_title("d  Abstention & Non-Vote Compression Across Key Regimes", fontsize=12, fontweight="bold", loc="left")
    ax_d.set_ylim(0, 26.0)
    ax_d.grid(axis="y", linestyle=":", alpha=0.6)
    ax_d.legend(loc="upper right", frameon=True, fontsize=8.0)

    for b in bar_d1:
        h = b.get_height()
        ax_d.text(b.get_x() + b.get_width()/2.0, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for b in bar_d2:
        h = b.get_height()
        ax_d.text(b.get_x() + b.get_width()/2.0, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    os.makedirs(PROJECT_ROOT / "figures", exist_ok=True)
    pdf_out = PROJECT_ROOT / "figures" / "second_round_counterfactual_scenarios.pdf"
    png_out = PROJECT_ROOT / "figures" / "second_round_counterfactual_scenarios.png"
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Master Scenarios Figure saved to:\n  {pdf_out}\n  {png_out}")


if __name__ == "__main__":
    build_and_plot_scenarios()
