"""Append diagnostic experiments239-240 at the existing poster scale."""
from pathlib import Path
import json,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
r=json.loads((HERE/'analysis.json').read_text());ph=json.loads((HERE/'phase-profile.json').read_text())['runs']
p=ROOT/'tools/plot_tiny_hillclimb.py';s=p.read_text();assert 'def draw_ten_update_profile' not in s
shutil.copy2(ROOT/'examples/tiny/hillclimb.png','/tmp/tiny-prl-pre240.png')
for a,b in [('1-238','1-240'),('all 238 experiments','all 240 experiments'),('== 238','== 240'),('range(1, 239)','range(1, 241)'),('ax.set_xlim(0, 238)','ax.set_xlim(0, 240)'),('150, 175, 238]','150, 175, 240]')]:s=s.replace(a,b)
s=s.replace('PAGE_H = 69.0','PAGE_H = 76.1')
s=s.replace('next(exps for key, _, exps in LANES if key == "R").append(238)','next(exps for key, _, exps in LANES if key == "R").append(238)\nCODES["D"].append(239)\nCODES["X"].append(240)\nnext(exps for key, _, exps in LANES if key == "R").append(239)\nLANES[-1][2].append(240)')
segments=[]
for row in ph:
 t=row['native_trainer_totals_seconds'];v=[row['startup_to_training_loop_log_seconds'],t['time/wait_for_batch'],t['time/forward_backward'],t['time/broadcast_weights'],row['final_checkpoint_log_seconds']]
 v.append(row['whole_rl_seconds']-sum(v));segments.append(v)
code='\n\nTEN_UPDATE_SEGMENTS = '+repr(segments)+'\n'
code+='''

def draw_ten_update_profile(pg):
    ax = pg.axes(1.95, 69.65, 6.95, 1.15)
    names = ["Startup", "Wait for batch", "Train + optimizer", "Publish weights", "Final checkpoint", "Other"]
    colors = [PALE, INK, ACCENT, ACCENT_TINT, SUB, CARD]
    for y, values, total in zip([1, 0], TEN_UPDATE_SEGMENTS, [158.865, 160.390]):
        left = 0
        for i, (value, name, color) in enumerate(zip(values, names, colors)):
            ax.barh(y, value, left=left, height=0.36, color=color, edgecolor=RULE, linewidth=0.4, label=name if y == 1 else None)
            if value >= 8:
                ax.text(left + value / 2, y, f"{value:.1f}", ha="center", va="center", fontsize=6.2,
                        fontfamily=pg.mono, color="white" if i in [1, 2] else INK)
            left += value
        ax.text(total + 1.5, y, f"{total:.1f} s", va="center", fontsize=6.5, fontfamily=pg.mono, color=INK)
    ax.set_yticks([1, 0], ["A · control\\n29,126 tokens", "B · early\\n31,677 tokens"])
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("whole RL process · seconds · native trainer buckets plus approximate log intervals", fontsize=6.2, fontfamily=pg.prose, color=SUB)
    clean_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.36), prop={"family": pg.prose, "size": 6.0})
    pg.text(GUTTER, 71.65, "239 · INCONCLUSIVE. One process per arm; early generates 8.8% more tokens. Trainer metrics match through update 4, diverge at 5.", size=6.7, color=SUB)
    pg.text(GUTTER, 71.82, "Wait dominates: 93.50 / 101.06 s; train + optimizer 9.41 / 9.88 s. Publish 13.48 / 12.64 s; final checkpoint about 6 / 5 s.", size=6.7, color=SUB)
    pg.text(GUTTER, 71.99, "No repeated speed verdict or new verified RL point. Both runs finish 10 updates and checkpoints; guards pass with zero job swap.", size=6.7, color=SUB)


def draw_serving_profile_screen(pg):
    ax = pg.axes(1.65, 73.15, 3.55, 1.1)
    vals = [10.271814, 1.692128, 0.796867, 1.823836]
    labels = ["Pure decode", "Mixed", "Prefill", "Sampling"]
    for y, value in zip([3, 2, 1, 0], vals):
        ax.barh(y, value, height=0.45, color=ACCENT if y == 3 else PALE)
        ax.text(value + 0.15, y, f"{value:.2f}", va="center", fontsize=6.5, fontfamily=pg.mono, color=INK)
    ax.set_yticks([3, 2, 1, 0], labels)
    ax.set_xlim(0, 12)
    ax.set_xlabel("measured serving stream spans · seconds", fontsize=6.2, fontfamily=pg.prose, color=SUB)
    clean_axes(ax)
    pg.text(5.6, 73.15, "240 · B1 shared scratch", size=9, color=INK)
    pg.text(5.6, 73.42, "38,440 to 23,576 B; same 253 registers.", size=6.5, color=SUB, mono=True)
    pg.text(5.6, 73.69, "Trim only: -0.04 / +0.27 / +0.02%", size=6.5, color=SUB, mono=True)
    pg.text(5.6, 73.88, "L1 control: -0.01 / +0.24 / -0.01%", size=6.5, color=SUB, mono=True)
    pg.text(5.6, 74.10, "Contexts 32 / 128 / 1024; both rejected.", size=6.7, color=SUB)
    pg.text(GUTTER, 74.72, "239 · Separate two-update V2 diagnostic, 6,810 measured calls. Pure decode accounts for 70.4% of these stream spans.", size=6.7, color=SUB)
    pg.text(GUTTER, 74.89, "Stream spans include launch gaps; CPU work overlaps them. This is neither isolated kernel busy time nor a wall-time comparison.", size=6.7, color=SUB)
    pg.text(GUTTER, 75.06, "240 · Exact outputs/graphs pass; B2 machine code unchanged. Both decoder screens rejected; no serving or RL sweep followed.", size=6.7, color=SUB)
'''
s=s.replace('\ndef build(output:',code+'\n\ndef build(output:')
s=s.replace('    draw_footer(pg, 68.4)','''    pg.section(68.9, "19", "Ten updates expose a workload mismatch", "Exp 239 · same invariant-serving profile; native wall buckets, approximate startup/checkpoint log intervals; work differs.")
    draw_ten_update_profile(pg)
    pg.section(72.4, "20", "Serving profile narrows the next kernel target", "Exp 239 diagnostic spans; Exp 240 math-preserving shared-memory and cache-preference screens.")
    draw_serving_profile_screen(pg)
    draw_footer(pg, 75.5)''')
p.write_text(s)
