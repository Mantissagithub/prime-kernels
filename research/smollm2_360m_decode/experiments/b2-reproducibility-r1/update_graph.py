"""Append experiments 237 and 238 without moving existing panels."""
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
r=json.loads((HERE/'final-report.json').read_text())
p=ROOT/'tools/plot_tiny_hillclimb.py';s=p.read_text()
assert 'def draw_repaired_rl_wall' not in s
shutil.copy2(ROOT/'examples/tiny/hillclimb.png','/tmp/tiny-prl-pre238.png')
for old,new in [('1-236','1-238'),('all 236 experiments','all 238 experiments'),('== 236','== 238'),('range(1, 237)','range(1, 239)'),('ax.set_xlim(0, 236)','ax.set_xlim(0, 238)'),('150, 175, 236]','150, 175, 238]')]:s=s.replace(old,new)
s=s.replace('PAGE_H = 62.2','PAGE_H = 69.0')
outcome='V' if r['verified'] else 'D'
s=s.replace('LANES[-1][2].append(236)',f'LANES[-1][2].append(236)\nCODES["A"].append(237)\nCODES["{outcome}"].append(238)\nLANES[-1][2].append(237)\nnext(exps for key, _, exps in LANES if key == "R").append(238)')
wall={x['run_id']:x['elapsed_seconds'] for x in r['runs']}
code='\n\nWALL_238 = '+repr(wall)+'''


def draw_batch_stability(pg):
    ax = pg.axes(2.7, 62.8, 6.2, 1.05)
    for y, before, total in [(1, 3, 16), (0, 2, 4)]:
        ax.hlines(y, 0, before, color=PALE, linewidth=1)
        ax.scatter([before], [y], color=INK, s=18)
        ax.scatter([0], [y], color=ACCENT, s=18)
        ax.annotate(f"{before}/{total} before", (before, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=6.5, fontfamily=pg.mono, color=INK)
        ax.annotate(f"0/{total} after", (0, y), xytext=(8, 8), textcoords="offset points",
                    va="center", fontsize=6.5, fontfamily=pg.mono, color=ACCENT)
    ax.set_yticks([1, 0], ["Native batch shape", "Mixed decode route"])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlim(-0.15, 4.4)
    ax.set_ylim(-0.45, 1.5)
    ax.set_xlabel("requests whose generated tokens differ from singleton execution", fontsize=6.5, fontfamily=pg.prose, color=SUB)
    clean_axes(ax)
    pg.text(GUTTER, 64.22, "237 · Invariant native operations fix all 80 fixed-weight serving requests. Mixed repair makes all 8 staggered outputs/logprobs exact.", size=6.7, color=SUB)
    pg.text(GUTTER, 64.39, "First divergent native gate/up projection: identical input row, 2,028/5,120 BF16 output words differ at 80 versus 160 prefill tokens.", size=6.7, color=SUB)
    pg.text(GUTTER, 64.56, "Opt-in only; FP32 training retained. Mixed steps recompute decode rows after native forward. Correctness result, no isolated speed claim.", size=6.7, color=SUB)


def draw_repaired_rl_wall(pg):
    order = ["c1", "e1", "e2", "c2", "c3", "e3", "e4", "c4", "c5", "e5"]
    labels = ["A1", "B1", "B2", "A2", "A3", "B3", "B4", "A4", "A5", "B5"]
    ax = pg.axes(0.95, 65.65, 7.95, 1.45)
    for early, color, label in ((False, INK, "A · serial startup"), (True, ACCENT, "B · early startup")):
        xs = [i for i,n in enumerate(order) if n.startswith("e") == early]
        ys = [WALL_238[order[i]] for i in xs]
        ax.plot(xs, ys, color=color, marker="o", markersize=3.5, linewidth=1, label=label)
        for x, value in zip(xs, ys):
            ax.annotate(f"{value:.1f}", (x,value), xytext=(0,-12 if early else 6),
                        textcoords="offset points", ha="center", fontsize=6.2, fontfamily=pg.mono, color=color)
    ax.set_xticks(range(10),labels)
    ax.set_xlim(-0.4,9.4)
    ax.set_ylim(45,76)
    ax.set_ylabel("whole RL process · seconds", fontsize=6.5, fontfamily=pg.prose, color=SUB)
    ax.set_xlabel("chronological runs · A-B-B-A, A-B-B-A, A-B", fontsize=6.6, fontfamily=pg.prose, color=SUB)
    clean_axes(ax)
    ax.legend(frameon=False,prop={"family":pg.prose,"size":6.5},loc="upper right")
'''
t=r['timing']
notes=[
 f"238 · Medians {t['control_median_seconds']:.2f} to {t['early_median_seconds']:.2f} s ({t['change_pct']:.1f}%), N=5/arm. Strict blocks: "+'/'.join('pass' if x['both_beat_both'] else 'miss' for x in t['strict_blocks'])+'.',
 'Exact finalized work: 166 rollouts / 18,138 tokens; both steps match loss, gradient norm and KL. Common-request token hashes match exactly.',
 f"Shutdown tail included: {r['request_count_range'][0]}-{r['request_count_range'][1]} requests, {r['generated_token_range'][0]:,}-{r['generated_token_range'][1]:,} generated tokens. All guards pass; zero job swap.",
 'Scope: 2 updates, B2-capable 360M, full 4.53 GB checkpoint. Both arms use invariant serving and mixed repair; only startup differs.',
 'Limits: some discarded zero-reward traces retain raw logprob variation; no universal bitwise or long-run speed claim.'
]
for i,note in enumerate(notes):code+=f'    pg.text(GUTTER, {67.45+i*.17:.2f}, {note!r}, size=6.7, color=SUB)\n'
s=s.replace('\ndef build(output:',code+'\n\ndef build(output:')
s=s.replace('    draw_footer(pg, 61.65)','''    pg.section(62.0, "17", "Stabilize serving before timing RL", "Exp 237 · fixed weights and request seeds; no training or throughput inference from these probes.")
    draw_batch_stability(pg)
    pg.section(64.95, "18", "Checkpointed two-sequence RL with matched training work", "Exp 238 · accepted kernel plus invariant native operations and mixed repair in both arms; early startup comparison.")
    draw_repaired_rl_wall(pg)
    draw_footer(pg, 68.4)''')
p.write_text(s)
