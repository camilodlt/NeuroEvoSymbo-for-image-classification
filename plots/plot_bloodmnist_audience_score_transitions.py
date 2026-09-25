from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / 'BloodMNIST/L16_WD01_asinh_R18_128/best_modules_boost_0/asinh/best_modules_program_stats_bestsfalse_false.csv'
OUTDIR = ROOT / 'plots'
OUTDIR.mkdir(parents=True, exist_ok=True)

score_cols = ['mean_cs_unigram_avg_single', 'mean_doctor_unigram_avg_single', 'mean_patient_unigram_avg_single']
keep_cols = ['module_key', 'checkpoint_path', 'rf_importance', 'rf_importance_rank', *score_cols]

with STATS.open(newline='') as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r['rf_importance'] = float(r['rf_importance'])
    r['rf_importance_rank'] = int(r['rf_importance_rank'])
    for c in score_cols:
        r[c] = float(r[c])
rows.sort(key=lambda r: r['rf_importance'], reverse=True)

imp = [r['rf_importance'] for r in rows]
norm = Normalize(vmin=min(imp), vmax=max(imp))
cmap = plt.cm.viridis
xs = [0, 1, 2]
audiences = ['Computer scientist', 'Doctor', 'Patient']

fig, ax = plt.subplots(figsize=(7.2, 4.8))
for r in rows:
    y = [r[c] for c in score_cols]
    q = norm(r['rf_importance'])
    ax.plot(xs, y, marker='o', linewidth=0.6 + 3.4 * q, alpha=0.18 + 0.72 * q, color=cmap(q))

median_scores = []
for c in score_cols:
    vals = sorted(r[c] for r in rows)
    n = len(vals)
    median_scores.append(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))
ax.plot(xs, median_scores, color='black', linewidth=2.5, marker='o', label='Median score')

ax.set_xticks(xs)
ax.set_xticklabels(audiences)
ax.set_ylim(1, 10)
ax.set_ylabel('Program interpretability score')
ax.set_title('BloodMNIST program scores across audiences')
ax.grid(axis='y', alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='lower left')

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Random-forest feature importance')
fig.tight_layout()

base = OUTDIR / 'bloodmnist_program_score_transitions_rf_importance'
fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')

with base.with_suffix('.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keep_cols)
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r[c] for c in keep_cols})

print(base.with_suffix('.pdf'))
print(base.with_suffix('.png'))
print(base.with_suffix('.csv'))
