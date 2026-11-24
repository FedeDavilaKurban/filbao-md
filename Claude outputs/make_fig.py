import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker
r = np.load(sys.argv[1]); out = sys.argv[2]
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.35), gridspec_kw=dict(width_ratios=[1.35, 1]))
cols = {'Full': '0.25', 'Q1': '#d95f02'}
names = {'Full': 'full sample', 'Q1': r'nearest quartile, $d_{\rm fil}<3.7\,h^{-1}$Mpc'}
for lab, dx in (('Full', -0.6), ('Q1', 0.6)):
    s, xi, sm, sig, b2 = r[f'{lab}_s'], r[f'{lab}_xi'], r[f'{lab}_smooth'], r[f'{lab}_sig'], float(r[f'{lab}_b'])**2
    mu, w, h = float(r[f'{lab}_mu']), float(r[f'{lab}_w']), float(r[f'{lab}_h'])
    sf = np.linspace(s[0], s[-1], 400)
    a.errorbar(s + dx, (xi - sm) / b2 * 1e3, yerr=sig / b2 * 1e3, fmt='o', ms=2.5, lw=0.8, capsize=1.2, color=cols[lab])
    a.plot(sf, h * np.exp(-0.5 * ((sf - mu) / w)**2) / b2 * 1e3, color=cols[lab], lw=1.4,
           label=f'{names[lab]}: $w$ = {w:.1f}')
a.axhline(0, color='0.6', lw=0.6)
a.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$'); a.set_ylabel(r'$(\xi_0-\xi_{\rm smooth})/b^2\ [10^{-3}]$')
a.set_title('(a) acoustic peak above the smooth part', fontsize=8)
a.legend(frameon=False, loc='upper left'); a.set_ylim(-1.9, 3.4)
ws = r['ws']; wq, wf = float(r['Q1_w']), float(r['Full_w'])
b.hist(ws, bins=np.arange(2, 25.5, 1.5), color='0.75', edgecolor='white', label=f'{len(ws)} random\ndilutions')
p16, p84 = np.percentile(ws, [16, 84])
b.axvspan(p16, p84, color='0.5', alpha=0.15, lw=0)
b.axvline(wq, color=cols['Q1'], lw=1.6, label='nearest quartile')
b.axvline(wf, color=cols['Full'], lw=1.2, ls='--', label='full sample')
b.set_xlabel(r'peak width $w\ [h^{-1}{\rm Mpc}]$'); b.set_ylabel('number of dilutions')
b.set_title('(b) peak width against random dilutions', fontsize=8)
b.legend(frameon=False, loc='upper right', handlelength=1.5); b.set_xlim(2, 25); b.set_ylim(0, 16); b.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
fig.tight_layout(w_pad=1.2)
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200)
print('16-84:', p16, p84, 'frac<=Q1', np.mean(ws <= wq))
