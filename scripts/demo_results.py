"""Display existing verification metrics without model/GPU dependencies."""
import json
from pathlib import Path
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
source = root.parent / 'verify' / 'results'
out = root / 'output'
out.mkdir(exist_ok=True)
data = json.loads((source / 'Qwen2_5-7B.json').read_text())
pairs = {k: v['loo_cos'] for k, v in data['A'].items() if isinstance(v, dict) and isinstance(v.get('loo_cos'), list) and v['loo_cos']}
models = {}
for p in sorted(source.glob('*.json')):
    if p.name.startswith('Dfu_'):
        continue
    d = json.loads(p.read_text())
    values = d.get('A', {}).get('honest|untruthful', {}).get('loo_cos', [])
    if isinstance(values, list) and values:
        models[d.get('model', p.stem)] = max(values)
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.3, 1]})
for pair, values in pairs.items():
    axes[0].plot(range(len(values)), values, label=pair.replace('|', ' / '), linewidth=2)
axes[0].set(title='Qwen2.5-7B: cross-context consistency', xlabel='Hidden-state index (0 = embeddings)', ylabel='LOO cosine (context average)')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=.15)
ordered = sorted(models.items(), key=lambda x:x[1])
axes[1].barh([k.split('/')[-1] for k,v in ordered], [v for k,v in ordered], color='#8c9eff')
for i, (_, value) in enumerate(ordered):
    axes[1].text(value+.008,i,f'{value:.3f}',va='center',fontsize=9)
axes[1].set(title='Honest / untruthful: peak across layers', xlabel='Maximum LOO cosine', xlim=(0,1))
fig.suptitle('IOTA | Existing verification results', fontsize=20)
fig.text(.5,.015,'Measured metrics only. No new model inference; concept direction vectors are unavailable.',ha='center',fontsize=10,color='#aaaaaa')
fig.tight_layout(rect=[0,.05,1,.94])
fig.savefig(out/'verification_preview.png', dpi=160)
rows = ''.join(f'<tr><td>{html.escape(k.replace("|"," / "))}</td><td>{max(v):.4f}</td><td>{v.index(max(v))}</td></tr>' for k,v in pairs.items())
page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>IOTA 验证结果</title><style>body{{background:#10121c;color:#e5e7eb;font:16px system-ui;max-width:1300px;margin:40px auto;padding:20px}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #333}}</style><h1>IOTA · Qwen2.5-7B 验证结果</h1><p>数据来源：verify/results/Qwen2_5-7B.json。此次展示复用已有验证数据，没有重新运行模型推理。LOO cosine 衡量跨上下文方向一致性；最大值取自所有层，不代表独立测试表现。</p><img src="verification_preview.png"><h2>概念对</h2><table><tr><th>概念对</th><th>最高 LOO cosine</th><th>Hidden-state index</th></tr>{rows}</table><p>原 demo 的方向向量是随机占位，因此这里仅展示已有实测指标。</p></html>'''
(out/'verification_report.html').write_text(page)
for pair, values in pairs.items():
    print(f'{pair}: peak={max(values):.4f}, index={values.index(max(values))}')
print(f'Report: {out / "verification_report.html"}')
