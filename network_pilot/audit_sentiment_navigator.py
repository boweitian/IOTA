"""Independently reconstruct frozen-policy predictions and exact-match holdout metrics."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from sentiment_prepare import DEFAULT,sha,write
from sentiment_models import predict

def read(path):return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]
def wilson(correct,n):
    z=1.959963984540054;p=correct/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-radius,center+radius]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args();out=args.out
    summary=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'dataset-manifest.json').read_text())
    frozen=json.loads((out/'frozen-policies.json').read_text());spec=json.loads((out/'experiment-spec.json').read_text())
    training=read(out/'train.jsonl');testing=read(out/'test-inputs.jsonl');gold=read(out/'test-labels.jsonl');records=read(out/'test-predictions.jsonl')
    assert len(training)==70 and len(testing)==len(gold)==len(records)==30
    assert not {r['id'] for r in training}&{r['id'] for r in testing}
    assert not {r['text'].lower() for r in training}&{r['text'].lower() for r in testing}
    assert np.bincount([r['label'] for r in training],minlength=5).tolist()==[14]*5
    assert np.bincount([r['label'] for r in gold],minlength=5).tolist()==[6]*5
    assert sha(out/'frozen-policies.json')==summary['policies_sha256']
    assert sha(out/'test-predictions.jsonl')==summary['predictions_sha256']
    assert sha(out/'search-geometry.npz')==summary['geometry_sha256']==manifest['files']['search-geometry.npz']
    assert summary['model']['model']=='Qwen/Qwen3.5-9B' and summary['model']['layer_index']==16
    for name,value in spec['source_sha256'].items():assert sha(Path(__file__).resolve().parent/name)==value,name
    selected=min(frozen['cv'],key=lambda m:(-m['accuracy'],m['configured_members'],m['nll'],m['name']))['name']
    assert selected==summary['selected']==frozen['selected']
    p=np.full((30,len(summary['actions']),5),.2);x=None
    for j in summary['test_agents_measured']:
        measured=np.load(out/'test-cache'/f'agent-{j:02d}.npz')
        assert str(measured['policies_sha256'])==summary['policies_sha256']
        p[:,j]=measured['probabilities']
        if j==0:x=measured['features']
    labels={row['id']:row['label'] for row in gold};y=np.array([labels[row['id']] for row in records])
    for name,policy in frozen['policies'].items():
        reconstructed=predict(policy,p,x)
        saved=np.array([row['policies'][name]['probabilities'] for row in records])
        np.testing.assert_allclose(reconstructed,saved,rtol=1e-12,atol=1e-12)
        assert int(np.sum(saved.argmax(1)==y))==summary['test_metrics'][name]['correct']
    report=dict(passed=True,checks=['100 disjoint sentence roots, class balanced 70/30','same 9B revision/layer and geometry file',
        'source hashes match frozen experiment','selected policy determined by training OOF only','frozen policy hash tied to all holdout caches',
        'all saved predictions independently reconstructed','all exact-match test accuracies independently recalculated'],
        test_count=30,selected=selected,policies_sha256=summary['policies_sha256'],
        caveat='Integrity hashing reads raw source bytes; holdout text and labels are not parsed or used by training/selection. Public pretraining contamination cannot be excluded.')
    write(out/'audit.json',report)
    base=summary['test_metrics']['baseline'];best=summary['test_metrics'][selected]
    lines=['# IOTA Navigator · SST-5 / Qwen3.5-9B','','Search 已构建的 geometry 保持冻结；Navigator 只学习如何组织方向配置、agent 群体及输入路由。',
        '',f"模型：{summary['model']['model']}；revision：{summary['model']['revision']}；layer index 16；FP32，TF32 关闭。",
        f"Geometry SHA256：`{summary['geometry_sha256']}`。九方向来自 Search，未用 benchmark 标签重建。",'',
        '斯坦福官方 SST-5 原始整句根节点；每类 14 训练、6 测试，共 70/30。不是全量 SST-5 标准成绩。','',
        '## 预先固定的评估规则','','70 个训练样本内做五折分层 OOF 验证，准确率优先，同分选更少 agent，再比较 NLL。之后在全部 70 题拟合一次并冻结策略。测试仅测量这些固定方案所需的 agent，不对测试集遍历49个方向以挑选答案。',
        '模型权重保持冻结；训练参数包括成员、概率组合权重和基于九方向输入 activation 投影的 ridge 路由器。',
        '评分是固定提示下 A–E 五个单 token 的类别概率；不是自由生成答案或 LLM judge。','',
        f"训练选择方案：`{selected}`。测试：{best['correct']}/30 = {100*best['accuracy']:.1f}%；原模型：{base['correct']}/30 = {100*base['accuracy']:.1f}%。",'',
        '| 方案 | 训练 OOF | 全训练集拟合 | 测试 | 群体成员数 |', '|---|---:|---:|---:|---:|']
    for cv in summary['cv']:
        name=cv['name'];m=summary['test_metrics'][name]
        lines.append(f"| {name} | {cv['correct']}/70 ({cv['accuracy']:.1%}) | {summary['train_metrics'][name]['accuracy']:.1%} | {m['correct']}/30 ({m['accuracy']:.1%}) | {cv['configured_members']} |")
    interval=wilson(best['correct'],30);baseline_interval=wilson(base['correct'],30)
    guess=np.array([row['policies'][selected]['prediction'] for row in records]);original=np.array([row['policies']['baseline']['prediction'] for row in records])
    gain=int(np.sum((guess==y)&(original!=y)));loss=int(np.sum((guess!=y)&(original==y)))
    lines.extend(['','## 结果解释','',f"相对原模型：新增答对 {gain} 题，同时新增答错 {loss} 题；净变化 {100*(best['accuracy']-base['accuracy']):+.1f} 个百分点。",
        f"所选方案的 95% Wilson 区间为 {interval[0]:.1%}–{interval[1]:.1%}；原模型为 {baseline_interval[0]:.1%}–{baseline_interval[1]:.1%}。每题约 3.33 个百分点。",'',
        '此试验只找到预定义49个配置与有限贪心群体搜索中的训练候选，不是全局最优性能证明。30个测试样本不足以据此宣布稳定提升。统一9B、revision和geometry提高可比性，但不能自动排除小样本误差、过拟合、数据标注噪声或预训练污染。','',
        '路由器只使用当前输入的几何特征。路由群体在成员评分之外需要一次零干预特征读取；若零干预模型本来就是成员，可复用。原模型等成本重复 greedy 预测不会产生新信息。','',
        '## 独立核查','',*['- '+check for check in report['checks']],
        '',f"数值评分检查：{summary['numerical_audit']}。",
        '',f"策略文件 SHA256：`{summary['policies_sha256']}`。",
        '',f"预测文件 SHA256：`{summary['predictions_sha256']}`。",'',
        '来源：https://nlp.stanford.edu/sentiment/ 。数据文件、样本ID、原始源哈希见 dataset-manifest.json。原始单Agent概率和输入投影保存在 train-cache/ 与 test-cache/。旧0.8B证据不参与本试验。'])
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True,selected=selected,selected_correct=best['correct'],baseline_correct=base['correct'],gain=gain,loss=loss),ensure_ascii=False))

if __name__=='__main__':main()
