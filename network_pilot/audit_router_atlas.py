"""Verify published direction assets against native states, vectors and rewards."""
import json
from pathlib import Path

import numpy as np

import build_router_atlas as atlas
import router_direction_scan as scan


def audit():
    results={};total=0
    for suite in scan.SUITES:
        folder=atlas.DIST/'assets/router-atlas'/suite
        data=json.loads((folder/'data.json').read_text())
        native=np.load(folder/'aligned_projection.npz');matrix=np.load(folder/'direction_matrix.npz')
        bank=np.load(folder/'all_direction_axes.npz');rows=data['rows'];actions=data['actions']
        source=scan.SOURCE/suite;p=json.loads((source/'protocol.json').read_text())
        features,_=atlas.state.load_features(source,p)
        train=np.array([r['split']=='train' for r in rows]);h=features['steer_last'].astype(float)
        np.testing.assert_allclose(native['mean'],h[train].mean(0),atol=1e-12)
        np.testing.assert_allclose(native['coordinates'],(h-native['mean'])@native['axes'].T,atol=1e-12)
        assert data['projection']['alignment_cosine']>1-1e-12
        assert np.max(np.abs(native['axes']@native['axes'].T-np.eye(3)))<1e-10
        for j,vector in enumerate(native['steering_vectors']):
            k=int(bank['mapping'][j])
            if k<0:assert np.linalg.norm(vector)==0;continue
            u=bank['axes'][k,0]*bank['signs'][j]
            assert float(u@(vector/np.linalg.norm(vector)))>1-1e-12
            np.testing.assert_allclose(bank['coordinates'][k],(h-bank['mean'])@bank['axes'][k].T,atol=6e-8)
        support=matrix['reward'][train].sum(0)
        for i,row in enumerate(rows):
            valid=np.flatnonzero(matrix['reward'][i]==1).tolist()
            assert valid==row['successful']
            assert row['best']==(valid[int(np.argmax(support[valid]))] if valid else None)
            assert row['observed']==int((matrix['reward'][i]>=0).sum())
            assert row['best_complete']==(row['observed']==109)
        if data['complete']:
            assert (matrix['reward']>=0).all()
            verification=json.loads((scan.DEFAULT/suite/'verification.json').read_text())
            assert verification['passed'] and verification['all_test_outputs_regraded']
        groups={}
        for group in ['train','test','historical','fresh']:
            chosen=[r for r in rows if (r['split']=='train' if group=='train' else
                    r['split']!='train' if group=='test' else r['split']!='train' and r['group']==group)]
            if not chosen:continue
            groups[group]=dict(n=len(chosen),covered=sum(bool(r['successful']) for r in chosen),
                complete=sum(r['best_complete'] for r in chosen),
                correct={method:sum(r['rewards'][r['predictions'][method]]==1 for r in chosen) for method in atlas.METHODS})
        results[suite]=dict(samples=len(rows),observed_cells=int((matrix['reward']>=0).sum()),
            complete=data['complete'],axis=data['projection']['axis_action'],
            axis_variance=data['projection']['variance_fraction'][0],groups=groups)
        total+=len(rows)*109
    output=dict(passed=True,complete=all(r['complete'] for r in results.values()),
        expected_cells=total,suites=results)
    atlas.core.save(scan.DEFAULT/'published_data_audit.json',output)
    lines=['# Navigator 逐题方向与对齐 PCA 分布','',
       '页面：http://127.0.0.1:8768/#capabilities；独立分析：http://127.0.0.1:8768/router-analysis.html','',
       '绑定 Qwen3.5-0.8B，block 15，1024 维；共 2340 题，每题 109 配置（包含不干预）。',
       '训练题沿用原始完整矩阵；912 道测试题新增 99,408 次匹配批次测量，Router 权重与选择保持冻结。','',
       '## 测量状态','',f'完整遍历完成：{output["complete"]}。原生投影、逐题成功集合与已发布矩阵核对：通过。','',
       '| Benchmark / 分组 | 题数 | 成功覆盖 | 原模型 | 最佳固定 | 原文本 Router | 新 Router |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for suite,r in results.items():
        for name,g in r['groups'].items():
            c=g['correct'];coverage=str(g['covered']) if g['complete']==g['n'] else '未测全'
            lines.append(f'| {suite} / {name} | {g["n"]} | {coverage} | {c["baseline"]} | {c["fixed"]} | {c["text"]} | {c["geometry_raw"]} |')
    lines+=['','## 几何定义与解释','',
       '默认横轴为该 benchmark 训练集最佳固定配置的实际有符号单位向量 u，x=(h−训练均值)·u。',
       '移除 u 分量后，仅在训练题上拟合正交 PCA。可将 u 切换为选中题的最佳代表或 Router 方向；全体题目随之使用同一个坐标系重投影。',
       '默认参照仅由训练集决定；切到选中测试题的最佳方向时，参照选择使用了该题评分，是事后诊断，剩余 PCA 的拟合样本仍只有训练题。所有非零配置的对齐 cosine 均核验大于 1−1e−12。界面坐标四舍五入至 7 位小数；NPZ 同时提供原始向量与完整精度坐标基。',
       '页面可切换各轴适配或真实等单位；各轴适配只改变显示比例，刻度及坐标不变。窄条分布意味着沿该方向的自然题目状态方差很小。箭头只是注入向量的投影，显示倍率可调，不能解释成模型生成轨迹。','',
       '二元评分通常存在多个并列最佳方向，全部成功集合保留。代表按训练支持最多的成功配置选取，这一口径会偏向全局常见方向；可切换柱状图到全部成功集合，避免将代表频次误当作唯一最优方向分布。',
       '109 配置全部失败时标记无成功方向；未测完时标记部分证据。测试题完整扫描是事后答案知情诊断，不能作为新的盲测成绩，也未用于重训当前 Router。','',
       '## 验证与文件','',
       'test_router_atlas.py：精确对齐、原生位移投影、正负方向与剂量、并列成功集合、原批次断点续扫。',
       'check_router_atlas_ui.py：三个 benchmark、分组、准确率、方向与结果筛选、逐题选轴、CSV/SVG 下载、手机布局及 Navigator 集成。',
       '各 benchmark 的 verification.json：原始输出重新评分、唯一完整矩阵、批次与 token 长度。',
       'assets/router-atlas/<benchmark>/ 包含 JSON、逐题 CSV、方向矩阵、默认投影、全部方向坐标系，以及训练／测试 PNG 与 SVG。','']
    (scan.DEFAULT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(output,ensure_ascii=False),flush=True)
    return output


if __name__=='__main__':audit()
