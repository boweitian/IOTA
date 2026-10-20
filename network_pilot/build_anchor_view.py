"""Publish the anchored search and its simple four-field per-sample records."""
import html,json,re,shutil
from pathlib import Path
from classification_generation import DEFAULT
from sentiment_prepare import sha

def build(dist):
    dist=Path(dist);out=DEFAULT/'precision-anchor-v1';summary=json.loads((out/'summary.json').read_text())
    assert summary['complete']
    audit=json.loads((out/'audit.json').read_text());assert audit['passed'] and audit['summary_sha256']==sha(out/'summary.json')
    spec=json.loads((out/'spec.json').read_text());diagnosis=json.loads((out/'error-diagnosis.json').read_text());design=json.loads((out/'design.json').read_text())
    parent_assets=dist/'assets/navigator9b/anli-r3-analysis';assets=parent_assets/'precision-anchor';assets.mkdir(exist_ok=True)
    for name in ['spec.json','summary.json','audit.json','design.json','extraction.json','error-diagnosis.json']:
        shutil.copy2(out/name,assets/name)
    shutil.copytree(out/'outputs',assets/'outputs',dirs_exist_ok=True)
    extension=None;extended_outputs=[];extended_dir=out/'length512-v1'
    if (extended_dir/'summary.json').exists() and (extended_dir/'audit.json').exists():
        extension=json.loads((extended_dir/'summary.json').read_text());ext_audit=json.loads((extended_dir/'audit.json').read_text())
        assert extension['passed'] and ext_audit['passed'] and ext_audit['summary_sha256']==sha(extended_dir/'summary.json')
        shutil.copytree(extended_dir/'outputs',assets/'length512-v1/outputs',dirs_exist_ok=True)
        for name in ['spec.json','summary.json','audit.json']:shutil.copy2(extended_dir/name,assets/'length512-v1'/name)
        extended_outputs=[[json.loads(l) for l in (extended_dir/'outputs'/f'action-{j:02d}.jsonl').read_text().splitlines()] for j in range(24)]
    original=json.loads((DEFAULT/'view-data.json').read_text());oldmix=[json.loads(l) for l in (DEFAULT/'top-three-mixture-v1/outputs.jsonl').read_text().splitlines()]
    view=json.loads(json.dumps(original));base_action=original['actions'][0];best_action=original['actions'][9]
    mix_spec=json.loads((DEFAULT/'top-three-mixture-v1/spec.json').read_text())
    labels=['不干预 baseline（63/100）','规则精确−2× 基座（77/100）','上次前三同等叠加（72/100）']+[a['name'] for a in spec['actions']]
    view['actions']=[base_action,best_action,dict(id='previous_top_three',prompt_template=spec['prompt_template'],requested_norm=mix_spec['regularization']['norm'])]+[dict(a,prompt_template=spec['prompt_template'],requested_norm=a['norm']) for a in spec['actions']]
    if extension:
        labels+=['512上限 · '+a['name'] for a in spec['actions']]
        view['actions']+=[dict(a,prompt_template=spec['prompt_template'],requested_norm=a['norm'],max_new_tokens=512) for a in spec['actions']]
    outputs=[[json.loads(l) for l in (out/'outputs'/f'action-{j:02d}.jsonl').read_text().splitlines()] for j in range(len(spec['actions']))]
    for i,row in enumerate(view['rows']):
        assert all(values[i]['id']==row['id'] for values in outputs)
        row['outputs']=[row['outputs'][0],row['outputs'][9],{k:oldmix[i][k] for k in ['text','prediction','format_valid','error']}]+[{k:values[i][k] for k in ['text','prediction','format_valid','error']} for values in outputs]
        if extension:row['outputs']+=[{k:values[i][k] for k in ['text','prediction','format_valid','error','replayed_for_512']} for values in extended_outputs]
    (assets/'view-data.json').write_text(json.dumps(view,ensure_ascii=False)+'\n')
    sample=(parent_assets/'samples.html').read_text()
    sample=re.sub(r'const NAMES=.*?;const \$',lambda _: 'const NAMES='+json.dumps(labels,ensure_ascii=False)+';const $',sample,count=1)
    sample=sample.replace('../../../capabilities.html','/precision-anchor.html').replace('← 返回 Navigator','← 返回固定基座实验')
    sample=sample.replace("(j===53?'。原则直接加入 prompt。':j===0?'。':'。第17个block最后位置，作用于输入末尾及后续解码。')","(j===0?'。':'。第17个block最后位置，作用于输入末尾及后续解码。')")
    (assets/'samples.html').write_text(sample)
    escape=html.escape;best=summary['results'][str(summary['best_observed'])];train=summary['results'][str(summary['train_selected'])]
    def sample_url(n=1,j=1):return f'assets/navigator9b/anli-r3-analysis/precision-anchor/samples.html?sample={n}&condition={j}'
    rows=''
    groups=[('anchor_replay','基座复跑'),('common_scale','少量加入第二、第三名：共同缩放'),('anchor_preserving','少量加入第二、第三名：固定基座投影')]
    # Native increments have a decomposition field; new and random additions do not.
    for mode,title in groups:
        indices=[j for j,a in enumerate(spec['actions']) if a['mode']==mode and (mode!='anchor_preserving' or 'decomposition' in a)]
        rows+='<tr><th colspan="8">'+title+'</th></tr>'
        for j in indices:rows+=metric_row(j,spec,summary,sample_url)
    for title,axis_filter in [('新候选方向增量',lambda a:a.get('axis') in [n['id'] for n in design['neighbors']]),('随机增量对照',lambda a:a.get('axis','').startswith('random'))]:
        rows+='<tr><th colspan="8">'+title+'</th></tr>'
        for j,a in enumerate(spec['actions']):
            if axis_filter(a):rows+=metric_row(j,spec,summary,sample_url)
    diagnoses=''
    for group in diagnosis['groups']:
        links='、'.join('<a href="'+sample_url(n,1)+'">'+str(n)+'</a>' for n in group['samples'])
        diagnoses+='<tr><td>'+group['name']+'</td><td>'+str(len(group['samples']))+'</td><td>'+links+'</td><td>'+group['hypothesis']+'</td></tr>'
    axes=''
    for axis in design['neighbors']:
        stat=summary['extraction']['statistics'][axis['id']]
        axes+='<tr><td>'+axis['name']+'</td><td>'+escape(axis['positive'])+'</td><td>'+escape(axis['negative'])+'</td><td>'+format(stat['cosine_with_negative_precision'],'.3f')+'</td></tr>'
    from build_generation_view import CSS
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>围绕规则精确−2的局部搜索</title><style>'''+CSS+'''</style><script src="theme.js"></script><link rel="stylesheet" href="theme-blue.css"></head><body><main><a href="index.html#capabilities">← Navigator</a><h1>围绕规则精确 −2× 的局部搜索</h1><p class="note">同一批已查看的100题，适应性探索。基座、新方向假设和候选选择均受这些题的标签影响，train/test只是原分组，不能作为独立泛化成绩。</p><p>原基座77/100；上次前三同等叠加72/100；本轮24个配置×100题。事后最佳：<b>'''+str(best['correct'])+'''/100</b>（'''+escape(best['name'])+'''）。按原train70题选出：'''+escape(train['name'])+'（'+str(train['train_correct'])+'/70；原test '+str(train['test_correct'])+'''/30）。</p><section><h2>各配置的完整结果</h2><p>“共同缩放”沿用原投影上限。“固定基座”将新增向量中与precision平行的分量去掉，保留原−2×投影；必要时只缩小新增增量。两种方法均限制活跃方向投影绝对值不超过2×。这批小幅原方向组合的共同缩放模式也可能恰好保留−2×总投影，差别主要在其他分量。</p><div class="scroll"><table><thead><tr><th>配置</th><th>全部100</th><th>train70</th><th>test30</th><th>修正基座错误</th><th>改坏基座正确</th><th>格式无效</th><th>precision投影</th></tr></thead><tbody>'''+rows+'''</tbody></table></div></section><section><h2>基座剩余23题：可见输出的定性诊断</h2><p>分组是诊断假设，不是对内部机制的证明；不修改官方标签。点击编号查看题目、标准答案、实际prompt和完整输出。</p><div class="scroll"><table><thead><tr><th>候选类型</th><th>题数</th><th>样本</th><th>观察</th></tr></thead><tbody>'''+diagnoses+'''</tbody></table></div></section><section><h2>三个新候选方向如何定义</h2><p>使用8个抽象上下文下的正负指令差提取，归一化到原1×长度。提取不含测试题、答案或题目专用词。先去掉其precision平行分量，再以−0.25、+0.25、+0.5倍作为基座的增量。语义相近不代表隐藏状态向量很接近。</p><div class="scroll"><table><thead><tr><th>方向</th><th>正向指令</th><th>负向指令</th><th>与−precision余弦</th></tr></thead><tbody>'''+axes+'''</tbody></table></div></section><p><a href="assets/navigator9b/anli-r3-analysis/precision-anchor/spec.json">实验协议</a> · <a href="assets/navigator9b/anli-r3-analysis/precision-anchor/summary.json">全部统计</a> · <a href="assets/navigator9b/anli-r3-analysis/precision-anchor/audit.json">独立核算</a></p></main></body></html>'''
    length_section=''
    if extension:
        ext_best=extension['results'][str(extension['best_observed'])]
        length_rows=''.join('<tr><td>'+escape(a['name'])+'</td><td>'+str(summary['results'][str(j)]['correct'])+'</td><td><a href="'+sample_url(1,27+j)+'">'+str(extension['results'][str(j)]['correct'])+'</a></td><td>'+str(extension['results'][str(j)]['invalid_format'])+'</td><td>'+str(extension['results'][str(j)]['vs256_fixed'])+'</td></tr>' for j,a in enumerate(spec['actions']))
        length_section='<section id="lengthControl"><h2>独立对照：将生成上限放宽到512</h2><p>原256上限有76条截断输出，涉及72个原始批次。实际重跑288次生成，每条原始token前缀均严格一致；另外2112条原输出已到EOS，直接复用。此处没有把缓存记录算成新生成。</p><p>512上限下，基座 '+str(extension['anchor_correct'])+'/100；最佳 '+str(ext_best['correct'])+'/100（'+escape(ext_best['name'])+'）。仅和同一上限的基座比较。</p><div class="scroll"><table><thead><tr><th>配置</th><th>256上限正确</th><th>512上限正确</th><th>512仍无效</th><th>延长后新增正确</th></tr></thead><tbody>'+length_rows+'</tbody></table></div></section>'
        page=page.replace('<section><h2>基座剩余23题：',length_section+'<section><h2>基座剩余23题：',1)
    (dist/'precision-anchor.html').write_text(page)
    report=['# 围绕规则精确−2×的局部搜索','',f"基座77/100，本轮24配置、2400次真实生成。事后最佳：{best['name']}，{best['correct']}/100。",'',design['limitation'],'','| 配置 | 全部100 | train70 | test30 | 修正基座 | 改坏基座 | 格式无效 |','|---|---:|---:|---:|---:|---:|---:|']
    for j,a in enumerate(spec['actions']):
        m=summary['results'][str(j)];report.append(f"| {a['name']} | {m['correct']} | {m['train_correct']} | {m['test_correct']} | {m['fixed']} | {m['damaged']} | {m['invalid_format']} |")
    report+=['','原precision方向正指令：严格遵守角色设定和场景中的每一条要求与细节。负指令：对设定和场景要求宽松处理，随意发挥。名称不是经认证的逻辑能力或因果机制。','','23题的定性分组：']
    for group in diagnosis['groups']:report.append(f"- {group['name']}：{group['samples']}。{group['hypothesis']}")
    ext_note=''
    if extension:
        report+=['','## 独立512上限对照','',f"实际新增288次生成，2112条原EOS完成输出复用，所有重跑前缀一致。512上限基座{extension['anchor_correct']}/100，最佳{ext_best['correct']}/100（{ext_best['name']}）。",'','| 配置 | 256正确 | 512正确 | 512格式无效 |','|---|---:|---:|---:|']
        for j,a in enumerate(spec['actions']):
            m=extension['results'][str(j)];report.append(f"| {a['name']} | {summary['results'][str(j)]['correct']} | {m['correct']} | {m['invalid_format']} |")
        ext_note='<p>另有统一512上限对照：基座 '+str(extension['anchor_correct'])+'/100，最佳 '+str(ext_best['correct'])+'/100；新增288次前缀核验生成，已完成EOS的记录复用。</p>'
    (out/'REPORT.md').write_text('\n'.join(report)+'\n');shutil.copy2(out/'REPORT.md',assets/'REPORT.md')
    return '<section id="precisionAnchor"><h2>以规则精确−2×为基座继续改进</h2><p>24个小幅混合、新方向及随机增量配置，2400次真实生成。原256上限基座77/100，本轮事后最佳 '+str(best['correct'])+'/100（'+escape(best['name'])+'）。</p>'+ext_note+'<p class="muted">同100题的适应性探索；不是独立测试成绩。</p><a target="_top" href="precision-anchor.html">查看全部配置、错题诊断和逐题输出 ↗</a></section>'

def metric_row(j,spec,summary,url):
    a=spec['actions'][j];m=summary['results'][str(j)]
    return '<tr><td>'+html.escape(a['name'])+'</td><td><a href="'+url(1,j+3)+'">'+str(m['correct'])+'</a></td>'+''.join('<td>'+str(m[k])+'</td>' for k in ['train_correct','test_correct','fixed','damaged','invalid_format'])+'<td>'+format(a['anchor_projection'],'.3f')+'</td></tr>'

if __name__=='__main__':
    build(DEFAULT.parent.parent.parent/'steering_lab/dist')
