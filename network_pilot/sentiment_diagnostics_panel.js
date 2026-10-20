(()=>{
const r=D.diagnostics;if(!r)return;$('sentimentDiagnostics').hidden=false;
const c=r.conditions,base=c.original_forward,gen=c.original_generation;
$('diagnosticFormat').textContent='独立递归解析已核对全部 '+r.source_audit.samples+' 题的原始树标签与文本。A–E 的词表总概率最低 '+percent(base.label_mass_quantiles[0])+'；真实生成格式无效 '+gen.invalid_format+' 题，与原分类预测一致 '+gen.agreement_with_original+'/100。';
$('diagnosticGranularity').textContent='原 '+r.original_errors+' 个错误中，'+r.adjacent_class_errors+' 个只差相邻一档。原预测粗化为负面／中性／正面后为 '+r.original_collapsed_three_class_correct+'/100；只看非中性样本，正负极性答对 '+r.original_non_neutral_polarity_correct+'/'+r.non_neutral_count+'。这表明强度边界与中性判定是主要误差来源之一。';
$('diagnosticClasses').textContent='原五类分别答对：'+base.confusion.map((row,i)=>LABELS[i]+' '+row[i]+'/'+row.reduce((a,b)=>a+b,0)).join('；')+'。';
const names={original_forward:'原提示 · 首 token 五标签评分',original_generation:'原提示 · 实际生成并解析',reversed_letters:'反转 A–E 与情感类别映射',semantic_generation:'直接生成英文情感标签',synthetic_calibration:'原提示 + 五条人工合成强度示例',three_class_forward:'单独提示三分类',precision_2_last:'规则精确 +2× · 最后输入 token',precision_2_all:'规则精确 +2× · 全输入位置',warmth_2_all:'情感共鸣 +2× · 全输入位置','precision_+2_last':'规则精确 +2× · 最后输入 token','precision_+2_all':'规则精确 +2× · 全输入位置','warmth_+2_all':'情感共鸣 +2× · 全输入位置','warmth_-2_all':'情感共鸣 −2× · 全输入位置'};
for(const [id,m] of Object.entries(c)){const tr=node('tr');for(const text of [names[id]||id,id==='three_class_forward'?'粗化三分类':'原五分类',m.correct+'/100 · '+percent(m.accuracy),m.train_correct+'/70',m.test_correct+'/30',m.agreement_with_original===undefined?'不同任务':m.agreement_with_original+'/100'])cell(tr,text);$('diagnosticConditions').querySelector('tbody').append(tr)}
const probe=c['precision_+2_last'];
$('diagnosticInjection').textContent='实际 hook 已观测到残差变化：规则精确 +2× 的注入向量范数 '+probe.shift_norm.toFixed(2)+'，最后 token 残差范数中位数 '+probe.residual_norm_quantiles[1].toFixed(2)+'，比例中位数 '+percent(probe.median_shift_to_residual_ratio)+'。上表将最后输入 token 与全输入位置注入分列比较，不能把两种干预范围当作同一个实验。';
$('diagnosticCoverage').textContent='原扫描 50% 不是模型能力上限：仅将几种提示形式的正确样本取并集，就有 '+r.coverage.prompt_variants_coverage+'/100 至少答对一次；加上全输入位置干预对照后为 '+r.coverage.all_five_class_diagnostics_coverage+'/100。这仍是依赖真实标签的事后覆盖，不能当作独立测试准确率。';
const warm=c['warmth_+2_all'],positive=warm.confusion.reduce((n,row)=>n+row[3]+row[4],0);
$('diagnosticWarmth').textContent='干预更广也未带来分类收益：“情感共鸣 +2×”作用于所有输入位置后，'+positive+'/100 被判成正面或很正面，准确率 '+percent(warm.accuracy)+'。它显著改变了输出类别倾向，不能据此说模型更善于判断评论情感。';
})();
