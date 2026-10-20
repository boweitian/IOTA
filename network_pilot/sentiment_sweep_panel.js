/* Every displayed sweep score derives from measured item/configuration responses. */
(()=>{
const scan=D.sweep;if(!scan)return;
const alphabet=(scan.label_tokens||['A','B','C','D','E']).join('');
$('sweepPending').hidden=true;$('sweepResults').hidden=false;
const palette=['#315f50','#bf7148','#497991','#8c6c91','#899b45','#ae8059','#5366a1','#b66576','#648c7d'];
let split='all',axis='all',histogram=null,activeItem=null;
const singles=scan.actions.map((a,j)=>[a,j]).filter(([a])=>a.coefficients.filter(v=>v!==0).length===1).map(([,j])=>j);
const fmt=v=>(v>=0?'+':'')+(v*100).toFixed(1)+' pp';
function metric(id,v){$(id).textContent=percent(v.accuracy)}
function score(v){return percent(v.accuracy)+' · '+v.correct+'/'+v.n}
function named(j){const a=scan.actions[j];if(j===0)return 'Baseline';const i=a.coefficients.findIndex(v=>v!==0);return singles.includes(j)?nodes[i].name+' '+(a.coefficients[i]>0?'+':'')+a.coefficients[i]+'×':a.id}
function svg(tag,attrs={},text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n}
function label(target,x,y,text,anchor='middle'){target.append(svg('text',{x,y,'text-anchor':anchor},text))}
function tooltip(n,text){n.append(svg('title',{},text));return n}
function drawDose(s){const chart=$('sweepDoseChart');chart.replaceChildren();const W=620,H=310,L=52,R=24,T=22,B=48;
 const show=s.directions.filter(d=>axis==='all'||d.id===axis),values=show.flatMap(d=>d.points.map(p=>p.accuracy));
 const low=Math.max(0,Math.floor((Math.min(...values)-.05)*10)/10),high=Math.min(1,Math.ceil((Math.max(...values)+.05)*10)/10);
 const xx=d=>L+(d+2)/4*(W-L-R),yy=v=>H-B-(v-low)/(high-low)*(H-B-T);
 for(let i=0;i<=4;i++){const v=low+(high-low)*i/4,y=yy(v);chart.append(svg('line',{x1:L,x2:W-R,y1:y,y2:y,class:'grid'}));label(chart,L-8,y+4,percent(v),'end')}
 chart.append(svg('line',{x1:L,x2:W-R,y1:yy(s.baseline.accuracy),y2:yy(s.baseline.accuracy),class:'baseline'}));
 for(const d of [-2,-1,0,1,2])label(chart,xx(d),H-B+24,(d>0?'+':'')+d+'×');
 $('sweepLegend').replaceChildren();
 for(const d of show){const k=nodes.findIndex(n=>n.id===d.id),color=palette[k];chart.append(svg('polyline',{points:d.points.map(p=>xx(p.dose)+','+yy(p.accuracy)).join(' '),fill:'none',stroke:color,'stroke-width':axis==='all'?1.8:3}));
  for(const p of d.points){const dot=tooltip(svg('circle',{cx:xx(p.dose),cy:yy(p.accuracy),r:4,fill:color,tabindex:0}),d.name+' '+p.dose+'×：'+score(p)+'；纠正 '+p.paired_vs_baseline.wins+' / 破坏 '+p.paired_vs_baseline.losses);dot.onclick=()=>{$('sweepAxis').value=d.id;axis=d.id;render()};chart.append(dot)}
  const item=node('span');const dot=node('i');dot.style.background=color;item.append(dot,node('span',d.name));$('sweepLegend').append(item)}
 label(chart,W/2,H-5,'干预倍数 · 纵轴按当前范围放大');
}
function drawSign(s){const chart=$('sweepSignChart');chart.replaceChildren();const rows=[['负向',s.trend.negative.mean_accuracy,'#bf7148'],['零干预',s.baseline.accuracy,'#89958b'],['正向',s.trend.positive.mean_accuracy,'#315f50']];
 for(let i=0;i<=4;i++){const y=255-i/4*210;chart.append(svg('line',{x1:50,x2:490,y1:y,y2:y,class:'grid'}));label(chart,43,y+4,(i*25)+'%','end')}
 rows.forEach(([name,v,color],i)=>{const x=90+i*140,y=255-v*210;chart.append(tooltip(svg('rect',{x,y,width:78,height:v*210,fill:color,rx:3}),name+' '+percent(v)));label(chart,x+39,y-10,percent(v));label(chart,x+39,280,name)})
}
function drawHistogram(s){const chart=$('sweepHistogram');chart.replaceChildren();const bins=s.coverage.single_directions.histogram,max=Math.max(...bins.map(b=>b.samples),1),L=44,R=12,T=30,B=44,W=620,H=300,step=(W-L-R)/bins.length;
 for(let i=0;i<=4;i++){const v=max*i/4,y=H-B-v/max*(H-B-T);chart.append(svg('line',{x1:L,x2:W-R,y1:y,y2:y,class:'grid'}));label(chart,L-7,y+4,v.toFixed(0),'end')}
 for(const b of bins){const x=L+b.correct_configurations*step,h=b.samples/max*(H-B-T),rect=tooltip(svg('rect',{x:x+1,y:H-B-h,width:step-2,height:Math.max(h,b.samples?2:0),fill:histogram===b.correct_configurations?'#bf7148':'#315f50',tabindex:b.samples?0:-1}),b.correct_configurations+' 个配置答对：'+b.samples+' 题');
  const choose=()=>{histogram=b.correct_configurations;$('sweepFilter').value='histogram';renderMatrix();drawHistogram(s)};rect.onclick=choose;rect.onkeydown=e=>{if(e.key==='Enter')choose()};chart.append(rect);if(b.samples)label(chart,x+step/2,H-B-h-6,b.samples);if(b.correct_configurations%4===0)label(chart,x+step/2,H-B+21,b.correct_configurations)}
 label(chart,W/2,H-4,'答对的单方向配置数量 / 36')
}
function showItem(row){activeItem=row.id;const target=$('sweepItemDetail');target.hidden=false;target.replaceChildren(node('h3',row.id+' · 真实类别 '+LABELS[row.gold]+' ('+alphabet[row.gold]+')'),node('p',row.text,'text'));
 target.append(node('p','Baseline：'+LABELS[row.baseline_prediction]+'；可行单方向 '+row.feasible_single.length+'/36；含混合 '+row.feasible.length+'/48。','small'));
 target.append(node('p','可行单方向：'+(row.feasible_single.map(named).join('、')||'无'),'small'));
 const select=node('select');select.setAttribute('aria-label','检查样本的配置概率');[0,...singles,...scan.actions.map((_,j)=>j).filter(j=>j&&!singles.includes(j))].forEach(j=>{const o=node('option',named(j));o.value=j;select.append(o)});
 const detail=node('p',undefined,'small');function update(){const j=Number(select.value),p=row.probabilities[j];detail.textContent=named(j)+' → '+LABELS[row.predictions[j]]+' · '+(row.predictions[j]===row.gold?'正确':'错误')+'；'+alphabet[0]+'–'+alphabet.at(-1)+' 概率：'+p.map(v=>percent(v)).join(' / ')}select.onchange=update;update();target.append(select,detail);
 for(const tr of $('sweepMatrix').querySelectorAll('tbody tr'))tr.classList.toggle('active',tr.dataset.id===row.id)
}
function renderMatrix(){const mode=$('sweepGridMode').value,indices=mode==='single'?singles:scan.actions.map((_,j)=>j).filter(j=>j),head=$('sweepMatrix').querySelector('thead'),body=$('sweepMatrix').querySelector('tbody');head.replaceChildren();body.replaceChildren();
 const groups=node('tr');for(const text of ['样本 / 真实标签','Baseline','可行数']){const th=node('th',text);th.rowSpan=2;groups.append(th)}
 for(const n of nodes){const th=node('th',n.name);th.colSpan=4;groups.append(th)}if(mode==='all'){const th=node('th','预定义混合配置');th.colSpan=12;groups.append(th)}head.append(groups);
 const doses=node('tr');for(const j of indices){const a=scan.actions[j],i=a.coefficients.findIndex(v=>v!==0),v=a.coefficients[i],th=node('th',singles.includes(j)?(v>0?'+':'')+v+'×':a.id.split(':')[1]);th.title=named(j);doses.append(th)}head.append(doses);
 const q=$('sweepQuery').value.toLowerCase(),filter=$('sweepFilter').value;
 const rows=D.sweep_rows.filter(r=>{if(split!=='all'&&r.split!==split)return false;if(q&&!(r.id+' '+r.text).toLowerCase().includes(q))return false;const base=r.baseline_prediction===r.gold,can=r.feasible_single.length>0;
  return filter==='all'||filter==='rescued'&&!base&&can||filter==='unsolved'&&!base&&!can||filter==='base_right'&&base||filter==='base_wrong'&&!base||filter==='histogram'&&(histogram===null||r.feasible_single.length===histogram)});
 for(const row of rows){const tr=node('tr');tr.dataset.id=row.id;tr.tabIndex=0;tr.classList.toggle('active',row.id===activeItem);const sample=node('td',undefined,'sample'),name=row.uid?row.source_split+' · '+row.uid.slice(0,8):row.id,button=node('button',name+' · '+alphabet[row.gold]);button.title=row.id;sample.append(button);tr.append(sample);
  const b=node('td',alphabet[row.baseline_prediction],row.baseline_prediction===row.gold?'good':'bad');tr.append(b,node('td',(mode==='single'?row.feasible_single.length:row.feasible.length)+'/'+indices.length));
  for(const j of indices){const pred=row.predictions[j],td=node('td',alphabet[pred],'trial '+(pred===row.gold?'good':'bad'));td.title=named(j)+' → '+LABELS[pred]+'；真实 '+LABELS[row.gold];tr.append(td)}tr.onclick=()=>showItem(row);tr.onkeydown=e=>{if(e.key==='Enter')showItem(row)};body.append(tr)}
 $('sweepRowCount').textContent=rows.length+' 题 · '+indices.length+' 个干预配置';if(!rows.length){const tr=node('tr'),td=node('td','没有匹配的样本。');td.colSpan=indices.length+3;tr.append(td);body.append(tr)}
}
function render(){const s=scan.splits[split],t=s.trend,c=s.coverage.single_directions,all=s.coverage.all_configurations;
 metric('sweepBase',s.baseline);$('sweepBaseN').textContent=s.baseline.correct+' / '+s.n;
 const ties=singles.filter(j=>s.action_metrics[j].correct===s.best_fixed_single.correct).length;
 metric('sweepBest',s.best_fixed_single);$('sweepBestName').textContent=named(s.best_fixed_single.index)+' 等 '+ties+' 个并列 · '+s.best_fixed_single.correct+'/'+s.n+' · 事后最大值';
 metric('sweepOracle',c);$('sweepOracleN').textContent=c.correct+' / '+s.n+' · 单方向干预，不含 baseline';
 $('sweepUnion').textContent=percent(c.with_baseline_accuracy);$('sweepRescue').textContent='额外可解 '+c.rescued_baseline_errors+' 题 · 总计 '+c.with_baseline_correct+'/'+s.n;
 metric('sweepAllOracle',all);$('sweepAllOracleN').textContent=all.correct+' / '+s.n+' · 最佳固定配置 '+named(s.best_fixed_any.index)+' '+percent(s.best_fixed_any.accuracy);
 metric('sweepTrainBest',s.train_selected_single);$('sweepTrainBestName').textContent=named(s.train_selected_single.index)+' · '+s.train_selected_single.correct+'/'+s.n;
 const delta=t.positive_minus_negative.mean_accuracy_delta;
 $('sweepTrendText').textContent='正向平均 '+percent(t.positive.mean_accuracy)+'，零干预 '+percent(s.baseline.accuracy)+'，负向平均 '+percent(t.negative.mean_accuracy)+'。正向 − 负向：'+fmt(delta)+'。'+(delta===0?'当前网格中正负平均准确率相同。':delta>0?'当前网格中正向平均准确率更高。':'当前网格中负向平均准确率更高。');
 drawDose(s);drawSign(s);drawHistogram(s);
 const table=$('sweepDirections').querySelector('tbody');table.replaceChildren();
 for(const d of s.directions){const tr=node('tr');cell(tr,d.name);for(const p of d.points)cell(tr,score(p));for(const dose of [1,2]){const p=s.signed_pairs.find(p=>p.axis===d.id&&p.dose===dose);cell(tr,fmt(p.comparison.accuracy_delta)+' · '+p.comparison.wins+'胜/'+p.comparison.losses+'负')}tr.onclick=()=>{axis=d.id;$('sweepAxis').value=axis;render()};table.append(tr)}
 const effect=$('sweepEffects').querySelector('tbody');effect.replaceChildren();for(const j of singles){const a=scan.actions[j],k=a.coefficients.findIndex(v=>v!==0);if(axis!=='all'&&nodes[k].id!==axis)continue;const m=s.action_metrics[j],tr=node('tr');for(const text of [named(j),m.correct+'/'+s.n,m.paired_vs_baseline.wins,m.paired_vs_baseline.losses,m.changed_prediction,percent(m.mean_gold_probability)])cell(tr,text);effect.append(tr)}
 $('sweepCoverageNote').textContent='Baseline 答错但可被单方向纠正：'+c.rescued_baseline_errors+' 题；baseline 正确、所有单方向都错：'+c.baseline_correct_but_no_steer+' 题；连 baseline 也无法答对：'+c.never_solved_even_with_baseline+' 题。';
 const pair=$('sweepPairText');pair.replaceChildren();for(const [key,title] of [['positive','正向'],['negative','负向']]){const v=t[key];pair.append(node('p',title+' 18 个配置：高于 baseline '+v.improved_configurations+' 个，持平 '+v.tied_configurations+' 个，低于 '+v.degraded_configurations+' 个。'));pair.append(node('p','跨配置累计纠正 '+v.fix_events+' 次，破坏 '+v.break_events+' 次。'))}
 const d=t.positive_minus_negative;pair.append(node('p','18 对方向与倍数：正向更高 '+d.paired_configurations_positive_better+' 对；负向更高 '+d.paired_configurations_negative_better+' 对；同分 '+d.paired_configurations_tied+' 对。'));
 pair.append(node('p','按每题的正负配置平均正确率比较：正向更好 '+d.samples_positive_better+' 题，负向更好 '+d.samples_negative_better+' 题，持平 '+d.samples_tied+' 题。'));
 $('sweepProbabilityNote').textContent='真实标签的平均概率：正向 '+percent(t.positive.mean_gold_probability)+'，baseline '+percent(s.baseline.mean_gold_probability)+'，负向 '+percent(t.negative.mean_gold_probability)+'。概率变化也可能尚不足以改变最高分标签。';
 renderMatrix();$('sweepItemDetail').hidden=true;activeItem=null;
}
for(const n of nodes){const o=node('option',n.name);o.value=n.id;$('sweepAxis').append(o)}
$('sweepSplit').onchange=()=>{split=$('sweepSplit').value;histogram=null;render()};$('sweepAxis').onchange=()=>{axis=$('sweepAxis').value;render()};
for(const id of ['sweepFilter','sweepGridMode'])$(id).onchange=renderMatrix;$('sweepQuery').oninput=renderMatrix;
render();
})();
