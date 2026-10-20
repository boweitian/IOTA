"""Mount the direction-aligned router diagnostic within the existing Navigator."""
from pathlib import Path

START='<!-- IOTA_ROUTER_ATLAS_START -->'
END='<!-- IOTA_ROUTER_ATLAS_END -->'


def mount_router_atlas(dist):
    dist=Path(dist);path=dist/'capabilities.html'
    if not path.exists() or not (dist/'router-analysis.html').exists():return
    page=path.read_text()
    if START in page:
        begin=page.index(START);finish=page.index(END,begin)+len(END)
        page=page[:begin]+page[finish:]
    block=START+'''<section id="routerAtlas" class="card" style="margin:18px 0 26px;padding:0;overflow:hidden">
<div class="card-head" style="padding:15px 20px;border-bottom:1px solid #dddeda"><div><h2>Router · 逐题最佳方向与正确／错误分布</h2><div class="subtitle">真实题目状态 · 与训练最佳方向精确对齐 · 可逐题查询</div></div><a href="router-analysis.html" target="_blank" style="font-size:11px">独立打开分析页 ↗</a></div>
<iframe id="routerAnalysisFrame" title="Router 逐题方向与对齐 PCA 分布" src="router-analysis.html?embedded=1&v=20261017-3" style="width:100%;height:1900px;border:0;display:block" loading="eager"></iframe>
<script>addEventListener('message',e=>{const f=document.getElementById('routerAnalysisFrame');if(!f||e.source!==f.contentWindow)return;if(location.protocol!=='file:'&&e.origin!==location.origin)return;if(e.data?.type==='iota:router-height'&&Number.isFinite(e.data.height)&&e.data.height>400&&e.data.height<20000)f.style.height=Math.ceil(e.data.height)+'px'});</script>
</section>'''+END
    anchor='<div class="workspace-head">'
    if anchor not in page:raise ValueError('Navigator mount location not found')
    page=page.replace(anchor,block+'\n'+anchor,1)
    temporary=path.with_suffix('.tmp.html');temporary.write_text(page);temporary.replace(path)
