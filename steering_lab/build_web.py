"""Static web copy of dist/ for claude.ai hosting: replay only, no model API, evidence split under 16 MB."""
import argparse
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DIST=ROOT/'dist'
CHUNK=12_000_000
STATIC='<script>window.IOTA_STATIC=true</script>'

def rep(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError(f'expected {count}x: {old[:70]}')
    return text.replace(old,new)

def embedded_bridge(page):
    # Iframes use #embedded: the host may drop query strings on published files.
    page=rep(page,"if(new URLSearchParams(location.search).get('embedded')==='1'){",
        "if(new URLSearchParams(location.search).get('embedded')==='1'||location.hash==='#embedded'){")
    return rep(page,'</head>',STATIC+'</head>')

def workspace():
    s=(DIST/'index.html').read_text()
    style=re.search(r'<style>.*?</style>',s,re.S).group(0)
    description=re.search(r'<meta name="description"[^>]*>',s).group(0)
    body=s[s.index('<body>')+6:s.rindex('</body>')]
    body=rep(body,'data-src="capabilities.html?embedded=1"','data-src="capabilities.html#embedded"')
    body=rep(body,'data-src="scenarios.html?embedded=1"','data-src="scenarios.html#embedded"')
    body=rep(body,",['下载方向资产','assets/steering/IOTA-expanded-steering-assets-v2.zip']","")
    body=rep(body,"if(location.protocol==='file:'){$('statusText').textContent='实测回放';$('runtimeNote').textContent='离线可查看全部已测结果；在线入口可直接生成新的角色情景。';return}",
        "if(location.protocol==='file:'||window.IOTA_STATIC){$('statusText').textContent='实测回放';$('runtimeNote').textContent='网页版回放全部已测结果；生成新的角色情景需要连接模型服务。';return}")
    body=rep(body,"if(location.protocol!=='file:')setInterval(connection,5000);","if(location.protocol!=='file:'&&!window.IOTA_STATIC)setInterval(connection,5000);")
    return '<title>IOTA 概念地图</title>\n'+description+'\n'+style+'\n'+STATIC+'\n'+body

def capabilities(out):
    s=(DIST/'capabilities.html').read_text()
    head='<script id="iotaEvidence" type="application/json">';i=s.index(head)+len(head);j=s.index('</script>',i)
    data=json.loads(s[i:j]);data['model'].pop('model_path',None)
    text=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    raw=text.encode()
    # Split on UTF-8 character boundaries so the pieces join back to the same text.
    names,parts,pos=[],[],0
    while pos<len(raw):
        end=min(pos+CHUNK,len(raw))
        while end<len(raw) and (raw[end]&0xC0)==0x80:end-=1
        parts.append(raw[pos:end]);pos=end
    assert b''.join(parts)==raw
    (out/'data').mkdir(parents=True,exist_ok=True)
    for k,p in enumerate(parts):
        name=f'data/capability-evidence-{k}.txt';(out/name).write_bytes(p);names.append(name)
    main_open=s.index('<script>',j);main_close=s.index('</script>',main_open)
    main=s[main_open+8:main_close]
    main=rep(main,"$('exportConfig').onclick=exportConfig;","$('exportConfig').hidden=true;")
    loader=('<script>(async()=>{const st=document.getElementById(\'status\');try{st.textContent=\'正在载入逐题证据…\';'
        f'const parts=await Promise.all({json.dumps(names)}.map(n=>fetch(n).then(r=>{{if(!r.ok)throw new Error(n);return r.text()}})));'
        'document.getElementById(\'iotaEvidence\').textContent=parts.join(\'\');const x=document.createElement(\'script\');'
        'x.textContent=document.getElementById(\'iotaMain\').textContent;document.body.append(x)}'
        'catch(e){st.textContent=\'证据文件载入失败，请刷新页面。\';st.classList.add(\'pending\')}})();</script>')
    page=s[:i]+'</script>'+s[j+9:main_open]+'<script type="text/plain" id="iotaMain">'+main+'</script>'+loader+s[main_close+9:]
    page=rep(page,'页面可离线打开，交互不会调用模型。','网页版回放已保存的实测结果，不会调用模型。')
    (out/'capabilities.html').write_text(embedded_bridge(page))
    return names

def scenarios(out):
    s=(DIST/'scenarios.html').read_text()
    s=rep(s,'<a href="index.html#capabilities" target="_top">IOTA Navigator ↗</a>','<a href="capabilities.html">IOTA Navigator ↗</a>')
    s=rep(s,'<a href="benchmark.html">原 benchmark 与控制基线</a> · <a href="RESEARCH.md">研究来源</a>','')
    s=rep(s,"async function api(path,body){","async function api(path,body){if(window.IOTA_STATIC)throw new Error('网页版只回放实测结果，不能实时生成。');")
    s=rep(s,"if(location.protocol!=='file:'){checkConnection();setInterval(checkConnection,5000)}",
        "if(location.protocol!=='file:'&&!window.IOTA_STATIC){checkConnection();setInterval(checkConnection,5000)}\n"
        "if(window.IOTA_STATIC){$('runtimeLink').hidden=true;$('exportProfile').hidden=true;$('exportSession').hidden=true;"
        "for(const id of ['runCompare','runWalker','buildDirection']){$(id).disabled=true;$(id).title='网页版只回放实测结果'}set('customStatus','网页版不能构建新方向；连接模型服务后可用。');"
        "set('runtimeMessage','网页版回放已经实测的角色、场景和配置；生成新结果需要连接模型服务。')}")
    # setBusy() re-enables the run buttons; keep them off in the static copy.
    s=rep(s,"function setBusy(value){busy=value;$('runCompare').disabled=value;$('runWalker').disabled=value;",
        "function setBusy(value){busy=value;$('runCompare').disabled=value||!!window.IOTA_STATIC;$('runWalker').disabled=value||!!window.IOTA_STATIC;")
    (out/'scenarios.html').write_text(embedded_bridge(s))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    (out/'index.html').write_text(workspace())
    names=capabilities(out);scenarios(out)
    (out/'assets/steering').mkdir(parents=True,exist_ok=True)
    for src,dst in [('STUDIO_RESULTS.md','STUDIO_RESULTS.md'),('PILOT.md','PILOT.md'),('assets/steering/RESULTS.md','assets/steering/RESULTS.md')]:
        (out/dst).write_bytes((DIST/src).read_bytes())
    sizes={str(f.relative_to(out)):f.stat().st_size for f in sorted(out.rglob('*')) if f.is_file()}
    print(json.dumps(dict(chunks=names,sizes=sizes,total=sum(sizes.values())),indent=1))

if __name__=='__main__':main()
