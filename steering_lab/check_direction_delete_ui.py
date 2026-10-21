"""Custom-direction lifecycle regressions with explicit API stubs; no model inference."""
import argparse,os,sys,json,copy,hashlib
from pathlib import Path
# Set PYTHONPATH and PLAYWRIGHT_BROWSERS_PATH externally for your installation.
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',required=True);parser.add_argument('--output',type=Path,default=Path(__file__).parent/'runtime/direction-delete');args=parser.parse_args()
root=args.output;root.mkdir(parents=True,exist_ok=True);base=args.url.rstrip('/')+'/';errors=[];requests=[];checks=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox']);ctx=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
 ctx.route('**/api/status',lambda r:r.fulfill(json={'online':True,'application':'iota-interactive'}));ctx.route('**/api/latest',lambda r:r.fulfill(json={'latest':None}))
 page=ctx.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.goto(base+'scenarios.html',wait_until='networkidle');D=page.evaluate('D');template=copy.deepcopy(D['direction_library'][0]);fixtures=[]
 for i in range(6):
  r=copy.deepcopy(template);r['axis']={'id':'custom-'+f'{i+1:012x}','name':'博爱测试' if i==0 else f'Brush style {i+1}','low':'博爱' if i==0 else 'Plain','high':'关怀' if i==0 else 'Vivid','positive':f'Use kind imagery {i}.','negative':f'Use plain wording {i}.'};r.pop('note',None);r['positions']='last';fixtures.append(r)
 fixtures[-1]=copy.deepcopy(template)
 target=fixtures[0]['axis']['id'];profile=copy.deepcopy(D['presets'][1]['profile']);profile['role']['name']='Mina';profile['coefficients']={c['id']:0 for c in D['atlas']['concepts']};profile['coefficients']['confidence']=.25;profile['coefficients'][target]=.75;profile['custom_axes']=[r['axis'] for r in fixtures]
 binding={k:D['atlas']['model'][k] for k in ['model','revision','layer_index']};binding['protocol_sha256']=D['protocol_sha256']
 messages=[{'role':'user','content':'Keep original: 博爱'},{'role':'assistant','content':'Original reply: 博爱','profile':copy.deepcopy(profile),'agent_name':'Mina'}]
 recipe={'format':'iota.character.recipe.v1','profile':copy.deepcopy(profile),'binding':binding,'blocks':{'format':'iota.blocks.v1','order':['confidence',target]}}
 seed={'iota-studio-custom-axes':fixtures,'iota-agent-state-v1':{'profile':profile,'binding':binding,'messages':messages},'iota-studio-recipes':[recipe],'iota-agent-name-xiaohei-v1':True,'iota-retired-directions-20261018-v1':[],'iota-intervention-blocks-v1:'+D['protocol_sha256']:{'order':['confidence',target],'confirmations':{},'tested':{}}}
 page.evaluate('(seed)=>{localStorage.clear();for(const [k,v] of Object.entries(seed))localStorage.setItem(k,JSON.stringify(v))}',seed);page.reload(wait_until='networkidle')
 page.locator('#agentInnerTab').click();page.locator('#customAxes').evaluate('(e)=>e.open=true');page.wait_for_timeout(50)
 assert page.locator('[data-delete-direction]').count()==6
 assert page.locator('[data-delete-direction="'+target+'"]').get_attribute('aria-label')=='Delete direction: 博爱测试'
 box=page.locator('[data-delete-direction="'+target+'"]').locator('..');assert '博Love' not in box.inner_text();assert '博爱' in box.inner_text()
 assert page.locator('.intervention-block[data-direction="'+target+'"] strong').inner_text()=='博爱测试'
 checks.append('six existing custom directions have delete buttons; user labels preserved')
 page.evaluate('setBusy(true)');assert page.locator('[data-delete-direction]:disabled').count()==6;page.evaluate('setBusy(false)');checks.append('deletion disabled while generation is busy')
 page.locator('[data-delete-direction="'+target+'"]').click();page.wait_for_timeout(40)
 assert page.locator('[data-delete-direction]').count()==5
 assert page.locator('.intervention-block[data-direction="'+target+'"]').count()==0
 assert page.evaluate('(id)=>!C.some(c=>c.id===id)&&!(id in profile.coefficients)&&!profile.custom_axes.some(a=>a.id===id)',target)
 assert page.evaluate('profile.coefficients.confidence')==.25
 assert page.locator('.chat-text').all_inner_texts()==[m['content'] for m in messages]
 assert page.evaluate('(id)=>!JSON.parse(localStorage.getItem("iota-studio-custom-axes")).some(r=>r.axis.id===id)',target)
 checks.append('delete removes library entry, coefficient and block; preserves other axes and chat')
 page.locator('#directionDeleteNotice button').click();page.wait_for_timeout(30)
 assert page.locator('[data-delete-direction]').count()==6 and page.evaluate('(id)=>profile.coefficients[id]',target)==.75
 assert page.locator('.intervention-block[data-direction="'+target+'"]').count()==1;checks.append('Undo restores direction, strength and active block')
 page.locator('[data-delete-direction="'+target+'"]').click();page.reload(wait_until='networkidle');page.locator('#agentInnerTab').click();page.locator('#customAxes').evaluate('(e)=>e.open=true')
 assert page.locator('[data-delete-direction]').count()==5;assert page.evaluate('(id)=>!C.some(c=>c.id===id)',target)
 page.locator('#shelf .shelfitem button').first.click();page.wait_for_timeout(30);assert 'Recipe loaded.' in page.locator('#saveStatus').inner_text();assert page.evaluate('(id)=>!C.some(c=>c.id===id)&&!(id in profile.coefficients)',target)
 page.locator('#profileFile').set_input_files({'name':'old-recipe.json','mimeType':'application/json','buffer':json.dumps(recipe).encode()});page.wait_for_timeout(70);assert 'Imported.' in page.locator('#saveStatus').inner_text();assert page.evaluate('(id)=>!C.some(c=>c.id===id)&&!(id in profile.coefficients)',target)
 checks.append('refresh, saved recipe load and old recipe import do not resurrect deleted directions')
 positive='Reply with warmth and respect.';negative='Reply in a plain neutral tone.';newaxis={'id':'custom-'+hashlib.sha256((positive+'\n'+negative).encode()).hexdigest()[:12],'name':'New style','positive':positive,'negative':negative,'low':negative[:15]+'…','high':positive[:15]+'…'}
 newpayload=copy.deepcopy(template);newpayload['axis']=newaxis;newpayload['positions']='last';newpayload.pop('note',None)
 def build(route):requests.append(route.request.post_data_json);route.fulfill(json=newpayload)
 ctx.route('**/iota/api/studio/direction',build)
 page.locator('#customName').fill('New style');page.locator('#customPositive').fill(positive);page.locator('#customNegative').fill(negative);page.locator('#buildDirection').click();page.wait_for_function('C.filter(c=>c.custom&&!c.source).length===6')
 assert page.locator('[data-delete-direction]').count()==6;assert len(requests)==1;checks.append('deletion frees a slot and a new direction can be built')
 page.locator('[data-delete-direction="'+newaxis['id']+'"]').click();page.locator('#buildDirection').click();page.wait_for_function('(id)=>C.some(c=>c.id===id)',arg=newaxis['id']);assert len(requests)==2;checks.append('intentional rebuild of an identical deleted direction is supported')
 assert page.evaluate('IOTADirections.remove("confidence")') is False;assert page.evaluate('(id)=>IOTADirections.remove(id)',D['faith_preset']['axis']['id']) is False;checks.append('built-in directions and the identity preset are protected')
 page.evaluate('() => {window.savedSetItem=Storage.prototype.setItem;Storage.prototype.setItem=function(k,v){if(k==="iota-retired-directions-20261018-v1")throw new Error("storage full");return window.savedSetItem.call(this,k,v)}}')
 assert page.evaluate('(id)=>IOTADirections.remove(id)',newaxis['id']) is False;assert page.evaluate('(id)=>C.some(c=>c.id===id)',newaxis['id']);page.evaluate('() => {Storage.prototype.setItem=window.savedSetItem}');checks.append('storage failure does not silently delete a direction')
 prefab=template['axis']['id'];page.evaluate('(id)=>IOTADirections.remove(id)',prefab);page.reload(wait_until='networkidle');assert page.evaluate('(id)=>!C.some(c=>c.id===id)',prefab);page.locator('#agentInnerTab').click();page.locator('#customAxes').evaluate('(e)=>e.open=true');checks.append('deleted prepackaged custom directions do not reappear from the bundled library')
 page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(root/'delete-directions-mobile.png'),full_page=True);assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1');checks.append('mobile layout fits the viewport')
 assert not errors,errors
 report={'passed':True,'base':base,'checks':checks,'console_errors':errors,'mock_direction_requests':len(requests)};(root/'delete-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False));browser.close()
