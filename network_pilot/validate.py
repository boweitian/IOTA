"""Default end-to-end generated-answer validation entrypoint."""
import argparse,json,subprocess,sys
from pathlib import Path
from classification_generation import ROOT,DEFAULT,prepare,summarize
from sentiment_prepare import write,sha

def finalize():
    summarize(DEFAULT,require_complete=True)
    from audit_generation import audit
    audit(DEFAULT)
    from build_generation_view import build
    build()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.status:
        outputs=list((DEFAULT/'outputs').glob('action-*.jsonl'));print(json.dumps(dict(completed_actions=len(outputs),total_actions=54,run=str(DEFAULT)),indent=2))
    elif a.finalize:finalize()
    else:
        prepare(DEFAULT)
        if len(list((DEFAULT/'outputs').glob('action-*.jsonl')))==54:finalize()
        else:
            array=subprocess.run(['sbatch','--parsable',str(ROOT/'run.sbatch')],capture_output=True,text=True,check=True).stdout.strip().split(';')[0]
            command=f'cd {ROOT} && {sys.executable} validate.py --finalize'
            final=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+array,'--job-name=iota-analysis-finalize','--partition=beacon','--qos=high','--account=angliece','--cpus-per-task=2','--mem=4G','--time=00:20:00','--output='+str(ROOT/'finalize-%j.log'),'--wrap='+command],capture_output=True,text=True,check=True).stdout.strip().split(';')[0]
            record=dict(array_job=array,finalizer_job=final,entrypoint_sha256=sha(Path(__file__)),run=str(DEFAULT));write(ROOT/'validation-launch.json',record);print(json.dumps(record,indent=2))
