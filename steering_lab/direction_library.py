"""Model-bound, precomputed directions; never silently rebuild with a different method."""
import json
from pathlib import Path
import re
import numpy as np

DIRECTORY=Path(__file__).resolve().parent/'custom_directions'

def registered_direction(axis,atlas,protocol_sha256):
    if axis.get('method')!='response_mean':return None
    if not re.fullmatch(r'custom-[0-9a-f]{12}',axis['id']):raise ValueError('Invalid registered direction ID')
    try:payload=json.loads((DIRECTORY/(axis['id']+'.json')).read_text())
    except (OSError,ValueError) as error:raise ValueError('回答状态方向资产尚未安装，无法用其他提取方法替代。') from error
    if payload.get('axis')!=axis:raise ValueError('Registered direction instructions do not match')
    provenance=payload.get('provenance',{})
    if provenance.get('protocol_sha256')!=protocol_sha256:raise ValueError('Registered direction geometry mismatch')
    model=provenance.get('model',{})
    if any(model.get(key)!=atlas['model'][key] for key in ['model','revision','layer_index','hidden_size']):
        raise ValueError('Registered direction model mismatch')
    vector=np.array(payload.get('native_direction',[]),dtype=float)
    if vector.shape!=(atlas['model']['hidden_size'],) or not np.isfinite(vector).all() or not np.linalg.norm(vector)>0:
        raise ValueError('Invalid registered direction vector')
    if not np.isclose(np.linalg.norm(vector),atlas['scale'],rtol=1e-5):raise ValueError('Registered direction must represent 1x')
    if payload.get('positions')!=axis['positions']:raise ValueError('Registered direction injection positions mismatch')
    return payload

def available_directions(atlas,protocol_sha256):
    result=[]
    for path in sorted(DIRECTORY.glob('custom-*.json')):
        axis=json.loads(path.read_text())['axis']
        payload=registered_direction(axis,atlas,protocol_sha256)
        if payload is not None:result.append(payload)
    return result
