"""Dataset-free regression: Z requires ONNX handshape AND index motion."""
import copy,json
import numpy as np
from ai.paths import ROOT
from api.main import AttemptRequest
from ai.motion import recognize_motion
from ai.onnx_alphabet import OnnxAlphabetClassifier
p=json.loads((ROOT/'integration/z_example_request.json').read_text());checks=[]
class CountingCNN:
 def __init__(self,forced=None):self.base=OnnxAlphabetClassifier();self.labels=self.base.labels;self.calls=0;self.forced=forced
 def features(self,*a):return self.base.features(*a)
 def predict_proba(self,x):
  self.calls+=1
  if self.forced:
   out=np.zeros((len(x),26));out[:,self.labels.index(self.forced)]=1;return out
  return self.base.predict_proba(x)
def run(payload,model):return recognize_motion(AttemptRequest(**payload),model,.5)
def check(name,ok):
 assert ok,name
 checks.append(name)
model=CountingCNN();out=run(p,model)
check('actual ONNX called for Z',model.calls==1 and 'onnx_frame_summary' in out['motion_details'])
check('Z accepted only with real shape and path',out.get('predicted_label')=='Z' and out['motion_details']['shape_gate_passed']['Z'])
check('no fabricated skill confidence',out.get('score') is None)
bad=CountingCNN('A');out=run(p,bad)
check('good Z path + wrong ONNX shape rejected',out['status']=='uncertain' and out['motion_details']['reason']=='z_handshape_not_confirmed')
q=copy.deepcopy(p)
for f in q['frames']: f['landmarks']=p['frames'][0]['landmarks']
out=run(q,CountingCNN('D'))
check('good ONNX shape + stationary finger rejected',out['status']=='uncertain')
q=copy.deepcopy(p)
for f in q['frames']:f['landmarks'][8][1]=p['frames'][0]['landmarks'][8][1]
check('horizontal wiggle rejected',run(q,CountingCNN('D'))['status']=='uncertain')
q=copy.deepcopy(p);rng=np.random.default_rng(17)
for f in q['frames']:f['landmarks'][8][0]+=float(rng.normal(0,.0007));f['landmarks'][8][1]+=float(rng.normal(0,.0007))
check('small index noise tolerated with shape gate',run(q,CountingCNN('D')).get('predicted_label')=='Z')
q=copy.deepcopy(p);q['frames'][20]['landmarks']=None
model=CountingCNN();out=run(q,model)
check('tracking loss rejected before classifier',out['status']=='uncertain' and model.calls==0)
q=copy.deepcopy(p);q['target_label']='J';out=run(q,CountingCNN('D'))
check('target never forces label',out.get('predicted_label')=='Z' and out['matches_target'] is False)
report=dict(passed=len(checks),checks=checks,synthetic_only=True,human_accuracy_measured=False,version='motion-v4')
(ROOT/'reports').mkdir(exist_ok=True)
(ROOT/'reports/z_index_checks.json').write_text(json.dumps(report,indent=2));print(report)
