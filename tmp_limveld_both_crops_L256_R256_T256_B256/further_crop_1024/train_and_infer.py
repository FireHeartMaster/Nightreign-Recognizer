import json, pickle, re, time
from pathlib import Path
import cv2, numpy as np

IMAGE_W=1024; IMAGE_H=1024
PATTERN_RE=re.compile(r'pattern_(\d+)_')

def parse_pid(name):
    m=PATTERN_RE.search(Path(name).name)
    if not m: raise ValueError(name)
    return int(m.group(1))

def list_images(d):
    return sorted(Path(d).glob('*.png'))

def read_gray(p):
    img=cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(p)
    if img.shape[:2] != (IMAGE_H, IMAGE_W): raise ValueError((p,img.shape))
    return img

def hog(gray, resize_to=(256,256)):
    r=cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)
    h=cv2.HOGDescriptor(resize_to,(32,32),(16,16),(16,16),9)
    f=h.compute(r).reshape(-1).astype(np.float32)
    return f/(np.linalg.norm(f)+1e-8)

def patch(gray,x,y,ps=72):
    half=ps//2; left=x-half; top=y-half; right=left+ps; bottom=top+ps
    src_left=max(0,left); src_right=min(gray.shape[1], right); src_top=max(0,top); src_bottom=min(gray.shape[0], bottom)
    out=np.zeros((ps,ps),dtype=gray.dtype)
    dst_left=src_left-left; dst_top=src_top-top
    out[dst_top:dst_top+(src_bottom-src_top), dst_left:dst_left+(src_right-src_left)] = gray[src_top:src_bottom, src_left:src_right]
    return out

def point_to_xy(pt):
    x=int(round(float(pt['x_further_px_bl'])))
    y=int((IMAGE_H-1)-round(float(pt['y_further_px_bl'])))
    return x,y

meta=json.load(open('/mnt/data/limveld_layout_slots_further_crop_bl_strict.json','r'))
layouts={int(l['patternId']): l for l in meta['layouts']}
imgs=list_images('/mnt/data/limveld_final/data')
image_feats=[]; slot_feats=[]; pids=[]
for i,p in enumerate(imgs):
    pid=parse_pid(p.name)
    gray=read_gray(p)
    image_feats.append(hog(gray))
    layout=layouts[pid]
    s=[]
    for pt in layout['slots']:
        x,y=point_to_xy(pt)
        s.append(hog(patch(gray,x,y), resize_to=(48,48)))
    slot_feats.append(np.stack(s,axis=0))
    pids.append(pid)
image_feats=np.stack(image_feats,axis=0)
slot_feats=np.stack(slot_feats,axis=0)
# calibrate on subset
margins=[]
for p in imgs[::max(1,len(imgs)//48)]:
    gray=read_gray(p)
    imgf=hog(gray)
    image_scores=imgf @ image_feats.T
    slot_descs=np.stack([hog(patch(gray,*point_to_xy(pt)), resize_to=(48,48)) for pt in layouts[pids[0]]['slots']],axis=0)
    slot_scores=np.mean(np.sum(slot_descs[None,:,:]*slot_feats,axis=2),axis=1)
    iz=(image_scores-image_scores.mean())/(image_scores.std()+1e-8)
    sz=(slot_scores-slot_scores.mean())/(slot_scores.std()+1e-8)
    hybrid=0.65*iz+0.35*sz
    order=np.argsort(-hybrid)
    margins.append(float(hybrid[order[0]]-hybrid[order[1]]))
threshold=float(np.quantile(np.asarray(margins,dtype=np.float32),0.10))
model={'pattern_ids':np.array(pids,dtype=np.int32),'image_features':image_feats,'slot_features':slot_feats,'layouts':{pid:[point_to_xy(pt) for pt in layouts[pid]['slots']] for pid in pids},'calibration':{'alpha':0.65,'temperature':0.12,'unknown_margin_threshold':threshold},'metadata_summary':{'layouts_count':len(layouts),'slot_count':len(meta['slot_definitions'])}}
with open('/mnt/data/limveld_final/models/layout_model.pkl','wb') as f: pickle.dump(model,f)
# sample inference
sample=imgs[0]
gray=read_gray(sample)
imgf=hog(gray)
image_scores=imgf @ image_feats.T
slot_descs=np.stack([hog(patch(gray,*xy), resize_to=(48,48)) for xy in model['layouts'][int(model['pattern_ids'][0])]],axis=0)
slot_scores=np.mean(np.sum(slot_descs[None,:,:]*slot_feats,axis=2),axis=1)
iz=(image_scores-image_scores.mean())/(image_scores.std()+1e-8)
sz=(slot_scores-slot_scores.mean())/(slot_scores.std()+1e-8)
hybrid=0.65*iz+0.35*sz
logits=hybrid/0.12; logits=logits-logits.max(); probs=np.exp(logits); probs=probs/probs.sum(); order=np.argsort(-hybrid)
top5=[]
for idx in order[:5]: top5.append({'patternId':int(model['pattern_ids'][idx]),'score':float(hybrid[idx]),'probability':float(probs[idx])})
res={'input':sample.name,'best_patternId':int(model['pattern_ids'][order[0]]),'top5':top5,'confidence':float(top5[0]['probability']),'margin_top1_top2':float(hybrid[order[0]]-hybrid[order[1]]),'unknown':bool((hybrid[order[0]]-hybrid[order[1]])<threshold)}
json.dump(res, open('/mnt/data/limveld_final/models/sample_infer.json','w'), indent=2)
json.dump({'num_images':len(imgs),'num_patterns':len(pids),'slot_count':len(meta['slot_definitions']),'unknown_margin_threshold':threshold}, open('/mnt/data/limveld_final/models/training_report.json','w'), indent=2)
print(json.dumps({'saved_model':'/mnt/data/limveld_final/models/layout_model.pkl','sample_best':res['best_patternId'],'model_bytes':Path('/mnt/data/limveld_final/models/layout_model.pkl').stat().st_size}, indent=2))
