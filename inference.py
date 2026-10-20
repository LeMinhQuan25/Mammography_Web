import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from model_v26 import ClinicalSparseBRAIxProtoPNet, set_config, _robust_lesion_topology_input, _finite_persistence_intervals, _intervals_to_persistence_image

MEAN=np.array([.485,.456,.406],np.float32)[:,None,None]
STD=np.array([.229,.224,.225],np.float32)[:,None,None]

def load_model(path, device='cpu'):
    # Checkpoints should only be loaded if trusted. Prefer weights_only=True.
    ck=torch.load(path,map_location='cpu',weights_only=True)
    cfg=ck['config'];set_config(cfg)
    state=ck['model_state_dict']
    n=int(state['prototype_vectors'].shape[0]); p=int(state['prototype_vectors'].shape[1])
    model=ClinicalSparseBRAIxProtoPNet(n,p,state['prototype_concept_ids'].long(),pretrained=False,temperature=float(cfg['temperature']))
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    return model,ck

def read_npz(path):
    with np.load(path,allow_pickle=False) as p:
        img=np.asarray(p['image'],dtype=np.uint8)
        mask=np.asarray(p['mask'],dtype=np.uint8) if 'mask' in p else None
    if img.shape!=(1536,768): raise ValueError('NPZ không đúng kích thước train (1536,768)')
    return img,mask

def preprocess_dicom(fileobj):
    # Compatibility renderer; pixel-intensity equivalence to training must be tested.
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    ds=pydicom.dcmread(fileobj)
    a=ds.pixel_array
    if a.ndim!=2: raise ValueError('Cần một ảnh mammogram 2D')
    try: a=apply_voi_lut(a,ds).astype(np.float32)
    except Exception: a=a.astype(np.float32)
    low,high=np.percentile(a,[0.1,99.9]); a=np.clip((a-low)/(high-low+1e-7),0,1)
    if str(getattr(ds,'PhotometricInterpretation',''))=='MONOCHROME1': a=1-a
    a=(a*255).astype(np.uint8)
    h,w=a.shape; scale=min(1536/h,768/w)
    hh,ww=max(1,round(h*scale)),max(1,round(w*scale))
    res=cv2.resize(a,(ww,hh),interpolation=cv2.INTER_AREA)
    pad=np.zeros((1536,768),dtype=np.uint8); yy=(1536-hh)//2;xx=(768-ww)//2
    pad[yy:yy+hh,xx:xx+ww]=res
    return pad, {'original_hw':[h,w],'scaled_hw':[hh,ww],'padding_tl':[yy,xx],'scale':scale}

def roi_topology(img,mask):
    # EXACT helpers from supplied source; uses GUDHI Cubical PH.
    if mask is None or not np.any(mask): return None
    from model_v26 import gd
    if gd is None:
        raise RuntimeError("Thiếu GUDHI. Cài đặt bằng: python -m pip install gudhi")
    z,valid=_robust_lesion_topology_input(img,mask,96)
    if not valid: return None
    h0=_intervals_to_persistence_image(_finite_persistence_intervals(z,0))
    h1=_intervals_to_persistence_image(_finite_persistence_intervals(z,1))
    return (torch.from_numpy(h0)[None,None],torch.from_numpy(h1)[None,None])

def run_prediction(model,img,mask,device='cpu'):
    img=np.asarray(img,dtype=np.uint8)
    a=np.repeat((img.astype(np.float32)/255)[None],3,axis=0)
    x=torch.from_numpy(((a-MEAN)/STD)[None]).to(device)
    topo=roi_topology(img,mask) if mask is not None else None
    with torch.inference_mode():
        if topo: o=model(x,topo[0].to(device),topo[1].to(device))
        else: o=model(x)
        pp=F.softmax(o['proto_logits'],1)[0].cpu().numpy()
        gp=F.softmax(o['global_logits'],1)[0].cpu().numpy()
        ens=(pp+gp)/2
        presence=o['presence'][0].cpu().numpy()
        weights=model.last_layer.weight.detach().cpu().numpy()
        proto_pred=int(pp.argmax())
        signed=presence*(weights[proto_pred]-weights[1-proto_pred])
        sim=o['similarity_maps'][0].cpu().numpy()
        active=model.prototype_gate.detach().cpu().numpy()>0.5
    return {'proto':pp,'global':gp,'ensemble':ens,'signed':signed,'sim':sim,'active':active,'proto_pred':proto_pred,'topology_mode':'ROI-based' if topo else 'NO-ROI (not validated)'}

def heatmap(img,sim):
    s=cv2.resize(sim.astype(np.float32),(img.shape[1],img.shape[0]))
    s=(s-s.min())/(np.ptp(s)+1e-8)
    color=cv2.applyColorMap((s*255).astype(np.uint8),cv2.COLORMAP_JET)
    rgb=cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
    return (0.68*np.stack([img]*3,axis=-1)+0.32*rgb).clip(0,255).astype(np.uint8)

def reference_patch(img,source):
    # Same center formula as training explanation; patch is an approximate visual crop.
    h,w=img.shape; yf=int(source['h']);xf=int(source['w']); hf=int(source['Hf']);wf=int(source['Wf'])
    cy=round((yf+.5)*h/hf);cx=round((xf+.5)*w/wf)
    yy=max(16,int(h*.06));xx=max(16,int(w*.06))
    return img[max(0,cy-yy):min(h,cy+yy),max(0,cx-xx):min(w,cx+xx)]
