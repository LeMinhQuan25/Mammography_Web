from pathlib import Path
import json
import numpy as np
import streamlit as st
from PIL import Image
import torch
from inference import load_model,read_npz,preprocess_dicom,run_prediction,heatmap,reference_patch

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Mammography V26 • Research',layout='wide',page_icon='🩻')
st.title('Mammography V26 — Prototype Explainability')
st.caption('Research-only prototype | Không sử dụng kết quả để chẩn đoán bệnh')
@st.cache_resource(show_spinner='Đang nạp mô hình V26...')
def get_model(p,device): return load_model(p,device)
@st.cache_data(show_spinner='Đang tìm ảnh prototype trong local...')
def index_npz(directory):
    d=Path(directory).expanduser()
    if not d.is_dir(): return {}
    return {p.stem:str(p) for p in d.rglob('*.npz')}

with st.sidebar:
    st.header('Cấu hình local')
    ckpath=st.text_input('Checkpoint (.pt)',str(ROOT/'checkpoints'/'final_model.pt'))
    npzdir=st.text_input('Thư mục NPZ local',str(ROOT/'local_npz'))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    st.write('Thiết bị:',device)
    show_roi=st.checkbox('Hiển thị ROI ground truth (nếu có)',False)
    topn=st.slider('Số prototypes mỗi phía',1,8,3)

sources=json.loads((ROOT/'assets'/'prototype_sources.json').read_text(encoding='utf-8'))
paths=index_npz(npzdir)
found=sum(1 for s in sources.values() if s.get('image_key') in paths)
st.info(f'Tìm thấy {len(paths)} NPZ local; nguồn prototype khả dụng: {found}/{len(sources)}. Không cần tải NPZ lên web.')
try:
    model,ck=get_model(ckpath,device)
    st.success(f'Checkpoint đã nạp • {model.num_prototypes} prototypes • {int((model.prototype_gate>0.5).sum())} active')
except Exception as exc:
    st.error(f'Không thể nạp checkpoint: {exc}')
    st.stop()

mode=st.radio('Nguồn dữ liệu',['NPZ local (đúng dữ liệu preprocessing)','DICOM upload (thử nghiệm)'],horizontal=True)
img=mask=None
if mode.startswith('NPZ'):
    if not paths:
        st.warning('Hãy chỉ định thư mục NPZ trong sidebar.');st.stop()
    selection=st.selectbox('Chọn file NPZ',list(paths),index=0)
    img,mask=read_npz(paths[selection])
    st.caption('NPZ đã có ảnh chuẩn hóa và ROI mask. ROI chỉ dùng để tái tạo nhánh topology và làm ground truth so sánh.')
else:
    f=st.file_uploader('Chọn DICOM nhũ ảnh toàn phần (.dcm)',type=['dcm'])
    if f is None: st.stop()
    try: img,geo=preprocess_dicom(f)
    except Exception as exc: st.error(str(exc));st.stop()
    st.warning('DICOM mới chưa được xác nhận preprocessing tương đương V26. Không có ROI mask: nhánh topology dùng chế độ fallback, điểm ensemble KHÔNG được coi là kết quả V26 đã kiểm chứng.')
    st.json(geo)
    maskfile=st.file_uploader('ROI mask tùy chọn (.png; dùng cho nghiên cứu, không phải định vị tự động)',type=['png'])
    if maskfile is not None:
        import cv2
        arr=np.array(Image.open(maskfile).convert('L'))
        mask=(cv2.resize(arr,(768,1536),interpolation=cv2.INTER_NEAREST)>127).astype(np.uint8)
        st.warning('ROI mask được người dùng cung cấp; đây KHÔNG phải tự động phát hiện.')

left,right=st.columns([1,1])
with left:
    st.subheader('Ảnh đầu vào')
    st.image(img,clamp=True)
    if show_roi and mask is not None:
        import cv2
        canvas=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,cnts,-1,(0,255,0),2)
        st.image(canvas,caption='Đường viền xanh: ground truth ROI, không phải vùng AI dự đoán')

if st.button('Phân tích ảnh',type='primary',use_container_width=True):
    try:
        with st.spinner('Đang inference (CPU có thể chậm)...'):
            r=run_prediction(model,img,mask,device)
        st.session_state['result']=r
        st.session_state['result_source']=(mode,selection if mode.startswith('NPZ') else f.name)
    except Exception as exc:
        st.error(f'Lỗi inference: {exc}');st.stop()

r=st.session_state.get('result')
current=(mode,selection if mode.startswith('NPZ') else f.name)
if r is None or st.session_state.get('result_source')!=current:
    st.stop()

with right:
    st.subheader('Dự đoán nghiên cứu')
    if mask is None: st.error('NO-ROI: Kết quả sau đây là thử nghiệm, KHÔNG phải inference V26 được đánh giá trên CBIS.')
    elif mode.startswith('DICOM'): st.warning('DICOM + mask: chưa kiểm thử tương đương pipeline preprocessing training.')
    pred=int(np.argmax(r['ensemble']))
    st.metric('Whole model: '+('MALIGNANT' if pred else 'BENIGN'),f'{r["ensemble"][1]*100:.1f}% P(malignant)')
    st.write(f'Global: {r["global"][1]*100:.1f}% | Prototype branch: {r["proto"][1]*100:.1f}%')
    st.caption('Ensemble = trung bình xác suất 2 nhánh, đúng với code gốc; các điểm chưa được hiệu chuẩn thành nguy cơ lâm sàng.')

st.subheader('Prototype explanation — giải thích riêng nhánh Prototype')
active=np.where(r['active'])[0]; values=r['signed'][active]
support=active[np.argsort(-values)[:topn]]
against=active[np.argsort(values)[:topn]]
for section,ids in [('Ủng hộ quyết định nhánh Prototype',support),('Phản đối quyết định nhánh Prototype',against)]:
    st.markdown('#### '+section)
    for j in ids:
        s=sources.get(str(int(j)),{})
        st.markdown(f'**P{j}** — {s.get("concept","Unknown")} · {s.get("signature", "")} · signed Δlogit={r["signed"][j]:+.4f}')
        a,b,c=st.columns(3)
        a.image(heatmap(img,r['sim'][j]),caption='Similarity heatmap • không phải lesion segmentation',use_container_width=True)
        a.caption('Màu nóng chỉ vùng tương đồng prototype; không đồng nghĩa ranh giới ung thư.')
        b.image(img,caption='Ảnh đang xét',use_container_width=True)
        key=s.get('image_key')
        if key in paths:
            pimg,_=read_npz(paths[key]);c.image(reference_patch(pimg,s),caption=f'TRAIN reference • {s.get("patient_id", "")}',use_container_width=True)
        else: c.warning('Chưa tìm thấy NPZ nguồn prototype này trong thư mục local.')
st.divider()
st.caption('Dữ liệu nghiên cứu CBIS-DDSM. Khi sử dụng NPZ + ROI thật, đây là evaluation có thông tin ROI, không chứng minh khả năng tự động phát hiện trên DICOM mới.')
