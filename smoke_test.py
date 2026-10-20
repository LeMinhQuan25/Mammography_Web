from pathlib import Path
from inference import load_model,read_npz
root=Path(__file__).parent
model,ck=load_model(root/'checkpoints'/'final_model.pt')
print('Checkpoint loaded strictly:',model.num_prototypes,'prototypes; config:',ck['scientific_version'])
