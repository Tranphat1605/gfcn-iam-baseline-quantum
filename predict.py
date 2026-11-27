"""Recognize one cropped text-line image using either trained model."""
import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', choices=['baseline','quantum'], required=True)
parser.add_argument('--image', type=Path, required=True)
parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
args = parser.parse_args()
if not args.image.is_file():
    parser.error('Image does not exist. Supply a cropped image of ONE text line.')
if args.device == 'cuda' and not torch.cuda.is_available():
    parser.error('CUDA is unavailable; use --device cpu.')
device = torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
torch.set_num_threads(2)
sys.path.insert(0, str(ROOT/args.model))
from source.models import GFCN
with (ROOT/args.model/'Datasets/iam_lines/iam_lines_64H_NKR/params.pkl').open('rb') as stream:
    stats = pickle.load(stream)
params = {'normalization':{'type':'instance','eps':0.001,'momentum':0.99,'num_group':32,'track_running_stats':False},
          'device':device,'vocab_size':len(stats['labels']),'input_channels':1,'nb_gate':6,
          'quantum_input_mean':float(stats['mean'][0]),'quantum_input_std':float(stats['std'][0]),
          'quanvolution_depth':2,'quanvolution_seed':0,'quantum_chunk_size':4096}
if args.model == 'quantum':
    from source.quantum_models import QuanvolutionOnlyGFCN
    model = QuanvolutionOnlyGFCN(params)
    output, filename = 'QGFCN_iam_4q_filter_only', 'best_90.pt'
else:
    model = GFCN(params)
    output, filename = 'GFCN_iam_lignes_64H_NKR_baseline', 'best_126.pt'
checkpoint = ROOT/args.model/'source/outputs'/output/'checkpoints'/filename
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
model.load_state_dict(saved['end_to_end_model_state_dict'], strict=True)
epoch = saved['epoch']+1
del saved
model.to(device).eval()
with Image.open(args.image) as source:
    image = ImageOps.exif_transpose(source)
    if image.mode in ['RGBA','LA'] or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        image = Image.alpha_composite(Image.new('RGBA',rgba.size,'white'), rgba)
    image = image.convert('L')
    if image.width < 32:
        parser.error('The line image must be at least 32 pixels wide.')
    image = image.resize((image.width,64), Image.Resampling.LANCZOS)
    data = (np.asarray(image, dtype=np.float32)-stats['mean'][0])/stats['std'][0]
with torch.inference_mode():
    indices = model(torch.from_numpy(data)[None,None].to(device)).argmax(dim=1)[0].cpu().tolist()
text, previous = [], None
for index in indices:
    if index != previous and index < len(stats['labels']):
        text.append(stats['labels'][index])
    previous = index
print(json.dumps({'model':args.model,'checkpoint':filename,'epoch':epoch,'device':str(device),
                  'prediction':''.join(text)}, ensure_ascii=False))
