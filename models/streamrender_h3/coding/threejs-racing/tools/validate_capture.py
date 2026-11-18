#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);args=p.parse_args()
m=json.loads((args.capture/'manifest.json').read_text());palette=json.loads((args.capture/'palette.json').read_text())
allowed={tuple(rgb):name for name,rgb in palette.items()};counts={name:0 for name in palette}
assert m['complete'] and m['next']==m['frames']
frames=sorted((args.capture/'semantic').glob('*.png'));assert len(frames)==m['frames']
controls=[json.loads(s) for s in (args.capture/'controls.jsonl').read_text().splitlines()];assert len(controls)==len(frames)
for i,file in enumerate(frames):
    assert file.name==f'{i:06d}.png'
    im=Image.open(file).convert('RGB');assert im.size==(1344,768)
    colors=im.getcolors(im.width*im.height)
    for count,rgb in colors:
        assert rgb in allowed,(file.name,rgb)
        counts[allowed[rgb]]+=count
    assert controls[i]['frame']==i and abs(controls[i]['timestamp']-i/24)<1e-9
for name in ['road','road_edge','hero_car','opponent_car','sky','grass']:
    assert counts[name]>0,name
report={'passed':True,'frames':len(frames),'size':[1344,768],'fps':24,'off_palette_pixels':0,'class_pixels':counts}
(args.capture/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
