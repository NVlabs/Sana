#!/usr/bin/env python3
"""Scene-specific temporal guardrails, additional to exact-palette validation.

These checks detect missing roads, abrupt class changes, and accidental close
passes in this controlled demo. They do not certify H3 output quality or define
universal thresholds for TORCS data.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);args=p.parse_args()
c=args.capture;m=json.loads((c/'manifest.json').read_text());rows=[];first=None
for f in sorted((c/'semantic').glob('*.png')):
 a=np.asarray(Image.open(f).convert('RGB'))[::4,::4];hero=(a==[255,0,0]).all(-1);ys,xs=np.where(hero)
 assert xs.size,f'No hero car: {f.name}'
 first=hero if first is None else first
 rows.append({'frame':int(f.stem),'road_pct':float((a==[128,64,128]).all(-1).mean()*100),'opponent_pct':float((a==[0,0,230]).all(-1).mean()*100),'sky_pct':float((a==[70,130,180]).all(-1).mean()*100),'hero_area_pct':float(hero.mean()*100),'hero_center_x_pct':float(xs.mean()/a.shape[1]*100),'hero_width_pct':float((xs.max()-xs.min()+1)/a.shape[1]*100),'hero_iou_first':float((hero&first).sum()/(hero|first).sum())})
assert len(rows)==m['frames'] and all(v['frame']==i for i,v in enumerate(rows))
summary={k:{'mean':float(np.mean([v[k] for v in rows])),'std':float(np.std([v[k] for v in rows])),'min':float(min(v[k]for v in rows)),'max':float(max(v[k]for v in rows))}for k in rows[0]if k!='frame'}
road=np.array([v['road_pct']for v in rows]);bad=[v['frame']for v in rows if v['road_pct']<5];jump=float(np.abs(np.diff(road)).max())
controls=[json.loads(l)for l in (c/'controls.jsonl').read_text().splitlines()]
gaps=[g for v in controls if 'opponentGaps' in v for g in v['opponentGaps']]
report={'frames':len(rows),'road_missing_frames':bad,'max_road_change_percentage_points':jump,'opponent_gap_min_m':min(gaps)if gaps else None,'metrics':summary,'measurement':'all frames; spatial stride 4','checks':{'road_present':not bad,'road_continuity':jump<8,'opponent_area_below_8pct':summary['opponent_pct']['max']<8,'minimum_opponent_gap_12m':bool(gaps)and min(gaps)>12}}
report['passed']=all(report['checks'].values());(c/'temporal_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert report['passed'],'Temporal validation failed'
