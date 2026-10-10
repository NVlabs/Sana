import {chromium} from 'playwright';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const baseUrl=process.env.CODEGAME_BASE_URL??'http://127.0.0.1:5191';
const count=Number(process.argv[2]??64);
const frames=Number(process.argv[3]??107);
const output=path.resolve(process.argv[4]??'exports/demo64_manifest.json');
const families=['steady_cruise','lane_offset','slalom','speed_pulse','brake_pulse','late_apex','close_traffic','mixed_dynamic'];
const speeds=[18,22,26,30,33,35,28,24];
const lanes=[-1.35,-.95,-.55,-.2,.2,.55,.95,1.35];

function makePolicy(index){
 const familyIndex=Math.floor(index/8)%families.length,variant=index%8;
 const p={
  id:`demo_${String(index).padStart(2,'0')}_${families[familyIndex]}`,
  family:families[familyIndex],
  variant,
  targetSpeed:speeds[variant],
  lookaheadBase:7+(variant%4)*.8,
  lookaheadSpeed:.20+(variant%3)*.035,
  steeringGain:2.35+(variant%5)*.18,
  laneOffset:0,
  weaveAmplitude:0,
  weaveFrequency:0,
  speedAmplitude:0,
  speedFrequency:0,
  brakePulse:0,
  phase:index*.47,
  opponentGaps:[18+(variant%4)*3,34+(variant%3)*5],
 };
 if(familyIndex===1)p.laneOffset=lanes[variant];
 if(familyIndex===2){p.weaveAmplitude=.45+variant*.12;p.weaveFrequency=.75+variant*.08;}
 if(familyIndex===3){p.speedAmplitude=3+variant*.65;p.speedFrequency=.75+variant*.09;}
 if(familyIndex===4){p.brakePulse=.10+variant*.035;p.targetSpeed=27+variant;}
 if(familyIndex===5){p.laneOffset=lanes[7-variant]*.75;p.lookaheadBase=5.5+variant*.65;p.steeringGain=2.7+variant*.16;}
 if(familyIndex===6){p.opponentGaps=[8+variant*1.4,16+variant*2];p.laneOffset=lanes[variant]*.55;}
 if(familyIndex===7){p.laneOffset=lanes[variant]*.65;p.weaveAmplitude=.25+variant*.07;p.weaveFrequency=.65+variant*.06;p.speedAmplitude=2+variant*.4;p.speedFrequency=.8+variant*.05;p.brakePulse=variant%2?.12:0;}
 return p;
}

assert.equal(count,64,'This demo batch is defined as exactly 64 paired trajectories');
assert.ok((frames-5)%17===0,'Frame count must satisfy H3 5+17*k');
await mkdir(path.dirname(output),{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
const records=[];
try{
 const page=await browser.newPage({viewport:{width:1440,height:1100}});
 page.on('pageerror',error=>console.error('PAGE ERROR',error));
 await page.goto(baseUrl);
 await page.waitForFunction(()=>window.racing);
 for(let index=0;index<count;index++){
  const policy=makePolicy(index);
  await page.evaluate(p=>{racing.setAutopilotPolicy(p);racing.setAuto(true);},policy);
  const result=await page.evaluate(n=>racing.record(n),frames);
  const captureDir=path.resolve('exports',result.id);
  const manifest=JSON.parse(await readFile(path.join(captureDir,'manifest.json'),'utf8'));
  const lines=(await readFile(path.join(captureDir,'controls.jsonl'),'utf8')).trim().split('\n');
  const controls=lines.map(line=>JSON.parse(line));
  assert.equal(manifest.complete,true);
  assert.equal(manifest.frames,frames);
  assert.equal(manifest.autopilotPolicy.id,policy.id);
  assert.equal(controls.length,frames);
  controls.forEach((control,frame)=>{
   assert.equal(control.frame,frame);
   assert.equal(control.timestamp,frame/manifest.fps);
   assert.equal(control.policyId,policy.id);
  });
  const record={
   index,
   id:policy.id,
   trajectoryFamily:policy.family,
   policy,
   captureId:result.id,
   captureDir,
   referenceImage:path.join(captureDir,'ref.png'),
   semanticFrames:path.join(captureDir,'semantic/%06d.png'),
   referenceVideo:path.join(captureDir,'semantic.mp4'),
   actionTranscript:path.join(captureDir,'controls.jsonl'),
   frames,
   fps:manifest.fps,
   duration:manifest.duration,
  };
  records.push(record);
  await writeFile(output,JSON.stringify({complete:false,count:records.length,frames,records},null,2));
  console.log(`CAPTURE ${index+1}/${count} ${policy.id} ${result.id}`);
 }
 await writeFile(output,JSON.stringify({complete:true,count:records.length,frames,records},null,2));
 console.log(`BATCH_MANIFEST ${output}`);
}finally{
 await browser.close();
}
