import * as T from 'three';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {CarAvatar} from './car.js';
import {makeWorld} from './native-world.js';
import {loadNative} from './native-assets.js';
import {SemanticView} from './semantic.js';
const W=1344,H=768,FPS=24;
const palette=await (await fetch('/palette.json')).json();
const scene=new T.Scene();scene.background=new T.Color(0xaeb9c1);scene.fog=new T.Fog(0xaeb9c1,500,1800);
scene.add(new T.HemisphereLight(0xe2e7ed,0x606655,2.0));const sun=new T.DirectionalLight(0xf2f4f5,.65);sun.position.set(60,100,30);scene.add(sun);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-16,right:16,top:16,bottom:-16,near:1,far:170});sun.shadow.bias=-.0002;sun.shadow.normalBias=.02;scene.add(sun.target);
const renderer=new T.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(1);renderer.setSize(W,H,false);renderer.toneMapping=T.ACESFilmicToneMapping;renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;document.querySelector('#rgb').append(renderer.domElement);
const pmrem=new T.PMREMGenerator(renderer);const room=new RoomEnvironment();scene.environment=pmrem.fromScene(room,.04).texture;scene.environmentIntensity=.28;room.dispose();pmrem.dispose();
const camera=new T.PerspectiveCamera(40,W/H,.1,2000),world=await makeWorld(scene);
const host={add:o=>scene.add(o),remove:o=>scene.remove(o)};
const profile={color:0x183868,bodyLength:4.6,bodyWidth:2.08,rideHeight:.36,maxSpeed:36,reverseSpeed:8,acceleration:12,brake:26,drag:1.2,turnRate:2,grip:.88};
const player=new CarAvatar({host,world,profile});player.object.userData.semantic='hero_car';
const opponents=[0x20c9df,0xe7c534].map(color=>{const car=new CarAvatar({host,world,profile:{...profile,color}});car.object.userData.semantic='opponent_car';return car;});
// Native TORCS geometry uses +X forward; avatar physics uses -Z forward.
for(const [i,avatar] of [player,...opponents].entries()){
 avatar.body.visible=false;
 const native=await loadNative('car',{semanticOverride:i===0?'hero_car':'opponent_car'});
 native.rotation.y=Math.PI/2;avatar.object.add(native);
}
const semantic=new SemanticView(scene,camera,palette,W,H);document.querySelector('#semantic').append(semantic.canvas);
const keys=new Set();let auto=false,busy=false,live=false,time=0,lap=0,lastIndex=0,liveIndex=0,lastControls={},session=null,livePending=Promise.resolve();
const status=document.querySelector('#status');
const wrap=v=>T.MathUtils.euclideanModulo(v+Math.PI,Math.PI*2)-Math.PI;
const opponentStates=[{distance:22,speed:0},{distance:38,speed:0}];
let chaseHeading=0,trackProgress=0;
const DEFAULT_AUTOPILOT_POLICY=Object.freeze({id:'default',targetSpeed:30,lookaheadBase:8,lookaheadSpeed:0.25,steeringGain:2.8,laneOffset:0,weaveAmplitude:0,weaveFrequency:0,speedAmplitude:0,speedFrequency:0,brakePulse:0,phase:0,opponentGaps:[22,38]});
let autopilotPolicy={...DEFAULT_AUTOPILOT_POLICY,opponentGaps:[...DEFAULT_AUTOPILOT_POLICY.opponentGaps]};
function setAutopilotPolicy(policy={}){
 autopilotPolicy={...DEFAULT_AUTOPILOT_POLICY,...policy,opponentGaps:[...(policy.opponentGaps??DEFAULT_AUTOPILOT_POLICY.opponentGaps)]};
 return {...autopilotPolicy,opponentGaps:[...autopilotPolicy.opponentGaps]};
}
// A continuous centreline coordinate avoids sample-index jumps in car following.
function trackDistance(position){const n=world.nearest(position);let d=n.index/800*world.length+position.clone().sub(n.point).dot(n.tangent);d+=Math.round((trackProgress-d)/world.length)*world.length;trackProgress=d;return d;}

function reset(){lastControls={throttle:0,brake:0,steering:0,handbrake:false,autopilot:auto,policyId:autopilotPolicy.id,targetSpeed:autopilotPolicy.targetSpeed,targetLaneOffset:autopilotPolicy.laneOffset};opponentStates.forEach((s,i)=>{s.distance=world.startDistance+autopilotPolicy.opponentGaps[i];s.speed=0;});trackProgress=world.startDistance;time=0;lap=0;lastIndex=0;player.placeAt(world.spawn);player.heading=world.spawn.heading;player.object.rotation.order="YXZ";player.object.rotation.y=player.heading;const slope=world.nearest(player.motion.position).tangent;player.object.rotation.x=Math.atan2(slope.y,Math.hypot(slope.x,slope.z));if(auto){player.speed=autopilotPolicy.targetSpeed;player.velocity.set(-Math.sin(player.heading)*autopilotPolicy.targetSpeed,0,-Math.cos(player.heading)*autopilotPolicy.targetSpeed);opponentStates.forEach(s=>s.speed=autopilotPolicy.targetSpeed);}
player.body.rotation.set(0,0,0);player.wheelSpin=0;player.steerAngle=0;for(const row of Object.values(player.wheels))for(const wheel of row){wheel.spin.rotation.x=0;wheel.steerPivot.rotation.y=0;}lastControls={...lastControls,position:player.object.position.toArray(),heading:player.heading,speed:player.speed,reverse:false};updateOpponents();updateCamera(true);render();}
function updateOpponents(dt=0){
 const progress=trackDistance(player.motion.position);
 opponents.forEach((car,i)=>{
  const state=opponentStates[i],gap=autopilotPolicy.opponentGaps[i];
  const error=progress+gap-state.distance;
  // Bounded acceleration and gap feedback prevent the old scripted 15s pass.
  const targetSpeed=T.MathUtils.clamp(player.speed+error*.9,0,38);
  state.speed+=T.MathUtils.clamp(targetSpeed-state.speed,-6*dt,14*dt);
  state.distance+=state.speed*dt;
  const t=T.MathUtils.euclideanModulo(state.distance/world.length,1);
  const p=world.curve.getPointAt(t),v=world.curve.getTangentAt(t),n=new T.Vector3(-v.z,0,v.x);
  car.object.position.copy(p).addScaledVector(n,i===0?-1.65:1.65);
  car.object.position.y=world.heightAt(p.x,p.z);car.object.rotation.order="YXZ";car.object.rotation.y=Math.atan2(-v.x,-v.z);car.object.rotation.x=Math.atan2(v.y,Math.hypot(v.x,v.z));
 });
}
function updateCamera(snap=false,dt=1/FPS){
 // Finite yaw response and a road-centred gaze produce turn-dependent framing.
 // No sinusoidal shake and no copied screen-space trajectory.
 if(snap)chaseHeading=player.heading;
 else chaseHeading+=wrap(player.heading-chaseHeading)*(1-Math.exp(-dt/.30));
 const back=new T.Vector3(Math.sin(chaseHeading),0,Math.cos(chaseHeading));
 const pos=player.object.position.clone().addScaledVector(back,6.02).add(new T.Vector3(0,1.88-6.02*world.nearest(player.motion.position).tangent.y,0));
 camera.position.copy(pos);
 const target=player.object.position.clone().add(new T.Vector3(0,1.0,0));
 camera.lookAt(target);
}
function step(dt){
  let throttle=Number(keys.has('KeyW')||keys.has('ArrowUp'))-Number(keys.has('KeyS')||keys.has('ArrowDown'));
  let steer=Number(keys.has('KeyD')||keys.has('ArrowRight'))-Number(keys.has('KeyA')||keys.has('ArrowLeft'));
  if(auto){
   const near=world.nearest(player.motion.position);
   const lookahead=autopilotPolicy.lookaheadBase+Math.abs(player.speed)*autopilotPolicy.lookaheadSpeed;
   const target=world.curve.getPointAt(((near.index/800)+lookahead/world.length)%1);
   const targetNear=world.nearest(target);
   const lane=autopilotPolicy.laneOffset+autopilotPolicy.weaveAmplitude*Math.sin(time*autopilotPolicy.weaveFrequency+autopilotPolicy.phase);
   target.addScaledVector(targetNear.normal,lane);
   const v=target.sub(player.motion.position),desired=Math.atan2(-v.x,-v.z);
   steer=T.MathUtils.clamp(-wrap(desired-player.heading)*autopilotPolicy.steeringGain,-1,1);
   const targetSpeed=autopilotPolicy.targetSpeed+autopilotPolicy.speedAmplitude*Math.sin(time*autopilotPolicy.speedFrequency+autopilotPolicy.phase);
   const braking=autopilotPolicy.brakePulse>0&&T.MathUtils.euclideanModulo(time+autopilotPolicy.phase,4)<autopilotPolicy.brakePulse;
   throttle=braking?-1:(player.speed<targetSpeed?1:0);
  }
  player.applyRuntimeInput({moveY:throttle,moveX:steer,jump:keys.has('Space'),run:false});player.tick(dt);
  const near=world.nearest(player.motion.position);
  if(Math.abs(near.lateral)>world.roadHalfWidth+.35){player.motion.position.copy(near.point).addScaledVector(near.normal,Math.sign(near.lateral)*(world.roadHalfWidth+.35));player.motion.position.y=near.point.y+.025;player.object.position.copy(player.motion.position);player.speed*=.75;player.velocity.multiplyScalar(.6);}
  if(lastIndex>720&&near.index<80)lap++;lastIndex=near.index;
  player.object.rotation.order="YXZ";player.object.rotation.x=Math.atan2(near.tangent.y,Math.hypot(near.tangent.x,near.tangent.z));
  time+=dt;updateOpponents(dt);updateCamera(false,dt);
  lastControls={throttle:Math.max(0,throttle),brake:throttle<0?1:0,reverse:throttle<0&&player.speed<0,steering:steer,handbrake:player.handbrake,autopilot:auto,policyId:autopilotPolicy.id,targetSpeed:autopilotPolicy.targetSpeed,targetLaneOffset:autopilotPolicy.laneOffset,position:player.object.position.toArray(),heading:player.heading,speed:player.speed,trackLateral:near.lateral,cameraPosition:camera.position.toArray(),cameraFov:camera.fov,opponentGaps:opponentStates.map(s=>s.distance-trackDistance(player.motion.position))};
}
function render(){world.updateTrees(camera);sun.position.copy(player.object.position).add(new T.Vector3(-30,55,20));sun.target.position.copy(player.object.position);renderer.render(scene,camera);semantic.render();document.querySelector('#speed').textContent=`${Math.abs(player.speed*3.6).toFixed(0)} km/h · ${lap} 圈`;}
async function api(path,body,headers={}){const r=await fetch('/api/'+path,{method:'POST',body,headers});if(!r.ok)throw Error(await r.text());return r.json();}
async function startSession(kind,frames){const config={kind,frames,width:W,height:H,fps:FPS,palette,autopilot:auto,autopilotPolicy:{...autopilotPolicy,opponentGaps:[...autopilotPolicy.opponentGaps]}};if(kind==='live'){return (await import('./runtime-client.js')).startRuntime(config);}return api('session',JSON.stringify(config),{'Content-Type':'application/json'});}
async function sendFrame(index){const blob=await semantic.png();const controls={...lastControls,frame:index,timestamp:index/FPS};if(session.runtime){return (await import('./runtime-client.js')).sendRuntimeFrame(session,index,blob,controls);}return api(`frame/${session.id}/${index}`,blob,{'Content-Type':'image/png','X-Controls':JSON.stringify(controls)});}
async function reference(){const blob=await new Promise(r=>renderer.domElement.toBlob(r,'image/png'));await api(`reference/${session.id}`,blob,{'Content-Type':'image/png'});}
async function record(frames=719){if(busy||live)throw Error('已有录制或语义流正在运行');busy=true;document.querySelector('#record').disabled=true;try{
  reset();session=await startSession('record',frames);render();await reference();
  for(let i=0;i<frames;i++){if(i>0)for(let j=0;j<2;j++)step(1/(FPS*2));render();await sendFrame(i);status.textContent=`正在导出 ${i+1}/${frames}`;}
  const result=await api(`finish/${session.id}`);status.textContent=`已保存：exports/${session.id}`;return result;
}finally{busy=false;document.querySelector('#record').disabled=false;}}
async function toggleLive(){if(busy)return;if(live){live=false;busy=true;try{await (await import('./runtime-client.js')).closeRuntime(session.id);await livePending;}finally{busy=false;}status.textContent='Streaming render 已停止';document.querySelector('#live').textContent='开启 streaming render';return;}busy=true;try{reset();session=await startSession('live',null);}finally{busy=false;}liveIndex=0;live=true;document.querySelector('#live').textContent='停止 streaming render';status.textContent='Streaming render · 24 fps 仿真时钟';}
function toggleAuto(){auto=!auto;document.querySelector('#auto').textContent=`自动驾驶：${auto?'开':'关'}`;}
window.addEventListener('keydown',e=>{if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space','Tab'].includes(e.code))e.preventDefault();keys.add(e.code);if(e.repeat)return;if(e.code==='KeyR'&&!busy&&!live)reset();if(e.code==='Tab')document.querySelector('#semantic').classList.toggle('full');});window.addEventListener('keyup',e=>keys.delete(e.code));window.addEventListener('blur',()=>keys.clear());
document.querySelector('#reset').onclick=()=>{if(!busy&&!live)reset();};document.querySelector('#auto').onclick=()=>{if(!busy)toggleAuto();};document.querySelector('#view').onclick=()=>document.querySelector('#semantic').classList.toggle('full');
document.querySelector('#record').onclick=()=>record().catch(e=>status.textContent=e.message);document.querySelector('#live').onclick=()=>toggleLive().catch(e=>status.textContent=e.message);
let last=performance.now(),acc=0;async function loop(now){const dt=Math.min((now-last)/1000,.1);last=now;
 try{if(!busy){if(live){if(liveIndex>0)for(let j=0;j<2;j++)step(1/(FPS*2));render();livePending=sendFrame(liveIndex);await livePending;liveIndex++;await new Promise(r=>setTimeout(r,Math.max(0,1000/FPS-(performance.now()-now))));}else{acc+=dt;while(acc>=1/120){step(1/120);acc-=1/120;}render();}}}catch(e){live=false;status.textContent=e.message;try{await (await import('./runtime-client.js')).closeRuntime(session?.id);}catch{}}requestAnimationFrame(loop);}
reset();requestAnimationFrame(loop);
window.racing={replayFrame(index){if(index===0){busy=true;auto=true;reset();}else{for(let j=0;j<2;j++)step(1/(FPS*2));render();}return lastControls;},rgbCanvas:renderer.domElement,player,world,scene,camera,semantic,palette,record,reset,step,render,setAuto:v=>{if(auto!==v)toggleAuto();},setAutopilotPolicy,state:()=>({time,auto,busy,live,autopilotPolicy:{...autopilotPolicy},position:player.object.position.toArray(),speed:player.speed}),controls:keys};
