import * as T from 'three';
import {loadNative} from './native-assets.js';
export async function makeWorld(scene){
 const [track,data]=await Promise.all([loadNative('track'),fetch('/torcs/centerline.json').then(r=>r.json())]);scene.add(track);
 const curve=new T.CatmullRomCurve3(data.points.map(p=>new T.Vector3(...p)),true,'centripetal');curve.arcLengthDivisions=8000;curve.updateArcLengths();
 const length=curve.getLength(),samples=Array.from({length:801},(_,i)=>{const point=curve.getPointAt(i/800),tangent=curve.getTangentAt(i/800);return{point,tangent,normal:new T.Vector3(-tangent.z,0,tangent.x).normalize()}});
 function nearest(position){let best=samples[0],dist=Infinity,index=0;for(let i=0;i<800;i++){const d=(position.x-samples[i].point.x)**2+(position.z-samples[i].point.z)**2;if(d<dist){dist=d;best=samples[i];index=i}}return{...best,index,distance:Math.sqrt(dist),lateral:position.clone().sub(best.point).dot(best.normal)}}
 // Sample actual road triangles for height and banking; no synthetic terrain approximation.
 const roads=track.children.filter(m=>m.userData.semantic==='road'),ray=new T.Raycaster();track.updateMatrixWorld(true);
 function heightAt(x,z){ray.set(new T.Vector3(x,150,z),new T.Vector3(0,-1,0));const hit=ray.intersectObjects(roads,false)[0];if(hit)return hit.point.y+.08;const n=nearest(new T.Vector3(x,0,z));return n.point.y+.08;}
 const approx=new T.Vector3(1089.463,35.262,-574.777),near=nearest(approx),startFraction=near.index/800,startDistance=startFraction*length,start=curve.getPointAt(startFraction),direction=curve.getTangentAt(startFraction);
 return{roadHalfWidth:6,curve,length,samples,nearest,updateTrees:()=>{},heightAt,bounds:{radius:2000},startDistance,spawn:{x:start.x,z:start.z,heading:Math.atan2(-direction.x,-direction.z)}};
}
