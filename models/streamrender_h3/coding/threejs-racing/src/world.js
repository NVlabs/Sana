import * as T from 'three';
import {addTorcsTrees} from './trees.js';
export function makeWorld(scene){
  const roadHalfWidth=Number(new URLSearchParams(location.search).get("roadHalfWidth")||4.5);
  if(roadHalfWidth<3.5||roadHalfWidth>6)throw Error("Invalid road half width");
  // Alternating-radius bends provide changing road geometry in a continuous drive.
  const points=[[0,0],[0,-85],[45,-165],[105,-245],[70,-335],[-35,-395],[-155,-325],[-200,-180],[-130,-40]];
  const curve=new T.CatmullRomCurve3(points.map(([x,z],i)=>new T.Vector3(x,12+7*Math.sin(i/points.length*Math.PI*2)+3*Math.sin(i/points.length*Math.PI*4),z)),true,'centripetal');
  const length=curve.getLength(),samples=Array.from({length:801},(_,i)=>{const point=curve.getPointAt(i/800),tangent=curve.getTangentAt(i/800);return{point,tangent,normal:new T.Vector3(-tangent.z,0,tangent.x).normalize()}});
  function mesh(g,color,label,pos=[0,0,0]){const o=new T.Mesh(g,new T.MeshStandardMaterial({color,roughness:.86,side:T.DoubleSide}));o.position.set(...pos);o.userData.semantic=label;o.receiveShadow=true;scene.add(o);return o}
  function box(size,pos,color,label){return mesh(new T.BoxGeometry(...size),color,label,pos)}
  function ribbon(a,b,y,color,label){const p=[],ix=[];samples.forEach(({point,normal},i)=>{for(const d of[a,b])p.push(point.x+normal.x*d,point.y+y,point.z+normal.z*d);if(i<800){const n=i*2;ix.push(n,n+2,n+1,n+1,n+2,n+3)}});const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(p,3));g.setIndex(ix);g.computeVertexNormals();return mesh(g,color,label)}
  mesh(new T.PlaneGeometry(4000,4000,128,128).rotateX(-Math.PI/2),0x79815f,'grass',[0,-.5,0]);
  function groundHeightAt(x,z){let best=Infinity,height=0;
    for(let j=0;j<800;j++){const a=samples[j].point,b=samples[j+1].point,dx=b.x-a.x,dz=b.z-a.z,t=T.MathUtils.clamp(((x-a.x)*dx+(z-a.z)*dz)/(dx*dx+dz*dz),0,1),ex=x-a.x-dx*t,ez=z-a.z-dz*t,d=ex*ex+ez*ez;if(d<best){best=d;height=a.y+(b.y-a.y)*t;}}
    return height*Math.exp(-best/10000)-.5;
  }
  const ground=new T.PlaneGeometry(640,640,128,128).rotateX(-Math.PI/2),gp=ground.attributes.position;
  for(let i=0;i<gp.count;i++){const x=gp.getX(i)-40,z=gp.getZ(i)-180;gp.setXYZ(i,x,groundHeightAt(x,z),z);}
  ground.computeVertexNormals();mesh(ground,0x79815f,'grass');
  const road=ribbon(-roadHalfWidth,roadHalfWidth,0,0x56585a,'road');
  // Fine deterministic asphalt noise, used by RGB only.
  const canvas=document.createElement('canvas');canvas.width=canvas.height=256;const ctx=canvas.getContext('2d'),im=ctx.createImageData(256,256);let seed=19;
  for(let i=0;i<im.data.length;i+=4){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const v=82+(seed/4294967296-.5)*16;im.data[i]=v;im.data[i+1]=v;im.data[i+2]=v;im.data[i+3]=255}ctx.putImageData(im,0,0);
  const tex=new T.CanvasTexture(canvas);tex.wrapS=tex.wrapT=T.RepeatWrapping;tex.colorSpace=T.SRGBColorSpace;
  const uv=[];samples.forEach((_,i)=>{uv.push(0,i*length/800/5,2.4,i*length/800/5)});road.geometry.setAttribute('uv',new T.Float32BufferAttribute(uv,2));road.material.color.setHex(0xffffff);road.material.map=tex;
  ribbon(-roadHalfWidth-.35,-roadHalfWidth,.012,0xd6d5cc,'road_edge');ribbon(roadHalfWidth,roadHalfWidth+.35,.012,0xd6d5cc,'road_edge');
  // Continuous rails avoid repeated disconnected slats in semantic input.
  for(const side of[-1,1]){const p=[],ix=[];samples.forEach(({point,normal},i)=>{for(const y of[.25,.82])p.push(point.x+normal.x*side*(roadHalfWidth+2.3),point.y+y,point.z+normal.z*side*(roadHalfWidth+2.3));if(i<800){const n=i*2;ix.push(n,n+2,n+1,n+1,n+2,n+3)}});const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(p,3));g.setIndex(ix);g.computeVertexNormals();mesh(g,0x9a9f9b,'barrier')}
  // Continuous terrain surrounds the circuit, so turning does not reveal an empty skyline.
  const radii=[330,700,1000,1400],segments=128,vertices=[],indices=[];
  for(let ring=0;ring<radii.length;ring++)for(let i=0;i<=segments;i++){
    const a=i/segments*Math.PI*2,r=radii[ring];
    const y=ring===0?-.2:[0,75,100,120][ring]+15*Math.sin(a*3+ring*.3)+10*Math.sin(a*7+ring);
    vertices.push(-40+Math.cos(a)*r,y,-180+Math.sin(a)*r);
    if(ring>0&&i<segments){const n=ring*(segments+1)+i,k=n-(segments+1);indices.push(k,n,k+1,k+1,n,n+1);}
  }
  const terrain=new T.BufferGeometry();terrain.setAttribute('position',new T.Float32BufferAttribute(vertices,3));terrain.setIndex(indices);terrain.computeVertexNormals();mesh(terrain,0x798765,'grass');
  // Small pit buildings remain beside the track, never overhead.
  const pit=samples[200].point.clone().addScaledVector(samples[200].normal,45);box([15,4,20],[pit.x,groundHeightAt(pit.x,pit.z)+2,pit.z],0xb3b5b3,'building');
  const updateTrees=addTorcsTrees(scene,samples,groundHeightAt);
  function nearest(position){let best=samples[0],dist=Infinity,index=0;for(let i=0;i<800;i++){const d=(position.x-samples[i].point.x)**2+(position.z-samples[i].point.z)**2;if(d<dist){dist=d;best=samples[i];index=i}}return{...best,index,distance:Math.sqrt(dist),lateral:position.clone().sub(best.point).dot(best.normal)}}
  const startFraction=Number(new URLSearchParams(location.search).get("startFraction")||.14),startDistance=startFraction*length,start=curve.getPointAt(startFraction),direction=curve.getTangentAt(startFraction);
  return{roadHalfWidth,curve,length,samples,nearest,updateTrees,heightAt:(x,z)=>{const n=nearest(new T.Vector3(x,0,z));const along=((x-n.point.x)*n.tangent.x+(z-n.point.z)*n.tangent.z)/(n.tangent.x**2+n.tangent.z**2);return n.point.y+along*n.tangent.y+.025;},bounds:{radius:1800},startDistance,spawn:{x:start.x,z:start.z,heading:Math.atan2(-direction.x,-direction.z)}};
}
