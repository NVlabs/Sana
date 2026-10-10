import * as T from 'three';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';

// A closed, lofted GT shell. Dimensions are metres; forward is -Z.
// All children inherit the owning car's single semantic label.
export function createCarBody(profile){
 const group=new T.Group(),geometries=[],materials=[],textures=[];
 const material=opts=>{const m=new T.MeshStandardMaterial(opts);materials.push(m);return m;};
 const paint=material({color:profile.color,roughness:.38,metalness:.38});
 const carbon=material({color:0x11151a,roughness:.48,metalness:.28});
 const glass=material({color:0x1a303c,roughness:.12,metalness:.62});
 const rubber=material({color:0x131415,roughness:.95});
 const alloy=material({color:0xa3a9b0,roughness:.25,metalness:.85});
 const accent=material({color:0xced84a,roughness:.36,metalness:.25});
 const red=material({color:0xae1017,emissive:0x680309,emissiveIntensity:.65,roughness:.22});
 const white=material({color:0xe6edee,emissive:0xabbcc4,emissiveIntensity:.3});
 function mesh(g,m,pos=[0,0,0],parent=group){geometries.push(g);const o=new T.Mesh(g,m);o.position.set(...pos);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}
 const box=(size,pos,m=paint,parent=group)=>mesh(new RoundedBoxGeometry(...size,3,Math.min(...size)*.2),m,pos,parent);
 function panel(points,m){const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(points.flat(),3));g.setAttribute('uv',new T.Float32BufferAttribute([0,1,0,0,1,0,1,1],2));g.setIndex([0,1,2,0,2,3]);g.computeVertexNormals();const o=mesh(g,m);o.material.side=T.DoubleSide;return o;}
 // Rounded shoulder, wide rear haunches, low nose: not stacked cuboids.
 const sections=[[-2.3,.78,.29,.53],[-2.12,.95,.24,.68],[-1.52,1.02,.25,.79],[-.82,.97,.27,.79],[.15,.94,.28,.79],[1.15,1.045,.27,.88],[1.9,1.04,.28,.86],[2.19,.94,.31,.72]];
 const ring=[[-.86,0],[-1,.19],[-1,.66],[-.94,.91],[-.76,1],[.76,1],[.94,.91],[1,.66],[1,.19],[.86,0]];
 const verts=[],idx=[];
 sections.forEach(([z,w,bottom,top],i)=>{ring.forEach(([x,y])=>verts.push(x*w,bottom+(top-bottom)*y,z));if(i)for(let j=0;j<ring.length;j++){const a=(i-1)*ring.length+j,b=(i-1)*ring.length+(j+1)%ring.length,c=i*ring.length+j,d=i*ring.length+(j+1)%ring.length;idx.push(a,b,c,b,d,c);}});
 for(let j=1;j<ring.length-1;j++){idx.push(0,j+1,j);const a=(sections.length-1)*ring.length;idx.push(a,a+j,a+j+1);}
 const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(verts,3));for(let i=0;i<idx.length;i+=3)[idx[i+1],idx[i+2]]=[idx[i+2],idx[i+1]];g.setIndex(idx);g.computeVertexNormals();mesh(g,paint);
 // Roof / pillars with sloping front, rear and side glazing.
 const roofPoints=[],roofIndices=[];
 for(let j=0;j<=12;j++){const x=(j/12*2-1)*.72,y=1.365-.065*(x/.72)**2;roofPoints.push(x,y,-.56,x,y,.58);if(j<12){const a=j*2;roofIndices.push(a,a+1,a+2,a+1,a+3,a+2);}}
 const roof=new T.BufferGeometry();roof.setAttribute('position',new T.Float32BufferAttribute(roofPoints,3));roof.setIndex(roofIndices);roof.computeVertexNormals();mesh(roof,paint);
 panel([[-.67,1.33,-.56],[.67,1.33,-.56],[.86,.8,-1.11],[-.86,.8,-1.11]],glass);
 panel([[-.68,1.33,.57],[-.89,.88,1.29],[.89,.88,1.29],[.68,1.33,.57]],glass);
 for(const side of [-1,1]){
  panel([[side*.69,1.32,-.51],[side*.69,1.32,.52],[side*.91,.84,1.14],[side*.88,.81,-1.01]],glass);
  function strut(a,b,r=.038,m=paint){const start=new T.Vector3(...a),end=new T.Vector3(...b),d=end.clone().sub(start);const o=mesh(new T.CylinderGeometry(r,r,d.length(),8),m,start.clone().add(end).multiplyScalar(.5).toArray());o.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),d.normalize());}
  strut([side*.69,1.34,-.55],[side*.91,.78,-1.13]);strut([side*.69,1.34,.57],[side*.94,.84,1.35],.065);
  strut([side*.7,1.33,.05],[side*.91,.8,.08],.029,carbon);
  box([.21,.12,.35],[side*1.035,.91,-.64]);
  box([.12,.13,3.1],[side*.96,.27,0],carbon);
  // Sculpted wheel arch lips; open centres show tyres rather than square wheel blocks.
  for(const z of [-1.43,1.4]){const arch=mesh(new T.TorusGeometry(.405,.044,8,32,Math.PI),paint,[side*1.01,.38,z]);arch.rotation.y=Math.PI/2;}
 }
 box([1.93,.09,.35],[0,.23,-2.14],carbon);box([1.84,.12,.32],[0,.29,2.13],carbon);
 box([1.35,.22,.045],[0,.45,-2.285],carbon);
 box([1.65,.10,.045],[0,.365,2.21],carbon);
 for(let i=-3;i<=3;i++)box([.023,.17,.42],[i*.21,.25,2.09],carbon);
 // GT rear wing with two pylons and upright endplates.
 box([2.12,.075,.35],[0,1.08,1.89],carbon);
 for(const side of [-1,1]){box([.045,.36,.14],[side*.66,.91,1.88],carbon);box([.045,.21,.42],[side*1.05,1.1,1.89],paint);}
 for(const side of [-1,1]){
  for(const offset of [0,.22]){const lamp=mesh(new T.CylinderGeometry(.082,.082,.055,24),red,[side*(.65+offset),.64,2.235]);lamp.rotation.x=Math.PI/2;}
  const light=box([.32,.1,.06],[side*.65,.62,-2.17],white);light.rotation.y=side*.15;
  const exhaust=mesh(new T.CylinderGeometry(.065,.065,.16,20),alloy,[side*.7,.34,2.25]);exhaust.rotation.x=Math.PI/2;
 }
 // Original livery, no external textures or model download needed.
 const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=256;const ctx=canvas.getContext('2d');
 ctx.fillStyle='#071a36';ctx.fillRect(0,0,1024,256);ctx.fillStyle='#d9e261';ctx.font='italic bold 68px sans-serif';ctx.textAlign='center';ctx.fillText('APEX MOTORSPORT',512,105);ctx.font='bold 82px sans-serif';ctx.fillText('37',855,220);ctx.fillStyle='#dce6ec';ctx.font='26px sans-serif';ctx.fillText('GT  /  ENDURANCE',400,192);
 const tex=new T.CanvasTexture(canvas);tex.colorSpace=T.SRGBColorSpace;textures.push(tex);const decal=material({map:tex,roughness:.42});
 panel([[-.62,.61,2.24],[-.62,.44,2.24],[.62,.44,2.24],[.62,.61,2.24]],decal);
 for(const side of [-1,1])panel([[side*.97,.74,1.87],[side*.99,.35,1.87],[side*.92,.34,2.2],[side*.9,.68,2.2]],accent);
 const wheels={front:[],rear:[]},wheelRadius=profile.rideHeight;
 for(const [row,z] of [['front',-1.43],['rear',1.4]])for(const side of [-1,1]){
  const steerPivot=new T.Group();steerPivot.position.set(side*.92,wheelRadius,z);group.add(steerPivot);const spin=new T.Group();steerPivot.add(spin);
  const tyre=mesh(new T.CylinderGeometry(wheelRadius,wheelRadius,.27,40),rubber,[0,0,0],spin);tyre.rotation.z=Math.PI/2;
  const hub=mesh(new T.CylinderGeometry(.235,.235,.282,32),carbon,[0,0,0],spin);hub.rotation.z=Math.PI/2;
  const rim=mesh(new T.TorusGeometry(.245,.021,8,36),alloy,[side*.148,0,0],spin);rim.rotation.y=Math.PI/2;
  for(let j=0;j<10;j++){const a=j*Math.PI/5;const spoke=box([.018,.038,.23],[side*.154,Math.sin(a)*.12,Math.cos(a)*.12],alloy,spin);spoke.rotation.x=-a;}
  const centre=mesh(new T.CylinderGeometry(.054,.054,.3,12),alloy,[0,0,0],spin);centre.rotation.z=Math.PI/2;
  wheels[row].push({steerPivot,spin});
 }
 group.userData.dispose=()=>{geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());textures.forEach(t=>t.dispose());};
 return {group,wheels,wheelRadius};
}
