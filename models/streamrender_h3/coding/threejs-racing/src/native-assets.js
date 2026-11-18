import * as T from 'three';
const palette=await(await fetch('/palette.json')).json();
const mapping=await(await fetch('/torcs/semantic_materials.json')).json();
const loader=new T.TextureLoader(),cache=new Map();
async function texture(kind,name){const key=kind+'/'+name;if(!cache.has(key))cache.set(key,loader.loadAsync('/torcs/textures/'+kind+'/'+name.replace(/\.rgb$/,'.png')).then(t=>{t.colorSpace=T.SRGBColorSpace;t.wrapS=t.wrapT=T.RepeatWrapping;return t}));return cache.get(key);}
export async function loadNative(kind,{semanticOverride=null}={}){
 const data=await(await fetch('/torcs/'+(kind==='track'?'e-track-3':'car1-stock1')+'.json')).json(),group=new T.Group(),batches=new Map();
 for(const m of data.meshes){const label=semanticOverride??mapping[m.texture];if(!label)throw Error('Unmapped TORCS texture: '+m.texture);const key=m.texture+'|'+label;if(!batches.has(key))batches.set(key,{texture:m.texture,label,positions:[],uv:[]});const b=batches.get(key);for(const x of m.positions)b.positions.push(x);for(const x of m.uv)b.uv.push(x);}
 await Promise.all([...batches.values()].map(async b=>{
  const map=await texture(kind,b.texture),geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(b.positions,3));geometry.setAttribute('uv',new T.Float32BufferAttribute(b.uv,2));geometry.computeVertexNormals();
  const alpha=true;
  const material=new T.MeshStandardMaterial({map,roughness:.9,side:T.DoubleSide,alphaTest:alpha?.5:0});const mesh=new T.Mesh(geometry,material);mesh.userData.semantic=b.label;mesh.castShadow=kind==='car';mesh.receiveShadow=true;
  if(alpha)mesh.userData.semanticMaterial=new T.ShaderMaterial({uniforms:{rgb:{value:new T.Vector3(...palette[b.label].map(v=>v/255))},alphaTexture:{value:map}},vertexShader:'varying vec2 tuv; void main(){tuv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',fragmentShader:'varying vec2 tuv;uniform sampler2D alphaTexture;uniform vec3 rgb;void main(){if(texture2D(alphaTexture,tuv).a<0.5)discard;gl_FragColor=vec4(rgb,1.0);}',side:T.DoubleSide,blending:T.NoBlending,toneMapped:false});
  group.add(mesh);
 }));return group;
}
