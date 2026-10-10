import * as T from 'three';
export class SemanticView {
  constructor(scene,camera,palette,width,height){
    this.scene=scene;this.camera=camera;this.palette=palette;
    this.renderer=new T.WebGLRenderer({antialias:false,alpha:false,preserveDrawingBuffer:true});
    this.renderer.setPixelRatio(1);this.renderer.setSize(width,height,false);
    this.renderer.toneMapping=T.NoToneMapping;
    this.renderer.outputColorSpace=T.LinearSRGBColorSpace;
    this.materials=Object.fromEntries(Object.entries(palette).map(([key,rgb])=>[key,new T.ShaderMaterial({
      uniforms:{rgb:{value:new T.Vector3(...rgb.map(v=>v/255))}},
      vertexShader:'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
      fragmentShader:'uniform vec3 rgb; void main(){gl_FragColor=vec4(rgb,1.0);}',
      side:T.DoubleSide,blending:T.NoBlending,depthTest:true,depthWrite:true,toneMapped:false,dithering:false
    })]));
    this.renderer.setClearColor(new T.Color().setRGB(...palette.sky.map(v=>v/255),T.LinearSRGBColorSpace),1);
  }
  render(){
    const changed=[],background=this.scene.background,fog=this.scene.fog;
    this.scene.background=null;this.scene.fog=null;
    this.scene.traverse(o=>{if(!o.isMesh)return; let a=o;while(a&&!a.userData.semantic)a=a.parent;const label=a?.userData.semantic??'unlabeled';if(!this.materials[label])throw Error(`Unknown semantic ${label}`);changed.push([o,o.material]);o.material=o.userData.semanticMaterial??this.materials[label];});
    try{this.renderer.render(this.scene,this.camera);}finally{for(const [o,m] of changed)o.material=m;this.scene.background=background;this.scene.fog=fog;}
  }
  get canvas(){return this.renderer.domElement;}
  async png(){return new Promise((resolve,reject)=>this.canvas.toBlob(b=>b?resolve(b):reject(Error('PNG encoding failed')),'image/png'));}
}
