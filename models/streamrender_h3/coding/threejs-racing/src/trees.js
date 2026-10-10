import * as T from 'three';
import silhouette from './tree-silhouette.json';

// Opaque triangle geometry traced from the vegetation class in an actual
// TORCS semantic input. Empty spaces contain no triangles; no alpha texture.
function treeGeometry() {
  const positions=[],colors=[];
  const {width,height,runs}=silhouette;
  for(const [row,left,right] of runs){
    const x0=(left/width-.5)*width/height,x1=(right/width-.5)*width/height;
    const y0=1-(row+1)/height,y1=1-row/height;
    const hash=((Math.imul(row+11,73856093)^Math.imul(left+7,19349663))>>>0)/4294967296;
    const bark=row>height*.82 && Math.abs((left+right)/2-width*.5)<width*.08;
    const color=new T.Color(bark?0x62513b:0x53693b).multiplyScalar(.70+hash*.5);
    for(const [x,y] of [[x0,y0],[x1,y0],[x1,y1],[x0,y0],[x1,y1],[x0,y1]]){
      positions.push(x,y,0);colors.push(color.r,color.g,color.b);
    }
  }
  const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(positions,3));g.setAttribute('color',new T.Float32BufferAttribute(colors,3));g.computeVertexNormals();return g;
}
export function addTorcsTrees(scene,samples,heightAt=()=>0){
  const geometry=treeGeometry();
  const material=new T.MeshBasicMaterial({vertexColors:true,side:T.DoubleSide});
  const trees=[];
  // Sparse roadside planting; heights and gaps vary deterministically.
  for(let i=9;i<800;i+=43){
    const {point,normal}=samples[i];
    for(const side of [-1,1]){
      if(side===1 && i%3===0)continue;
      const offset=27+(i%7)*2.1;
      const tree=new T.Mesh(geometry,material);
      const x=point.x+normal.x*side*offset,z=point.z+normal.z*side*offset;tree.position.set(x,heightAt(x,z),z);
      const height=6.8+(i%5)*.48;
      tree.scale.set(height*(side===1?.92:1),height,height);
      tree.userData.semantic='vegetation';tree.userData.torcsTree=true;
      scene.add(tree);trees.push(tree);
    }
  }
  // A camera-facing opaque silhouette preserves training-frame morphology.
  // Update once before both RGB and semantic passes so their masks agree.
  return camera=>{for(const tree of trees)tree.rotation.y=Math.atan2(camera.position.x-tree.position.x,camera.position.z-tree.position.z);};
}
