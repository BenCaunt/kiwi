import * as THREE from "three";

export function renderedMeshBounds(object: THREE.Object3D): THREE.Box3 {
  object.updateMatrixWorld(true);
  const bounds = new THREE.Box3().makeEmpty();
  const point = new THREE.Vector3();

  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const position = child.geometry.getAttribute("position");
    if (!position) return;
    const index = child.geometry.getIndex();
    const count = index?.count ?? position.count;
    for (let offset = 0; offset < count; offset += 1) {
      const vertex = index ? index.getX(offset) : offset;
      point.fromBufferAttribute(position, vertex).applyMatrix4(child.matrixWorld);
      bounds.expandByPoint(point);
    }
  });

  return bounds;
}
