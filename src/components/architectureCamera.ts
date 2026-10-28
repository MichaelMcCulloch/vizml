import { MathUtils, OrthographicCamera, PerspectiveCamera, Vector3 } from "three";

export type Projection = "orthographic" | "perspective";
export type ArchitectureCamera = OrthographicCamera | PerspectiveCamera;

/** Vertical world-space span at the orbit target, for matching projection modes. */
export function cameraViewHeight(camera: ArchitectureCamera, target: Vector3) {
  if (camera instanceof OrthographicCamera) return (camera.top - camera.bottom) / camera.zoom;
  return 2 * camera.position.distanceTo(target) * Math.tan(MathUtils.degToRad(camera.fov / 2)) / camera.zoom;
}

export function setCameraViewHeight(camera: ArchitectureCamera, target: Vector3, height: number) {
  if (camera instanceof OrthographicCamera) {
    camera.zoom = (camera.top - camera.bottom) / height;
  } else {
    // Perspective scroll uses dolly distance, so keep its lens at the chosen FOV.
    camera.zoom = 1;
    const direction = camera.position.clone().sub(target).normalize();
    const distance = height / (2 * Math.tan(MathUtils.degToRad(camera.fov / 2)));
    camera.position.copy(target).addScaledVector(direction, distance);
  }
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
}

/** Label sizing and screen-plane offsets must use each label's own depth. */
export function pixelsPerWorldUnit(camera: ArchitectureCamera, point: Vector3, viewportHeight: number) {
  const depth = -point.clone().applyMatrix4(camera.matrixWorldInverse).z;
  if (depth < camera.near || depth > camera.far) return 0;
  const scale = viewportHeight * camera.projectionMatrix.elements[5] / 2;
  return camera instanceof PerspectiveCamera ? scale / depth : scale;
}

/** Fit individual glyph corners, including the extra room nearer corners need. */
export function perspectiveFitDistance(
  camera: PerspectiveCamera,
  target: Vector3,
  corners: readonly Vector3[],
  frame: { x: number; y: number },
) {
  const right = new Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const up = new Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const back = new Vector3().setFromMatrixColumn(camera.matrixWorld, 2);
  const horizontal = camera.projectionMatrix.elements[0] / frame.x;
  const vertical = camera.projectionMatrix.elements[5] / frame.y;
  let distance = 5.2 * camera.projectionMatrix.elements[5];
  for (const corner of corners) {
    const offset = corner.clone().sub(target);
    distance = Math.max(distance, offset.dot(back) + Math.max(
      (Math.abs(offset.dot(right)) + 1) * horizontal,
      (Math.abs(offset.dot(up)) + 1) * vertical,
      camera.near * 2,
    ));
  }
  return distance;
}
