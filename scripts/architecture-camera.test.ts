import { expect, test } from "bun:test";
import { OrthographicCamera, PerspectiveCamera, Vector3 } from "three";
import { cameraViewHeight, setCameraViewHeight, pixelsPerWorldUnit, perspectiveFitDistance } from "../src/components/architectureCamera";

test("switching projections preserves the target-plane framing and orbit direction", () => {
  const target = new Vector3(18, -4, 6);
  const orthographic = new OrthographicCamera(-24, 24, 12, -12, 0.1, 1000);
  orthographic.zoom = 2.4;
  orthographic.position.copy(target).add(new Vector3(18, 12, 100));
  orthographic.lookAt(target);
  orthographic.updateProjectionMatrix();
  orthographic.updateMatrixWorld();
  const perspective = new PerspectiveCamera(45, 2, 0.1, 1000);
  perspective.position.copy(orthographic.position);
  perspective.quaternion.copy(orthographic.quaternion);
  setCameraViewHeight(perspective, target, cameraViewHeight(orthographic, target));

  const right = new Vector3().setFromMatrixColumn(orthographic.matrixWorld, 0);
  const up = new Vector3().setFromMatrixColumn(orthographic.matrixWorld, 1);
  for (const point of [target, target.clone().addScaledVector(right, 3), target.clone().addScaledVector(up, 2)]) {
    const a = point.clone().project(orthographic), b = point.clone().project(perspective);
    expect(b.x).toBeCloseTo(a.x, 10);
    expect(b.y).toBeCloseTo(a.y, 10);
  }
  expect(perspective.getWorldDirection(new Vector3()).distanceTo(orthographic.getWorldDirection(new Vector3()))).toBeLessThan(1e-10);
  expect(cameraViewHeight(perspective, target)).toBeCloseTo(10, 10);

  // Dolly closer in perspective, then return to orthographic without a zoom jump.
  perspective.position.lerp(target, 0.5);
  setCameraViewHeight(orthographic, target, cameraViewHeight(perspective, target));
  expect(cameraViewHeight(orthographic, target)).toBeCloseTo(5, 10);
});

test("perspective fitting contains near and distant glyph corners at wide and narrow aspects", () => {
  const target = new Vector3(20, -5, 8);
  for (const aspect of [0.35, 1, 2.5]) {
    const camera = new PerspectiveCamera(45, aspect, 0.1, 1000);
    camera.position.copy(target).add(new Vector3(20, 12, 100));
    camera.lookAt(target);
    camera.updateMatrixWorld();
    const corners: Vector3[] = [];
    for (const x of [-24, 24]) for (const y of [-10, 10]) for (const z of [-18, 18]) {
      corners.push(target.clone().add(new Vector3(x, y, z)));
    }
    const frame = { x: 0.9, y: 0.8 };
    const distance = perspectiveFitDistance(camera, target, corners, frame);
    const back = camera.position.clone().sub(target).normalize();
    camera.position.copy(target).addScaledVector(back, distance);
    camera.updateMatrixWorld();
    for (const corner of corners) {
      const screen = corner.clone().project(camera);
      expect(Math.abs(screen.x)).toBeLessThan(frame.x);
      expect(Math.abs(screen.y)).toBeLessThan(frame.y);
      expect(screen.z).toBeGreaterThan(-1);
      expect(screen.z).toBeLessThan(1);
    }
  }
});

test("perspective labels measure screen pixels at their own depth and reject clipped points", () => {
  const camera = new PerspectiveCamera(45, 2, 0.1, 1000);
  camera.position.set(0, 0, 10);
  camera.updateMatrixWorld();
  const near = new Vector3(0, 0, 5), far = new Vector3(0, 0, -5);
  const nearScale = pixelsPerWorldUnit(camera, near, 600), farScale = pixelsPerWorldUnit(camera, far, 600);
  expect(nearScale / farScale).toBeCloseTo(3, 10);
  for (const point of [near, far]) {
    const scale = pixelsPerWorldUnit(camera, point, 600);
    const top = point.clone().add(new Vector3(0, 52 / scale, 0)).project(camera);
    const bottom = point.clone().project(camera);
    expect((top.y - bottom.y) * 300).toBeCloseTo(52, 10);
  }
  expect(pixelsPerWorldUnit(camera, new Vector3(0, 0, 11), 600)).toBe(0);
  expect(pixelsPerWorldUnit(camera, new Vector3(0, 0, 9.99), 600)).toBe(0);
  expect(pixelsPerWorldUnit(camera, new Vector3(0, 0, -1001), 600)).toBe(0);
});

test("orthographic labels keep the same pixel scale at different depths", () => {
  const camera = new OrthographicCamera(-20, 20, 10, -10, 0.1, 1000);
  camera.zoom = 2;
  camera.position.set(0, 0, 10);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  expect(pixelsPerWorldUnit(camera, new Vector3(0, 0, 5), 600)).toBe(60);
  expect(pixelsPerWorldUnit(camera, new Vector3(0, 0, -5), 600)).toBe(60);
});
