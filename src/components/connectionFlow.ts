import * as THREE from "three";

export type FlowProfile = {
  phase: number;
  rate: number;
  width: number;
  strength: number;
};

/** Distance along the drawn path, rather than the curve's uneven parameter t. */
export function pathProgress(points: THREE.Vector3[]) {
  const progress = new Float32Array(points.length);
  let length = 0;
  for (let i = 1; i < points.length; i++) {
    length += points[i].distanceTo(points[i - 1]);
    progress[i] = length;
  }
  if (length > 0)
    for (let i = 1; i < points.length; i++) progress[i] /= length;
  return { progress, length };
}

/** One envelope per connection; every strand shares its timing and intensity. */
export function flowProfile(weight: number, key: string, length: number): FlowProfile {
  const throughput = THREE.MathUtils.clamp(Number.isFinite(weight) ? weight : 0, 0, 1);
  const distance = Math.max(0, Number.isFinite(length) ? length : 0);
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  const width = THREE.MathUtils.clamp(1.2 / Math.max(distance, 0.01), 0.08, 0.3);
  const travel = Math.max(6, distance / (0.45 + 0.15 * throughput));
  return {
    phase: (hash % 1000) / 1000,
    rate: 1 / (travel * (1 + 2 * width)),
    width,
    strength: THREE.MathUtils.smoothstep(throughput, 0, 0.12) * (0.65 + 0.35 * throughput),
  };
}

export type FlowAppearance = {
  profile: FlowProfile;
  color: THREE.ColorRepresentation;
  endColor?: THREE.ColorRepresentation;
  opacity: number;
  surface?: boolean;
};

/** Attach the same flow coordinates to lines, line batches, ribbons, and tubes. */
export function applyFlowAttributes(
  geometry: THREE.BufferGeometry,
  progress: ArrayLike<number>,
  appearance: FlowAppearance,
  across?: ArrayLike<number>,
) {
  const count = progress.length;
  const colors = new Float32Array(count * 3);
  const profiles = new Float32Array(count * 4);
  const opacities = new Float32Array(count * 2);
  const from = new THREE.Color(appearance.color);
  const to = new THREE.Color(appearance.endColor ?? appearance.color);
  const tint = new THREE.Color();
  const { phase, rate, width, strength } = appearance.profile;
  for (let i = 0; i < count; i++) {
    tint.copy(from).lerp(to, progress[i]).toArray(colors, i * 3);
    profiles.set([phase, rate, width, strength], i * 4);
    opacities.set([appearance.opacity, appearance.surface ? 1 : 0], i * 2);
  }
  geometry.setAttribute("aProgress", new THREE.Float32BufferAttribute(progress, 1));
  geometry.setAttribute("aAcross", new THREE.Float32BufferAttribute(
    across ?? new Float32Array(count).fill(0.5), 1,
  ));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("aFlow", new THREE.BufferAttribute(profiles, 4));
  geometry.setAttribute("aAppearance", new THREE.BufferAttribute(opacities, 2));
  return geometry;
}

export function createFlowMaterial(theme: "light" | "dark") {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uHighlight: { value: new THREE.Color(theme === "light" ? "#203047" : "#dce7ef") },
    },
    vertexShader: `
      attribute float aProgress;
      attribute float aAcross;
      attribute vec3 aColor;
      attribute vec4 aFlow;
      attribute vec2 aAppearance;
      varying float vProgress;
      varying float vAcross;
      varying vec3 vColor;
      varying vec4 vFlow;
      varying vec2 vAppearance;
      void main() {
        vProgress = aProgress;
        vAcross = aAcross;
        vColor = aColor;
        vFlow = aFlow;
        vAppearance = aAppearance;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float uTime;
      uniform vec3 uHighlight;
      varying float vProgress;
      varying float vAcross;
      varying vec3 vColor;
      varying vec4 vFlow;
      varying vec2 vAppearance;
      void main() {
        float width = vFlow.z;
        float head = fract(uTime * vFlow.y + vFlow.x) * (1.0 + 2.0 * width) - width;
        float offset = vProgress - head;
        // A long soft tail and a rounded leading edge. The envelope leaves
        // the entire path before wrapping, so there is no flashing seam.
        float sweep = smoothstep(-width, -width * 0.15, offset)
          * (1.0 - smoothstep(0.0, width * 0.5, offset));
        sweep *= vFlow.w;
        float edge = mix(1.0,
          smoothstep(0.0, 0.18, vAcross) * smoothstep(0.0, 0.18, 1.0 - vAcross),
          vAppearance.y);
        float tips = smoothstep(0.0, 0.035, vProgress)
          * smoothstep(0.0, 0.035, 1.0 - vProgress);
        float alpha = vAppearance.x * edge * tips
          * (1.0 + mix(0.7, 1.1, vAppearance.y) * sweep);
        vec3 color = mix(vColor, uHighlight, 0.1 * sweep);
        gl_FragColor = vec4(color, alpha);
        #include <colorspace_fragment>
      }
    `,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.NormalBlending,
  });
}
