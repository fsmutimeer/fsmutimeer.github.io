import * as THREE from 'three';

export function fibonacciSphere(count: number, radius: number): Float32Array {
  const positions = new Float32Array(count * 3);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i += 1) {
    const y = count === 1 ? 0 : 1 - (i / (count - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    positions[i * 3] = Math.cos(theta) * r * radius;
    positions[i * 3 + 1] = y * radius;
    positions[i * 3 + 2] = Math.sin(theta) * r * radius;
  }
  return positions;
}

export function nearestConnections(positions: Float32Array, neighbors = 2): Float32Array {
  const count = positions.length / 3;
  const pairs = new Set<string>();
  const segments: number[] = [];

  for (let i = 0; i < count; i += 1) {
    const ix = positions[i * 3];
    const iy = positions[i * 3 + 1];
    const iz = positions[i * 3 + 2];
    const distances: { j: number; d: number }[] = [];
    for (let j = 0; j < count; j += 1) {
      if (i === j) continue;
      const dx = ix - positions[j * 3];
      const dy = iy - positions[j * 3 + 1];
      const dz = iz - positions[j * 3 + 2];
      distances.push({ j, d: dx * dx + dy * dy + dz * dz });
    }
    distances.sort((a, b) => a.d - b.d);
    for (let n = 0; n < neighbors; n += 1) {
      const j = distances[n]?.j;
      if (j === undefined) continue;
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (pairs.has(key)) continue;
      pairs.add(key);
      segments.push(
        ix,
        iy,
        iz,
        positions[j * 3],
        positions[j * 3 + 1],
        positions[j * 3 + 2],
      );
    }
  }

  return new Float32Array(segments);
}

export function createPipelineCurve() {
  return new THREE.CatmullRomCurve3(
    [
      new THREE.Vector3(-7.4, -0.2, -1.8),
      new THREE.Vector3(-3.6, 0.55, 0.4),
      new THREE.Vector3(-0.2, 0.1, 1.6),
      new THREE.Vector3(3.4, -0.35, 0.2),
      new THREE.Vector3(7.2, 0.4, -1.4),
    ],
    false,
    'catmullrom',
    0.45,
  );
}

export const pipelineStations = [0.12, 0.38, 0.64, 0.88] as const;
