'use client';

import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { sceneState } from '@/lib/scene-state';
import { createPipelineCurve, pipelineStations } from './geometry';

export function Pipeline() {
  const groupRef = useRef<THREE.Group>(null);
  const stationsRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const curve = useMemo(() => createPipelineCurve(), []);

  const tubeGeometry = useMemo(
    () => new THREE.TubeGeometry(curve, 96, 0.028, 8, false),
    [curve],
  );

  useFrame((state) => {
    const snapshot = sceneState.get();
    const visible = snapshot.section === 'approach' ? 1 : snapshot.section === 'work' ? 0.18 : 0.05;
    const group = groupRef.current;
    if (group) {
      group.visible = visible > 0.04;
      group.position.y = THREE.MathUtils.lerp(group.position.y, snapshot.section === 'approach' ? 0 : -0.4, 0.06);
    }

    const mesh = stationsRef.current;
    if (!mesh) return;
    const t = state.clock.elapsedTime;
    pipelineStations.forEach((u, i) => {
      const point = curve.getPointAt(u);
      dummy.position.copy(point);
      const active = snapshot.section === 'approach' && snapshot.techIndex === i;
      const scale = 0.16 + (active ? 0.12 : 0) + Math.sin(t * 2 + i) * 0.02;
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <group ref={groupRef} position={[0, -0.15, -0.4]}>
      <mesh geometry={tubeGeometry}>
        <meshBasicMaterial color="#c8f36a" transparent opacity={0.35} />
      </mesh>
      <mesh geometry={tubeGeometry}>
        <meshBasicMaterial color="#83e4dd" transparent opacity={0.12} wireframe />
      </mesh>
      <instancedMesh ref={stationsRef} args={[undefined, undefined, pipelineStations.length]}>
        <octahedronGeometry args={[1, 0]} />
        <meshStandardMaterial
          color="#c8f36a"
          emissive="#c8f36a"
          emissiveIntensity={1.4}
          roughness={0.25}
          metalness={0.15}
        />
      </instancedMesh>
    </group>
  );
}

export { createPipelineCurve };
