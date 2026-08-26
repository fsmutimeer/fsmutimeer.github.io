'use client';

import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { sceneState } from '@/lib/scene-state';
import { fibonacciSphere, nearestConnections } from './geometry';

type ClusterProps = {
  count: number;
};

export function Cluster({ count }: ClusterProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  const packetsRef = useRef<THREE.Points>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const { positions, connections, packetSeeds } = useMemo(() => {
    const nextPositions = fibonacciSphere(count, 3.35);
    const nextConnections = nearestConnections(nextPositions, 2);
    const seeds = new Float32Array(18);
    for (let i = 0; i < seeds.length; i += 1) seeds[i] = Math.random();
    return { positions: nextPositions, connections: nextConnections, packetSeeds: seeds };
  }, [count]);

  const lineGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(connections, 3));
    return geometry;
  }, [connections]);

  const packetGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(18 * 3), 3));
    return geometry;
  }, []);

  useFrame((state) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const snapshot = sceneState.get();
    const t = state.clock.elapsedTime;

    const workPulse = snapshot.section === 'work' ? snapshot.workIndex : -1;
    const contact = snapshot.section === 'contact' ? 1 : 0;
    const approach = snapshot.section === 'approach' ? 1 : 0;
    const clusterScale = THREE.MathUtils.lerp(1, 0.22, contact * 0.85 + approach * 0.35);

    for (let i = 0; i < count; i += 1) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      const breathe = Math.sin(t * 0.55 + i * 0.37) * 0.06;
      dummy.position.set(x * clusterScale, y * clusterScale + breathe, z * clusterScale);
      const hub = i % 9 === 0;
      const highlighted = workPulse >= 0 && i % 4 === workPulse;
      const scale = (hub ? 0.09 : 0.045) * (highlighted ? 1.7 : 1) * (contact && i === 0 ? 4.2 : 1);
      dummy.scale.setScalar(scale);
      dummy.rotation.set(t * 0.12 + i * 0.01, t * 0.08, 0);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.rotation.y = t * 0.045 + snapshot.progress * 0.55;
    mesh.rotation.x = Math.sin(t * 0.12) * 0.06;

    const traffic = snapshot.reducedMotion
      ? 0
      : THREE.MathUtils.clamp(Math.abs(snapshot.scrollVelocity) / 14, 0, 1);

    if (linesRef.current) {
      linesRef.current.rotation.copy(mesh.rotation);
      const material = linesRef.current.material as THREE.LineBasicMaterial;
      material.opacity = THREE.MathUtils.lerp(0.22, 0.04, contact) + traffic * 0.42;
    }

    const packets = packetsRef.current;
    if (packets) {
      packets.rotation.copy(mesh.rotation);
      const attr = packets.geometry.getAttribute('position') as THREE.BufferAttribute;
      const segmentCount = connections.length / 6;
      const packetMat = packets.material as THREE.PointsMaterial;
      packetMat.opacity = 0.72 + traffic * 0.28;
      packetMat.size = 0.045 + traffic * 0.03;
      for (let i = 0; i < 18; i += 1) {
        const seg = Math.floor(packetSeeds[i] * segmentCount) % Math.max(segmentCount, 1);
        const speed = (0.08 + packetSeeds[i] * 0.12) * (1 + traffic * 3.4);
        const u = (t * speed + packetSeeds[i]) % 1;
        const a = seg * 6;
        attr.setXYZ(
          i,
          THREE.MathUtils.lerp(connections[a], connections[a + 3], u) * clusterScale,
          THREE.MathUtils.lerp(connections[a + 1], connections[a + 4], u) * clusterScale,
          THREE.MathUtils.lerp(connections[a + 2], connections[a + 5], u) * clusterScale,
        );
      }
      attr.needsUpdate = true;
    }
  });

  return (
    <group>
      <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
        <icosahedronGeometry args={[1, 0]} />
        <meshStandardMaterial
          color="#c8f36a"
          emissive="#c8f36a"
          emissiveIntensity={0.85}
          roughness={0.28}
          metalness={0.2}
          transparent
          opacity={0.92}
        />
      </instancedMesh>
      <lineSegments ref={linesRef} geometry={lineGeometry}>
        <lineBasicMaterial color="#83e4dd" transparent opacity={0.2} />
      </lineSegments>
      <points ref={packetsRef} geometry={packetGeometry}>
        <pointsMaterial color="#c8f36a" size={0.045} sizeAttenuation transparent opacity={0.9} />
      </points>
    </group>
  );
}
