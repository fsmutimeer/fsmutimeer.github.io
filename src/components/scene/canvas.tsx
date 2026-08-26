'use client';

import { useEffect, useMemo, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Sparkles } from '@react-three/drei';
import * as THREE from 'three';
import { sceneState } from '@/lib/scene-state';
import { Cluster } from './cluster';
import { Pipeline, createPipelineCurve } from './pipeline';
import { grainFragment, grainVertex } from './shaders/grain';
import { glowFragment, glowVertex } from './shaders/glow';

function CameraRig() {
  const { camera } = useThree();
  const curve = useMemo(() => createPipelineCurve(), []);
  const look = useMemo(() => new THREE.Vector3(), []);
  const target = useMemo(() => new THREE.Vector3(), []);
  const desired = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const snapshot = sceneState.get();
    const { progress, section, mouseX, mouseY } = snapshot;

    if (section === 'approach') {
      const u = THREE.MathUtils.clamp((progress - 0.55) / 0.22, 0, 1);
      const point = curve.getPointAt(u);
      const ahead = curve.getPointAt(Math.min(u + 0.08, 1));
      desired.set(point.x, point.y + 1.15, point.z + 3.4);
      look.copy(ahead);
    } else if (section === 'contact') {
      desired.set(0.15, 0.2, 3.55);
      look.set(0, 0, 0);
    } else if (section === 'work') {
      desired.set(1.35, 0.45, 5.7);
      look.set(0, 0.1, 0);
    } else if (section === 'about') {
      desired.set(-5.2, 1.12, 6.9);
      look.set(1.15, 0.08, 0);
    } else {
      const mobile = snapshot.isMobile;
      desired.set(mobile ? 0.15 : 0.1, mobile ? 1.55 : 1.32, mobile ? 9.4 : 9.05);
      look.set(mobile ? 0.1 : -2.2, 0.12, 0);
    }

    desired.x += mouseX * 0.55;
    desired.y += mouseY * 0.28;
    camera.position.lerp(desired, 0.055);
    target.lerp(look, 0.06);
    camera.lookAt(target);
  });

  return null;
}

function Lights() {
  const keyRef = useMemo(() => ({ current: null as THREE.PointLight | null }), []);

  useFrame((state) => {
    const snapshot = sceneState.get();
    const light = keyRef.current;
    if (!light) return;
    const pulse = 1.6 + Math.sin(state.clock.elapsedTime * 1.4) * 0.25;
    light.intensity = snapshot.section === 'contact' ? 3.2 : pulse;
  });

  return (
    <>
      <ambientLight intensity={0.18} />
      <pointLight
        ref={(node) => {
          keyRef.current = node;
        }}
        position={[2.4, 3.2, 4.6]}
        color="#c8f36a"
        intensity={1.8}
        distance={24}
      />
      <pointLight position={[-4.2, -1.4, 2.2]} color="#83e4dd" intensity={1.15} distance={18} />
      <pointLight position={[0, 4.5, -3]} color="#ffc878" intensity={0.35} distance={20} />
    </>
  );
}

function CoreNode() {
  const meshRef = useMemo(() => ({ current: null as THREE.Mesh | null }), []);

  useFrame((state) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const snapshot = sceneState.get();
    const t = state.clock.elapsedTime;
    const contact = snapshot.section === 'contact' ? 1 : 0;
    const scale = THREE.MathUtils.lerp(0.22, 0.82, contact);
    mesh.scale.setScalar(scale + Math.sin(t * 2.1) * 0.03);
    mesh.rotation.y = t * 0.35;
    mesh.rotation.z = t * 0.12;
    const material = mesh.material as THREE.MeshStandardMaterial;
    material.emissiveIntensity = 1.2 + contact * 1.6 + Math.sin(t * 3) * 0.2;
  });

  return (
    <mesh
      ref={(node) => {
        meshRef.current = node;
      }}
    >
      <icosahedronGeometry args={[1, 1]} />
      <meshStandardMaterial
        color="#c8f36a"
        emissive="#c8f36a"
        emissiveIntensity={1.3}
        roughness={0.2}
        metalness={0.25}
        wireframe
      />
    </mesh>
  );
}

function CoreGlow() {
  const uniforms = useMemo(
    () => ({
      uColor: { value: new THREE.Color('#c8f36a') },
      uBoost: { value: 1.8 },
    }),
    [],
  );

  useFrame((state) => {
    uniforms.uBoost.value = 1.6 + Math.sin(state.clock.elapsedTime * 2.2) * 0.4;
  });

  return (
    <mesh scale={1.18}>
      <icosahedronGeometry args={[1, 1]} />
      <shaderMaterial
        uniforms={uniforms}
        vertexShader={glowVertex}
        fragmentShader={glowFragment}
        transparent
        depthWrite={false}
      />
    </mesh>
  );
}

function GrainPass({ enabled }: { enabled: boolean }) {
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uIntensity: { value: 0.09 },
    }),
    [],
  );

  useFrame((state) => {
    uniforms.uTime.value = state.clock.elapsedTime;
  });

  if (!enabled) return null;

  return (
    <mesh frustumCulled={false} renderOrder={1000}>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        uniforms={uniforms}
        vertexShader={grainVertex}
        fragmentShader={grainFragment}
        transparent
        depthTest={false}
        depthWrite={false}
        fog={false}
      />
    </mesh>
  );
}

function FogRig() {
  const { scene } = useThree();
  useEffect(() => {
    scene.fog = new THREE.Fog('#050807', 7, 22);
    scene.background = new THREE.Color('#050807');
  }, [scene]);
  return null;
}

export function SceneCanvas() {
  const [paused, setPaused] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const mobile = window.matchMedia('(max-width: 760px)').matches;
    setIsMobile(mobile);
    sceneState.set({ isMobile: mobile });

    const onVisibility = () => setPaused(document.hidden);
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);

  const nodeCount = isMobile ? 40 : 96;

  return (
    <div className="scene-root" aria-hidden="true">
      <Canvas
        dpr={isMobile ? [1, 1.25] : [1, 1.75]}
        gl={{
          antialias: !isMobile,
          alpha: false,
          powerPreference: 'high-performance',
          stencil: false,
        }}
        camera={{ fov: 42, near: 0.1, far: 80, position: [0.45, 1.32, 8.85] }}
        frameloop={paused ? 'never' : 'always'}
      >
        <FogRig />
        <Lights />
        <CoreGlow />
        <CoreNode />
        <Cluster count={nodeCount} />
        <Pipeline />
        <Sparkles
          count={isMobile ? 18 : 48}
          scale={7.4}
          size={isMobile ? 1.4 : 2.2}
          speed={0.35}
          color="#c8f36a"
          opacity={0.45}
        />
        <CameraRig />
        <GrainPass enabled={!isMobile} />
      </Canvas>
    </div>
  );
}