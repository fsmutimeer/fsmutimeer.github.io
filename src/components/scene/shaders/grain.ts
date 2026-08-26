export const grainVertex = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const grainFragment = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform float uTime;
uniform float uIntensity;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
  float n = hash(vUv * vec2(1640.0, 920.0) + floor(uTime * 24.0));
  float grain = (n - 0.5) * uIntensity;
  gl_FragColor = vec4(vec3(1.0), abs(grain));
}
`;
