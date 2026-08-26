export const glowVertex = /* glsl */ `
varying vec3 vNormal;
void main() {
  vNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const glowFragment = /* glsl */ `
precision highp float;
uniform vec3 uColor;
uniform float uBoost;
varying vec3 vNormal;

void main() {
  float fresnel = pow(1.0 - abs(vNormal.z), 2.4);
  vec3 color = uColor * (0.25 + fresnel * uBoost);
  gl_FragColor = vec4(color, 0.22 + fresnel * 0.35);
}
`;
