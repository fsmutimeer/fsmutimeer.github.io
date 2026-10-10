"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

type DiagramProps = {
  slug: string;
  active: boolean;
  particleActive?: boolean;
};

type DiagramParticle = {
  x: number;
  y: number;
  red: number;
  green: number;
  blue: number;
  alpha: number;
  size: number;
  offsetX: number;
  offsetY: number;
  delay: number;
};

const particleStyleProperties = [
  "fill",
  "fill-opacity",
  "stroke",
  "stroke-opacity",
  "stroke-width",
  "stroke-dasharray",
  "stroke-dashoffset",
  "stroke-linecap",
  "stroke-linejoin",
  "opacity",
  "font-family",
  "font-size",
  "font-weight",
  "letter-spacing",
  "text-anchor",
  "text-transform",
  "transform",
  "transform-origin",
];

async function sampleDiagram(svg: SVGSVGElement): Promise<DiagramParticle[]> {
  const viewBox = svg.viewBox.baseVal;
  const sourceWidth = Math.max(1, Math.ceil(viewBox.width || 400));
  const sourceHeight = Math.max(1, Math.ceil(viewBox.height || 116));
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(sourceWidth));
  clone.setAttribute("height", String(sourceHeight));

  const originalElements = [svg, ...svg.querySelectorAll("*")];
  const clonedElements = [clone, ...clone.querySelectorAll("*")];
  originalElements.forEach((element, index) => {
    const computed = getComputedStyle(element);
    clonedElements[index]?.setAttribute(
      "style",
      particleStyleProperties
        .map((property) => {
          const value =
            index === 0 && property === "opacity"
              ? "1"
              : computed.getPropertyValue(property);
          return `${property}:${value}`;
        })
        .join(";"),
    );
  });

  const url = URL.createObjectURL(
    new Blob([new XMLSerializer().serializeToString(clone)], {
      type: "image/svg+xml;charset=utf-8",
    }),
  );
  const image = new Image();
  try {
    image.src = url;
    await image.decode();
    const sampler = document.createElement("canvas");
    sampler.width = sourceWidth;
    sampler.height = sourceHeight;
    const context = sampler.getContext("2d", { willReadFrequently: true });
    if (!context) return [];
    context.drawImage(image, 0, 0, sourceWidth, sourceHeight);

    const { data } = context.getImageData(0, 0, sourceWidth, sourceHeight);
    const particles: DiagramParticle[] = [];
    for (let y = 1; y < sourceHeight; y += 3) {
      for (let x = 1; x < sourceWidth; x += 3) {
        const offset = (y * sourceWidth + x) * 4;
        const alpha = data[offset + 3] / 255;
        if (alpha < 0.2) continue;
        particles.push({
          x: x / sourceWidth,
          y: y / sourceHeight,
          red: data[offset],
          green: data[offset + 1],
          blue: data[offset + 2],
          alpha,
          size: 1.2 + Math.random() * 1.1,
          offsetX: (Math.random() - 0.5) * 42,
          offsetY: (Math.random() - 0.5) * 32,
          delay: Math.random() * 0.18,
        });
      }
    }

    const stride = Math.max(1, Math.ceil(particles.length / 480));
    return particles.filter((_, index) => index % stride === 0);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function MigraxDiagram() {
  return (
    <svg viewBox="0 0 400 116" fill="none" aria-hidden="true">
      <text className="work-diagram-label" x="24" y="18">
        @Entity
      </text>
      <text className="work-diagram-label" x="262" y="18">
        SQL + rollback
      </text>
      {[0, 1, 2, 3].map((row) => (
        <g key={row}>
          <rect
            className="work-diagram-bar"
            x="24"
            y={30 + row * 16}
            width={row === 3 ? 86 : 114}
            height="8"
            rx="1"
          />
          <rect
            className={`work-diagram-bar${row === 2 ? " is-mismatch" : ""}`}
            x="262"
            y={30 + row * 16}
            width={row === 2 ? 72 : 114}
            height="8"
            rx="1"
          />
        </g>
      ))}
      <path className="work-diagram-path" d="M146 58h38m32 0h38" />
      <circle className="work-diagram-node" cx="200" cy="58" r="13" />
      <circle className="work-diagram-node-core" cx="200" cy="58" r="4.5" />
      <rect
        className="work-diagram-bar is-mismatch"
        x="262"
        y="101"
        width="10"
        height="6"
        rx="1"
      />
      <text className="work-diagram-label" x="278" y="107">
        lint
      </text>
    </svg>
  );
}

function PipelineDiagram() {
  return (
    <svg viewBox="0 0 400 100" fill="none" aria-hidden="true">
      <path className="work-diagram-path" d="M42 42h316" />
      {[42, 142, 258, 358].map((x, index) => (
        <g key={x}>
          <circle className="work-diagram-node" cx={x} cy="42" r="10" />
          <circle className="work-diagram-node-core" cx={x} cy="42" r="3.5" />
          <text className="work-diagram-label" x={x} y="78" textAnchor="middle">
            {["commit", "build", "scan", "sync"][index]}
          </text>
        </g>
      ))}
    </svg>
  );
}

function DoctorDiagram() {
  return (
    <svg viewBox="0 0 400 116" fill="none" aria-hidden="true">
      <text className="work-diagram-label" x="24" y="18">
        Quarkus config
      </text>
      <text className="work-diagram-label" x="236" y="18">
        K8s manifests
      </text>
      {[0, 1, 2, 3].map((row) => (
        <g key={row}>
          <rect
            className="work-diagram-bar"
            x="24"
            y={28 + row * 16}
            width="140"
            height="8"
            rx="1"
          />
          <rect
            className={`work-diagram-bar${row === 1 || row === 2 ? " is-mismatch" : ""}`}
            x="236"
            y={28 + row * 16}
            width={row === 1 || row === 2 ? 88 : 140}
            height="8"
            rx="1"
          />
        </g>
      ))}
      <rect
        className="work-diagram-bar is-mismatch"
        x="236"
        y="101"
        width="10"
        height="6"
        rx="1"
      />
      <text className="work-diagram-label" x="252" y="107">
        difference
      </text>
    </svg>
  );
}

const diagrams: Record<string, () => ReactNode> = {
  migrax: MigraxDiagram,
  "quarkus-doctor": DoctorDiagram,
  "gitops-tekton": PipelineDiagram,
};

export function WorkDiagram({
  slug,
  active,
  particleActive = false,
}: DiagramProps) {
  const Diagram = diagrams[slug] ?? PipelineDiagram;
  const rootRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<DiagramParticle[] | null>(null);
  const progressRef = useRef(0);
  const particleVisibleRef = useRef(false);
  const [particleVisible, setParticleVisible] = useState(false);

  useEffect(() => {
    const root = rootRef.current;
    const canvas = canvasRef.current;
    const svg = root?.querySelector("svg");
    if (!root || !canvas || !svg) return;

    const finePointer = window.matchMedia("(pointer: fine)").matches;
    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (reducedMotion || !finePointer) {
      particleVisibleRef.current = false;
      setParticleVisible(false);
      return;
    }

    if (!particleActive && !particleVisibleRef.current) return;

    let cancelled = false;
    let frame = 0;
    let resizeObserver: ResizeObserver | undefined;

    const animate = async () => {
      try {
        if (particleActive && !particlesRef.current) {
          particlesRef.current = await sampleDiagram(svg);
        }
        if (cancelled) return;

        const particles = particlesRef.current;
        const context = canvas.getContext("2d");
        if (!particles?.length || !context) {
          particleVisibleRef.current = false;
          setParticleVisible(false);
          return;
        }
        canvas.dataset.particleCount = String(particles.length);

        const dimensions = { width: 0, height: 0 };
        const resizeCanvas = () => {
          const rect = root.getBoundingClientRect();
          const ratio = Math.min(window.devicePixelRatio || 1, 2);
          dimensions.width = rect.width;
          dimensions.height = rect.height;
          canvas.width = Math.max(1, Math.round(rect.width * ratio));
          canvas.height = Math.max(1, Math.round(rect.height * ratio));
          context.setTransform(ratio, 0, 0, ratio, 0, 0);
        };

        const draw = (progress: number) => {
          context.clearRect(0, 0, dimensions.width, dimensions.height);
          for (const particle of particles) {
            const local = Math.max(
              0,
              Math.min(1, (progress - particle.delay) / (1 - particle.delay)),
            );
            const eased = 1 - (1 - local) ** 3;
            const x = particle.x * dimensions.width + particle.offsetX * eased;
            const y = particle.y * dimensions.height + particle.offsetY * eased;
            const alpha = particle.alpha * (1 - eased * 0.24);
            context.fillStyle = `rgb(${particle.red} ${particle.green} ${particle.blue})`;
            context.globalAlpha = alpha;
            context.fillRect(x, y, particle.size, particle.size);
          }
          context.globalAlpha = 1;
        };

        resizeCanvas();
        resizeObserver = new ResizeObserver(() => {
          resizeCanvas();
          draw(progressRef.current);
        });
        resizeObserver.observe(root);

        const from = progressRef.current;
        const to = particleActive ? 1 : 0;
        const start = performance.now();
        const duration = particleActive ? 520 : 420;

        if (particleActive) {
          particleVisibleRef.current = true;
          setParticleVisible(true);
        }

        const tick = (now: number) => {
          if (cancelled) return;
          const linear = Math.min(1, (now - start) / duration);
          const eased = particleActive ? 1 - (1 - linear) ** 3 : linear ** 2;
          const progress = from + (to - from) * eased;
          progressRef.current = progress;
          draw(progress);

          if (linear < 1) {
            frame = requestAnimationFrame(tick);
          } else {
            progressRef.current = to;
            draw(to);
            if (!particleActive) {
              particleVisibleRef.current = false;
              setParticleVisible(false);
            }
          }
        };

        frame = requestAnimationFrame(tick);
      } catch {
        particleVisibleRef.current = false;
        setParticleVisible(false);
      }
    };

    void animate();
    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
      resizeObserver?.disconnect();
    };
  }, [particleActive]);

  return (
    <div
      ref={rootRef}
      className={`work-diagram${active ? " is-active" : ""}${particleVisible ? " is-particle-visible" : ""}`}
      aria-hidden="true"
    >
      <Diagram />
      <canvas
        ref={canvasRef}
        className="work-diagram-particles"
        data-testid={`canvas-project-particles-${slug}`}
      />
    </div>
  );
}
