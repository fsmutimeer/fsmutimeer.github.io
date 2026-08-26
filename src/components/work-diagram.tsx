'use client';

import type { ReactNode } from 'react';

type DiagramProps = {
  number: string;
  active: boolean;
};

function KafkaDiagram() {
  return (
    <svg viewBox="0 0 400 96" fill="none" aria-hidden="true">
      <path className="work-diagram-path" d="M52 48h296" />
      {[52, 150, 250, 348].map((x, index) => (
        <g key={x}>
          <circle className="work-diagram-node" cx={x} cy="48" r="11" />
          <circle className="work-diagram-node-core" cx={x} cy="48" r="4" />
          {index < 3 && (
            <polygon
              className="work-diagram-packet"
              style={{ animationDelay: `${index * 0.55}s` }}
              points={`${x + 28},44 ${x + 40},48 ${x + 28},52`}
            />
          )}
        </g>
      ))}
    </svg>
  );
}

function ClusterDiagram() {
  return (
    <svg viewBox="0 0 400 96" fill="none" aria-hidden="true">
      {[88, 200, 312].map((x) => (
        <rect key={x} className="work-diagram-box" x={x - 22} y="14" width="44" height="36" rx="2" />
      ))}
      <path className="work-diagram-path" d="M88 50v14h112m0-14v14h112" />
      <rect className="work-diagram-worker" x="156" y="64" width="88" height="22" rx="2" />
    </svg>
  );
}

function PipelineDiagram() {
  return (
    <svg viewBox="0 0 400 96" fill="none" aria-hidden="true">
      <path className="work-diagram-path" d="M40 48h320" />
      <circle className="work-diagram-node" cx="48" cy="48" r="10" />
      <circle className="work-diagram-node-core" cx="48" cy="48" r="3.5" />
      <polygon className="work-diagram-gate" points="200,28 224,48 200,68 176,48" />
      <circle className="work-diagram-node" cx="352" cy="48" r="10" />
      <circle className="work-diagram-node-core" cx="352" cy="48" r="3.5" />
      <text className="work-diagram-label" x="48" y="84" textAnchor="middle">
        commit
      </text>
      <text className="work-diagram-label" x="200" y="84" textAnchor="middle">
        scan
      </text>
      <text className="work-diagram-label" x="352" y="84" textAnchor="middle">
        sync
      </text>
    </svg>
  );
}

function DoctorDiagram() {
  return (
    <svg viewBox="0 0 400 96" fill="none" aria-hidden="true">
      <text className="work-diagram-label" x="70" y="18">
        application.properties
      </text>
      <text className="work-diagram-label" x="250" y="18">
        Deployment.yaml
      </text>
      {[0, 1, 2, 3].map((row) => (
        <g key={row}>
          <rect className="work-diagram-bar" x="24" y={28 + row * 16} width="140" height="8" rx="1" />
          <rect
            className={`work-diagram-bar${row === 1 || row === 2 ? ' is-mismatch' : ''}`}
            x="236"
            y={28 + row * 16}
            width={row === 1 ? 88 : 140}
            height="8"
            rx="1"
            style={{ animationDelay: `${row * 0.28}s` }}
          />
        </g>
      ))}
    </svg>
  );
}

const diagrams: Record<string, () => ReactNode> = {
  '01': KafkaDiagram,
  '02': ClusterDiagram,
  '03': PipelineDiagram,
  '04': DoctorDiagram,
};

export function WorkDiagram({ number, active }: DiagramProps) {
  const Diagram = diagrams[number] ?? KafkaDiagram;
  return (
    <div className={`work-diagram${active ? ' is-active' : ''}`} aria-hidden="true">
      <Diagram />
    </div>
  );
}
