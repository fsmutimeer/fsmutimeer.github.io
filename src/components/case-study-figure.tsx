import type { ReactNode } from 'react';

function KafkaFigure() {
  return (
    <svg viewBox="0 0 720 220" fill="none" aria-hidden="true">
      {/* Producers / Core Services */}
      <rect className="study-fig-box" x="24" y="36" width="136" height="52" rx="3" />
      <text className="study-fig-label" x="92" y="58" textAnchor="middle">
        Quarkus Service
      </text>
      <text className="study-fig-label" x="92" y="74" textAnchor="middle" fill="#6f8b82">
        Domain Producer
      </text>

      <rect className="study-fig-box" x="24" y="124" width="136" height="52" rx="3" />
      <text className="study-fig-label" x="92" y="146" textAnchor="middle">
        Integration Service
      </text>
      <text className="study-fig-label" x="92" y="162" textAnchor="middle" fill="#6f8b82">
        Camel · Producer
      </text>

      {/* Central Kafka Event Stream Backbone */}
      <rect className="study-fig-box is-accent" x="224" y="24" width="272" height="162" rx="4" />
      <text className="study-fig-label" x="360" y="44" textAnchor="middle" fill="var(--acid)">
        Kafka Event Backbone
      </text>

      {/* Event Topics / Partition Streams */}
      <rect className="study-fig-box" x="240" y="58" width="240" height="38" rx="2" />
      <text className="study-fig-label" x="252" y="80" textAnchor="start">
        topic: domain.events.v1
      </text>
      <circle className="study-fig-node" cx="430" cy="77" r="7" />
      <circle className="study-fig-core" cx="430" cy="77" r="2.5" />
      <circle className="study-fig-node" cx="456" cy="77" r="7" />
      <circle className="study-fig-core" cx="456" cy="77" r="2.5" />

      <rect className="study-fig-box" x="240" y="110" width="240" height="38" rx="2" />
      <text className="study-fig-label" x="252" y="132" textAnchor="start">
        topic: notifications.v1
      </text>
      <circle className="study-fig-node" cx="430" cy="129" r="7" />
      <circle className="study-fig-core" cx="430" cy="129" r="2.5" />
      <circle className="study-fig-node" cx="456" cy="129" r="7" />
      <circle className="study-fig-core" cx="456" cy="129" r="2.5" />

      <text className="study-fig-label" x="360" y="172" textAnchor="middle" fill="#6f8b82">
        Partitioned Log · Replayable Events
      </text>

      {/* Decoupled Consumers */}
      <rect className="study-fig-box" x="560" y="26" width="136" height="42" rx="2" />
      <text className="study-fig-label" x="628" y="46" textAnchor="middle">
        Notification Service
      </text>
      <text className="study-fig-label" x="628" y="59" textAnchor="middle" fill="#6f8b82">
        Consumer Group
      </text>

      <rect className="study-fig-box" x="560" y="84" width="136" height="42" rx="2" />
      <text className="study-fig-label" x="628" y="104" textAnchor="middle">
        Relay &amp; Webhook
      </text>
      <text className="study-fig-label" x="628" y="117" textAnchor="middle" fill="#6f8b82">
        Consumer Group
      </text>

      <rect className="study-fig-box" x="560" y="142" width="136" height="42" rx="2" />
      <text className="study-fig-label" x="628" y="162" textAnchor="middle">
        Email Dispatch
      </text>
      <text className="study-fig-label" x="628" y="175" textAnchor="middle" fill="#6f8b82">
        Consumer Group
      </text>

      {/* Connecting Flow Paths */}
      <path className="study-fig-path" d="M160 62h40v15h24" />
      <path className="study-fig-path" d="M160 150h40v-21h24" />
      <path className="study-fig-path" d="M480 77h42v-30h38" />
      <path className="study-fig-path" d="M480 105h80" />
      <path className="study-fig-path" d="M480 129h42v34h38" />

      {/* Descriptive subtitle */}
      <text className="study-fig-label" x="360" y="206" textAnchor="middle">
        Decoupled asynchronous event path · Replayable log · Independent consumer groups
      </text>
    </svg>
  );
}

function ClusterFigure() {
  return (
    <svg viewBox="0 0 720 220" fill="none" aria-hidden="true">
      <text className="study-fig-label" x="180" y="28" textAnchor="middle">
        Red Hat OpenShift Container Platform
      </text>
      <text className="study-fig-label" x="540" y="28" textAnchor="middle">
        OKD Community Platform
      </text>
      {[96, 180, 264].map((x) => (
        <rect key={`a-${x}`} className="study-fig-box is-accent" x={x} y="48" width="56" height="40" rx="2" />
      ))}
      {[456, 540, 624].map((x) => (
        <rect key={`b-${x}`} className="study-fig-box is-accent" x={x} y="48" width="56" height="40" rx="2" />
      ))}
      <text className="study-fig-label" x="180" y="108" textAnchor="middle">
        Control Plane (HA)
      </text>
      <text className="study-fig-label" x="540" y="108" textAnchor="middle">
        Control Plane (HA)
      </text>
      <rect className="study-fig-box" x="120" y="128" width="120" height="40" rx="2" />
      <rect className="study-fig-box" x="480" y="128" width="120" height="40" rx="2" />
      <text className="study-fig-label" x="180" y="188" textAnchor="middle">
        Worker Node Pool · KVM
      </text>
      <text className="study-fig-label" x="540" y="188" textAnchor="middle">
        Worker Node Pool · KVM
      </text>
      <path className="study-fig-path" d="M180 88v40" />
      <path className="study-fig-path" d="M540 88v40" />
    </svg>
  );
}

function GitOpsFigure() {
  return (
    <svg viewBox="0 0 720 220" fill="none" aria-hidden="true">
      {[
        ['Git push', 48],
        ['Tekton', 196],
        ['Trivy', 344],
        ['Argo CD', 492],
        ['OpenShift', 640],
      ].map(([label, x]) => (
        <g key={label}>
          <circle className="study-fig-node" cx={x as number} cy="88" r="16" />
          <circle className="study-fig-core" cx={x as number} cy="88" r="5" />
          <text className="study-fig-label" x={x as number} y="128" textAnchor="middle">
            {label}
          </text>
        </g>
      ))}
      <path className="study-fig-path" d="M64 88h116m36 0h96m36 0h96m36 0h96" />
      <text className="study-fig-label" x="270" y="176">
        Helm values per environment · App-of-Apps · Wazuh on the cluster
      </text>
    </svg>
  );
}

function DoctorFigure() {
  return (
    <svg viewBox="0 0 720 220" fill="none" aria-hidden="true">
      <text className="study-fig-label" x="130" y="28">
        application.properties / yaml
      </text>
      <rect className="study-fig-box" x="24" y="42" width="220" height="64" rx="2" />
      <text className="study-fig-label" x="500" y="28">
        Deployment · Helm · Kustomize
      </text>
      <rect className="study-fig-box" x="430" y="42" width="266" height="64" rx="2" />
      <rect className="study-fig-box is-accent" x="210" y="128" width="300" height="48" rx="2" />
      <text className="study-fig-label" x="360" y="158" textAnchor="middle">
        quarkus-doctor · Maven verify · no cluster
      </text>
      <path className="study-fig-path" d="M134 106v46h76" />
      <path className="study-fig-path" d="M563 106v46h-53" />
      <text className="study-fig-label" x="360" y="204" textAnchor="middle">
        red build or JSON report — not a CrashLoop
      </text>
    </svg>
  );
}

const figures: Record<string, () => ReactNode> = {
  'quarkus-kafka': KafkaFigure,
  'openshift-okd': ClusterFigure,
  'gitops-tekton': GitOpsFigure,
  'quarkus-doctor': DoctorFigure,
};

export function CaseStudyFigure({ slug, caption }: { slug: string; caption: string }) {
  const Figure = figures[slug];
  if (!Figure) return null;
  return (
    <figure className="study-figure">
      <Figure />
      <figcaption className="mono">{caption}</figcaption>
    </figure>
  );
}
