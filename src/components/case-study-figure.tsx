import type { ReactNode } from 'react';

function KafkaFigure() {
  return (
    <svg viewBox="0 0 720 220" fill="none" aria-hidden="true">
      <text className="study-fig-label" x="70" y="28" textAnchor="middle">
        Quarkus module
      </text>
      <rect className="study-fig-box" x="18" y="42" width="104" height="44" rx="2" />
      <text className="study-fig-label" x="360" y="28" textAnchor="middle">
        Kafka · Camel
      </text>
      <rect className="study-fig-box is-accent" x="292" y="42" width="136" height="44" rx="2" />
      <text className="study-fig-label" x="620" y="22" textAnchor="middle">
        notification
      </text>
      <text className="study-fig-label" x="620" y="78" textAnchor="middle">
        relay · email
      </text>
      <rect className="study-fig-box" x="558" y="32" width="124" height="28" rx="2" />
      <rect className="study-fig-box" x="558" y="88" width="124" height="28" rx="2" />
      <path className="study-fig-path" d="M122 64h170" />
      <path className="study-fig-path" d="M428 64h130" />
      <path className="study-fig-path" d="M558 102h-80v-38" />
      <text className="study-fig-label" x="70" y="148" textAnchor="middle">
        Keycloak RBAC
      </text>
      <rect className="study-fig-box" x="18" y="158" width="104" height="36" rx="2" />
      <text className="study-fig-label" x="360" y="148" textAnchor="middle">
        MongoDB
      </text>
      <rect className="study-fig-box" x="292" y="158" width="136" height="36" rx="2" />
      <path className="study-fig-path" d="M70 86v72" />
      <path className="study-fig-path" d="M360 86v72" />
    </svg>
  );
}

function ClusterFigure() {
  return (
    <svg viewBox="0 0 720 220" fill="none" aria-hidden="true">
      <text className="study-fig-label" x="180" y="28" textAnchor="middle">
        OpenShift · Assisted Installer
      </text>
      <text className="study-fig-label" x="540" y="28" textAnchor="middle">
        OKD · KVM
      </text>
      {[96, 180, 264].map((x) => (
        <rect key={`a-${x}`} className="study-fig-box is-accent" x={x} y="48" width="56" height="40" rx="2" />
      ))}
      {[456, 540, 624].map((x) => (
        <rect key={`b-${x}`} className="study-fig-box is-accent" x={x} y="48" width="56" height="40" rx="2" />
      ))}
      <text className="study-fig-label" x="180" y="108" textAnchor="middle">
        3 control
      </text>
      <text className="study-fig-label" x="540" y="108" textAnchor="middle">
        3 control
      </text>
      <rect className="study-fig-box" x="128" y="128" width="104" height="40" rx="2" />
      <rect className="study-fig-box" x="488" y="128" width="104" height="40" rx="2" />
      <text className="study-fig-label" x="180" y="188" textAnchor="middle">
        1 worker · on-prem KVM
      </text>
      <text className="study-fig-label" x="540" y="188" textAnchor="middle">
        1 worker · on-prem KVM
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
