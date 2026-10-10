import type { ReactNode } from 'react';

function MigraxFigure() {
  return (
    <svg viewBox="0 0 720 220" fill="none" aria-hidden="true">
      <rect className="study-fig-box" x="24" y="40" width="168" height="64" rx="2" />
      <text className="study-fig-label" x="108" y="68" textAnchor="middle">
        JPA entities
      </text>
      <text className="study-fig-label" x="108" y="86" textAnchor="middle" fill="#6f8b82">
        Hibernate mapping
      </text>

      <rect className="study-fig-box is-accent" x="262" y="40" width="196" height="64" rx="2" />
      <text className="study-fig-label" x="360" y="68" textAnchor="middle" fill="var(--acid)">
        migrax generate
      </text>
      <text className="study-fig-label" x="360" y="86" textAnchor="middle" fill="#6f8b82">
        diff · lint
      </text>

      <rect className="study-fig-box" x="528" y="40" width="168" height="64" rx="2" />
      <text className="study-fig-label" x="612" y="68" textAnchor="middle">
        SQL + rollback
      </text>
      <text className="study-fig-label" x="612" y="86" textAnchor="middle" fill="#6f8b82">
        review · edit
      </text>

      <path className="study-fig-path" d="M192 72h70m196 0h70" />
      <path className="study-fig-path" d="M612 104v40H490" />

      <rect className="study-fig-box" x="230" y="124" width="260" height="40" rx="2" />
      <text className="study-fig-label" x="360" y="149" textAnchor="middle">
        verify → migrate → drift
      </text>

      <text className="study-fig-label" x="360" y="200" textAnchor="middle">
        Locks, failing rows, and breaking changes are flagged before production
      </text>
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
  migrax: MigraxFigure,
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
