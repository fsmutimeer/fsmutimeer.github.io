'use client';

import { lifecycle, principles, stack, technologies, type Technology } from '@/lib/content';

export function Approach({
  selectedTechnology,
  selectedStage,
  onSelectTechnology,
  onSelectStage,
}: {
  selectedTechnology: Technology;
  selectedStage: number;
  onSelectTechnology: (technology: Technology) => void;
  onSelectStage: (index: number) => void;
}) {
  return (
    <section className="section approach" id="approach" aria-labelledby="approach-heading">
      <div className="container">
        <div className="split">
          <div>
            <div className="section-label mono">03 / the capability story</div>
            <h2 className="section-title" id="approach-heading">
              The path, not the logo wall.
            </h2>
            <p className="section-intro">
              Work is what shipped. This is how it moves: a service boundary, a Git commit, a
              scanned image, a pod the cluster will admit.
            </p>
          </div>
          <div className="stack-list" data-testid="list-capabilities">
            {stack.map(({ number, title, copy, Icon }) => (
              <div className="stack-item" key={number} data-testid={`row-capability-${number}`}>
                <span className="stack-no mono">{number}</span>
                <div>
                  <h3>{title}</h3>
                  <p>{copy}</p>
                </div>
                <Icon className="stack-arrow" size={18} aria-hidden="true" />
              </div>
            ))}
          </div>
        </div>
        <div className="principles">
          {principles.map(({ title, copy, Icon }, index) => {
            const testId = ['platform', 'reliability', 'practical'][index];
            return (
            <div
              className="principle"
              key={title}
              data-testid={`card-principle-${testId}`}
            >
              <Icon className="principle-icon" size={22} aria-hidden="true" />
              <h3>{title}</h3>
              <p>{copy}</p>
              <span className="principle-index mono">{String(index + 1).padStart(2, '0')}</span>
            </div>
            );
          })}
        </div>
        <div className="platform-spine" id="platform" aria-labelledby="platform-heading">
          <div className="platform-spine-head">
            <div>
              <div className="section-label mono">04 / platform spine</div>
              <h2 className="section-title" id="platform-heading">
                From commit to a signal you can trust.
              </h2>
            </div>
            <p className="section-intro">
              The tools matter. The handoffs between them matter more. Pick a piece, then walk the
              life cycle.
            </p>
          </div>
          <div className="technology-explorer">
            <div className="technology-logos" role="list" aria-label="Platform technologies">
              {technologies.map((technology) => {
                const { name, label, copy, Icon } = technology;
                return (
                  <button
                    className={`technology-card ${selectedTechnology.name === name ? 'is-active' : ''}`}
                    type="button"
                    key={name}
                    role="listitem"
                    aria-pressed={selectedTechnology.name === name}
                    data-testid={`button-technology-${name.toLowerCase()}`}
                    data-cursor="hover"
                    onClick={() => onSelectTechnology(technology)}
                  >
                    <Icon className="technology-icon" aria-hidden="true" />
                    <span className="technology-name">{name}</span>
                    <span className="technology-label mono">{label}</span>
                    <span className="technology-copy">{copy}</span>
                  </button>
                );
              })}
            </div>
            <div className="technology-detail" aria-live="polite" data-testid="panel-technology-detail">
              <div className="technology-detail-top">
                <span className="mono">{selectedTechnology.label}</span>
                <span className="signal-pulse" aria-hidden="true" />
              </div>
              <h3>{`${selectedTechnology.name}.`}</h3>
              <p>{selectedTechnology.detail}</p>
              <span className="technology-detail-route mono">
                /platform/{selectedTechnology.name.toLowerCase()}
              </span>
            </div>
          </div>
          <div className="lifecycle-explorer">
            <div className="lifecycle-heading">
              <div className="section-label mono">software life cycle</div>
              <span className="mono lifecycle-status">
                <span /> pipeline healthy
              </span>
            </div>
            <div className="lifecycle-steps" role="tablist" aria-label="Software lifecycle stages">
              {lifecycle.map(({ number, name, Icon }, index) => (
                <button
                  className={`lifecycle-step ${selectedStage === index ? 'is-active' : ''}`}
                  type="button"
                  role="tab"
                  aria-selected={selectedStage === index}
                  aria-controls={`lifecycle-panel-${number}`}
                  key={number}
                  data-testid={`button-lifecycle-${name.toLowerCase()}`}
                  data-cursor="hover"
                  onClick={() => onSelectStage(index)}
                >
                  <span className="lifecycle-step-top">
                    <span className="mono">{number}</span>
                    <Icon size={16} aria-hidden="true" />
                  </span>
                  <strong>{name}</strong>
                </button>
              ))}
            </div>
            <div className="lifecycle-track" aria-hidden="true">
              <span style={{ width: `${(selectedStage / (lifecycle.length - 1)) * 100}%` }} />
            </div>
            <div
              className="lifecycle-panel"
              id={`lifecycle-panel-${lifecycle[selectedStage].number}`}
              role="tabpanel"
              aria-live="polite"
              data-testid="panel-lifecycle-stage"
            >
              <div>
                <span className="mono lifecycle-command">
                  <span className="prompt">$</span> {lifecycle[selectedStage].command}
                </span>
                <p>{lifecycle[selectedStage].copy}</p>
              </div>
              <div className="lifecycle-outcome">
                <span className="mono">output</span>
                <strong>{lifecycle[selectedStage].outcome}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
