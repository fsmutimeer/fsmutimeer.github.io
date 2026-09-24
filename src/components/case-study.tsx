import { ArrowUpRight } from 'lucide-react';
import { doctorRules, type Project } from '@/lib/content';
import { profile } from '@/lib/profile';
import { withBasePath } from '@/lib/base-path';
import { CaseStudyFigure } from './case-study-figure';
import { WorkDiagram } from './work-diagram';

const captions: Record<string, string> = {
  'quarkus-kafka': 'IT22 services: Kafka between modules, Keycloak for roles, MongoDB for data.',
  'openshift-okd': 'Two IT22 clusters on KVM. Each is 3 control-plane nodes + 1 worker.',
  'gitops-tekton': 'Git → Tekton + Trivy → Argo CD on OpenShift. Wazuh on the cluster.',
  'quarkus-doctor': 'Public tool. Reads config and YAML. Does not call a cluster.',
};

export function CaseStudy({ project }: { project: Project }) {
  const { caseStudy } = project;
  const homeWork = withBasePath('/#work');

  return (
    <article className="study">
      <header className="study-bar">
        <div className="container study-bar-inner">
          <a className="brand" href={withBasePath('/')}>
            <span className="brand-mark">{profile.initials}</span>
            <span>
              {profile.name}
              <span className="brand-dot">.</span>
            </span>
          </a>
          <nav className="study-bar-links" aria-label="Case study">
            <a href={withBasePath('/about/')}>About me</a>
            <a href={homeWork}>Selected work</a>
            <a href={profile.cvUrl} target="_blank" rel="noopener noreferrer">
              CV
            </a>
          </nav>
        </div>
      </header>

      <div className="container study-body">
        <p className="eyebrow mono">
          case study {project.number} / {project.slug}
        </p>
        <p className="project-sub mono">{project.scope}</p>
        <h1 className="study-title">{project.title}</h1>
        <p className="study-lede">{project.copy}</p>
        <p className="hero-stack mono">{project.subtitle}</p>
        <div className="study-meta">
          <span className="mono">{project.role}</span>
          <span>{project.outcome}</span>
        </div>
        <div className="tags">
          {project.tags.map((tag) => (
            <span className="pill pill-accent" key={tag}>
              {tag}
            </span>
          ))}
        </div>
        <div className="work-metrics">
          {project.metrics.map((metric) => (
            <div className="metric mono" key={metric}>
              {metric}
            </div>
          ))}
        </div>

        <div className="study-diagram-row" aria-hidden="true">
          <WorkDiagram number={project.number} active />
        </div>

        {caseStudy.proprietary && (
          <p className="study-note mono">
            Proprietary IT22 work. Customer names, hostnames, and unpublished volumes are omitted.
          </p>
        )}

        <section className="study-block">
          <h2>Problem</h2>
          <p>{caseStudy.problem}</p>
        </section>
        <section className="study-block">
          <h2>Design</h2>
          <p>{caseStudy.design}</p>
        </section>
        <section className="study-block">
          <h2>Path</h2>
          <p>{caseStudy.path}</p>
        </section>

        <CaseStudyFigure slug={project.slug} caption={captions[project.slug] ?? project.outcome} />

        {project.slug === 'quarkus-doctor' && (
          <section className="study-block">
            <h2>What the current rule set fails on</h2>
            <p>
              Full list is in the docs. These are the errors that match the brief on the home page.
            </p>
            <div className="study-table-wrap">
              <table className="study-table">
                <thead>
                  <tr>
                    <th>Code</th>
                    <th>Level</th>
                    <th>Meaning</th>
                  </tr>
                </thead>
                <tbody>
                  {doctorRules.map((rule) => (
                    <tr key={rule.code}>
                      <td className="mono">{rule.code}</td>
                      <td className="mono">{rule.level}</td>
                      <td>{rule.meaning}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        <section className="study-block">
          <h2>Constraints</h2>
          <ul>
            {caseStudy.constraints.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
        <section className="study-block">
          <h2>Decisions</h2>
          <div className="study-decisions">
            {caseStudy.decisions.map((decision) => (
              <article key={decision.title}>
                <h3>{decision.title}</h3>
                <p>{decision.copy}</p>
              </article>
            ))}
          </div>
        </section>
        <section className="study-block">
          <h2>Limits</h2>
          <ul>
            {caseStudy.limits.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>

        <div className="study-footer">
          <a className="button" href={homeWork}>
            Back to selected work
          </a>
          {project.href && (
            <a
              className="text-link"
              href={project.href}
              target="_blank"
              rel="noopener noreferrer"
            >
              {project.hrefLabel ?? 'Documentation'} <ArrowUpRight size={14} aria-hidden="true" />
            </a>
          )}
          {project.repoUrl && (
            <a
              className="text-link"
              href={project.repoUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              Source <ArrowUpRight size={14} aria-hidden="true" />
            </a>
          )}
          <a className="text-link" href={`mailto:${profile.email}`}>
            {profile.email} <ArrowUpRight size={14} aria-hidden="true" />
          </a>
        </div>
      </div>
    </article>
  );
}
