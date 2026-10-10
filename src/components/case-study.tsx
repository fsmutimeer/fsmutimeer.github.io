'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, ArrowRight, ArrowUpRight, Check } from 'lucide-react';
import { doctorRules, projects, type Project } from '@/lib/content';
import { profile } from '@/lib/profile';
import { withBasePath } from '@/lib/base-path';
import { CaseStudyFigure } from './case-study-figure';
import { WorkDiagram } from './work-diagram';
import { Nav } from './nav';
import { Cursor } from './cursor';

const captions: Record<string, string> = {
  migrax: 'JPA entities → generated SQL with rollback → lint, verify, migrate, and drift checks.',
  'gitops-tekton': 'Git → Tekton CI + Trivy scanning → Argo CD GitOps delivery on OpenShift.',
  'quarkus-doctor': 'Static analysis tool: validates Quarkus configuration against Kubernetes manifests.',
};

export function CaseStudy({ project }: { project: Project }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const { caseStudy } = project;
  const homeWork = withBasePath('/#work');
  const index = projects.findIndex((p) => p.slug === project.slug);
  const prev = index > 0 ? projects[index - 1] : undefined;
  const next = index < projects.length - 1 ? projects[index + 1] : undefined;
  const pathSteps = caseStudy.path.includes('→')
    ? caseStudy.path.split('→').map((step) => step.trim()).filter(Boolean)
    : null;

  return (
    <main className="portfolio-shell study-page">
      <div className="grain" aria-hidden="true" />
      <Cursor />

      <Nav
        menuOpen={menuOpen}
        onToggle={() => setMenuOpen((v) => !v)}
        onClose={() => setMenuOpen(false)}
        minimal
      />

      <article className="study">
        <div className="container study-body">
          <div className="study-nav-strip">
            <Link
              className="study-back-link mono"
              href={homeWork}
              data-cursor="hover"
              data-cursor-label="back"
            >
              ← Selected work
            </Link>
          </div>

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
          <ul className="work-card-highlights study-highlights">
            {project.metrics.map((metric) => (
              <li key={metric}>
                <Check size={13} aria-hidden="true" />
                {metric}
              </li>
            ))}
          </ul>

          <div className="study-diagram-row" aria-hidden="true">
            <WorkDiagram slug={project.slug} active />
          </div>

          {caseStudy.proprietary && (
            <p className="study-note mono">
              Proprietary production work. Customer names, hostnames, and unpublished volumes are omitted.
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
            {pathSteps ? (
              <ol className="study-path">
                {pathSteps.map((step, i) => (
                  <li key={step}>
                    <span className="study-path-no mono">{String(i + 1).padStart(2, '0')}</span>
                    {step}
                  </li>
                ))}
              </ol>
            ) : (
              <p>{caseStudy.path}</p>
            )}
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
          {caseStudy.retrospective && (
            <section className="study-block">
              <h2>What I&apos;d change</h2>
              <p>{caseStudy.retrospective}</p>
            </section>
          )}

          <nav className="study-pager" aria-label="More case studies">
            {prev ? (
              <Link
                className="study-pager-link"
                href={withBasePath(`/work/${prev.slug}/`)}
                data-cursor="hover"
                data-cursor-label="prev"
              >
                <span className="mono"><ArrowLeft size={12} aria-hidden="true" /> Previous</span>
                <strong>{prev.title}</strong>
              </Link>
            ) : <span />}
            {next ? (
              <Link
                className="study-pager-link is-next"
                href={withBasePath(`/work/${next.slug}/`)}
                data-cursor="hover"
                data-cursor-label="next"
              >
                <span className="mono">Next <ArrowRight size={12} aria-hidden="true" /></span>
                <strong>{next.title}</strong>
              </Link>
            ) : <span />}
          </nav>

          <div className="study-footer">
            <Link className="button" href={homeWork} data-cursor="hover" data-cursor-label="work">
              Back to selected work
            </Link>
            {project.href && (
              <Link
                className="text-link"
                href={project.href}
                target="_blank"
                rel="noopener noreferrer"
                data-cursor="hover"
                data-cursor-label="docs"
              >
                {project.hrefLabel ?? 'Documentation'} <ArrowUpRight size={14} aria-hidden="true" />
              </Link>
            )}
            {project.repoUrl && (
              <Link
                className="text-link"
                href={project.repoUrl}
                target="_blank"
                rel="noopener noreferrer"
                data-cursor="hover"
                data-cursor-label="repo"
              >
                Source <ArrowUpRight size={14} aria-hidden="true" />
              </Link>
            )}
            <Link
              className="text-link"
              href={`mailto:${profile.email}`}
              data-cursor="hover"
              data-cursor-label="mail"
            >
              {profile.email} <ArrowUpRight size={14} aria-hidden="true" />
            </Link>
          </div>
        </div>
      </article>
    </main>
  );
}
