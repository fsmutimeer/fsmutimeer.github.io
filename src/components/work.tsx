'use client';

import { ChevronRight, X, ArrowUpRight } from 'lucide-react';
import { projects, type Project } from '@/lib/content';
import { Magnetic } from './magnetic';

export function Work({
  onOpenBrief,
}: {
  onOpenBrief: (project: Project) => void;
}) {
  return (
    <section className="section work" id="work" aria-labelledby="work-heading">
      <div className="work-pin">
        <div className="container work-head">
          <div>
            <div className="section-label mono">02 / selected systems</div>
            <h2 className="section-title" id="work-heading">
              The work behind
              <br />
              the cluster.
            </h2>
          </div>
          <p className="section-intro">
            Three systems from IT22—the services, the on-prem cluster, the GitOps path—and
            quarkus-doctor, the Maven plugin I built to catch Quarkus config bugs before deploy.
          </p>
        </div>
        <div className="work-track">
          {projects.map((project) => (
            <Magnetic className="work-panel-magnet" key={project.number} strength={0.08}>
              <article
                className="work-panel"
                data-testid={`card-project-${project.number}`}
                data-cursor="hover"
                data-cursor-label="brief"
              >
                <div className="work-panel-no mono">{project.number}</div>
                <div className="project-sub mono">{project.subtitle}</div>
                <h3>{project.title}</h3>
                <p data-testid={`text-project-copy-${project.number}`}>{project.copy}</p>
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
                <div className="work-panel-actions">
                  <button
                    className="text-link brief-button"
                    type="button"
                    data-testid={`button-read-brief-${project.number}`}
                    data-cursor="hover"
                    data-cursor-label="brief"
                    onClick={() => onOpenBrief(project)}
                  >
                    Read the brief <ChevronRight size={12} aria-hidden="true" />
                  </button>
                  {project.href && (
                    <a
                      className="text-link"
                      href={project.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      data-testid={`link-project-docs-${project.number}`}
                      data-cursor="hover"
                      data-cursor-label="open"
                    >
                      {project.hrefLabel ?? 'Documentation'} <ArrowUpRight size={12} aria-hidden="true" />
                    </a>
                  )}
                </div>
              </article>
            </Magnetic>
          ))}
        </div>
      </div>
    </section>
  );
}

export function BriefDialog({
  project,
  onClose,
}: {
  project: Project;
  onClose: () => void;
}) {
  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="brief-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="brief-title"
        data-testid="dialog-project-brief"
      >
        <div className="dialog-head">
          <div>
            <div className="dialog-number">{project.number} / SYSTEM BRIEF</div>
            <h2 className="dialog-title" id="brief-title">
              {project.title}
            </h2>
          </div>
          <button
            className="dialog-close"
            type="button"
            aria-label="Close project brief"
            data-testid="button-close-brief"
            data-cursor="hover"
            data-cursor-label="close"
            onClick={onClose}
          >
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <div className="dialog-body">
          <div className="project-sub mono">{project.subtitle}</div>
          <p>{project.detail}</p>
          <div className="detail-grid">
            <div className="detail-box">
              <strong>Signal</strong>
              <span>{project.metrics.join(' · ')}</span>
            </div>
            <div className="detail-box">
              <strong>Role</strong>
              <span>{project.role}</span>
            </div>
            <div className="detail-box">
              <strong>Outcome</strong>
              <span>{project.outcome}</span>
            </div>
            <div className="detail-box">
              <strong>Tools in the path</strong>
              <span>{project.tags.join(' · ')}</span>
            </div>
          </div>
          <div className="dialog-footer">
            {project.href && (
              <a
                className="text-link"
                href={project.href}
                target="_blank"
                rel="noopener noreferrer"
                data-testid="link-brief-docs"
                data-cursor="hover"
                data-cursor-label="open"
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
                data-testid="link-brief-repo"
                data-cursor="hover"
                data-cursor-label="open"
              >
                Source <ArrowUpRight size={14} aria-hidden="true" />
              </a>
            )}
            <a
              className="text-link"
              href="#contact"
              data-testid="link-brief-contact"
              data-cursor="hover"
              data-cursor-label="talk"
              onClick={onClose}
            >
              Talk through a similar problem <ArrowUpRight size={14} aria-hidden="true" />
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}
