'use client';

import { useEffect, useRef, useState } from 'react';
import { ChevronRight, X, ArrowUpRight } from 'lucide-react';
import gsap from 'gsap';
import { projects, type Project } from '@/lib/content';
import { withBasePath } from '@/lib/base-path';
import { sceneState } from '@/lib/scene-state';
import { Magnetic } from './magnetic';
import { SplitTitle } from './split-title';
import { WorkDiagram } from './work-diagram';

export function Work({
  onOpenBrief,
}: {
  onOpenBrief: (project: Project, origin: DOMRect) => void;
}) {
  const [hovered, setHovered] = useState<number | null>(null);
  const [pinnedIndex, setPinnedIndex] = useState(0);

  useEffect(() => sceneState.subscribe((snapshot) => {
    setPinnedIndex(snapshot.workIndex);
  }), []);

  return (
    <section className="section work" id="work" aria-labelledby="work-heading">
      <div className="work-pin">
        <div className="container work-head">
          <div>
            <div className="section-label mono">03 / selected work</div>
            <SplitTitle id="work-heading" lines={['Four pieces', 'of work.']} />
          </div>
          <p className="section-intro">
            Cards 01–03 are parts of the same IT22 job. Card 04 is a public tool I wrote. IT22 pages
            omit customer names and unpublished numbers.
          </p>
        </div>
        <div className="work-track">
          {projects.map((project, index) => (
            <Magnetic className="work-panel-magnet" key={project.number} strength={0.08}>
              <article
                className="work-panel"
                data-testid={`card-project-${project.number}`}
                data-work-index={index}
                data-cursor="hover"
                data-cursor-label="inspect"
                onPointerEnter={() => setHovered(index)}
                onPointerLeave={() => setHovered(null)}
              >
                <WorkDiagram
                  number={project.number}
                  active={hovered === index || (hovered === null && pinnedIndex === index)}
                />
                <div className="work-panel-no mono">{project.number}</div>
                <div className="project-sub mono">{project.scope}</div>
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
                  <a
                    className="text-link"
                    href={withBasePath(`/work/${project.slug}/`)}
                    data-testid={`link-case-study-${project.number}`}
                    data-cursor="hover"
                    data-cursor-label="open"
                  >
                    Read the case study <ArrowUpRight size={12} aria-hidden="true" />
                  </a>
                  <button
                    className="text-link brief-button"
                    type="button"
                    data-testid={`button-read-brief-${project.number}`}
                    data-cursor="hover"
                    data-cursor-label="inspect"
                    onClick={(event) => {
                      const panel = (event.currentTarget as HTMLElement).closest('.work-panel');
                      onOpenBrief(project, (panel ?? event.currentTarget).getBoundingClientRect());
                    }}
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
  origin,
  onClose,
}: {
  project: Project;
  origin: DOMRect | null;
  onClose: () => void;
}) {
  const backdropRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const closingRef = useRef(false);

  useEffect(() => {
    const backdrop = backdropRef.current;
    const dialog = dialogRef.current;
    if (!backdrop || !dialog) return;

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced || !origin) {
      gsap.set(backdrop, { opacity: 1 });
      gsap.set(dialog, { clearProps: 'transform,clipPath' });
      return;
    }

    const dest = dialog.getBoundingClientRect();
    const scaleX = origin.width / dest.width;
    const scaleY = origin.height / dest.height;
    const dx = origin.left + origin.width / 2 - (dest.left + dest.width / 2);
    const dy = origin.top + origin.height / 2 - (dest.top + dest.height / 2);

    gsap.set(backdrop, { opacity: 0 });
    gsap.set(dialog, {
      x: dx,
      y: dy,
      scaleX,
      scaleY,
      clipPath: 'inset(12% 12% 12% 12%)',
      transformOrigin: 'center center',
    });

    const intro = gsap.timeline();
    intro.to(backdrop, { opacity: 1, duration: 0.35, ease: 'power2.out' }, 0);
    intro.to(
      dialog,
      {
        x: 0,
        y: 0,
        scaleX: 1,
        scaleY: 1,
        clipPath: 'inset(0% 0% 0% 0%)',
        duration: 0.55,
        ease: 'power3.out',
      },
      0,
    );

    return () => {
      intro.kill();
    };
  }, [origin, project]);

  const dismiss = () => {
    if (closingRef.current) return;
    const backdrop = backdropRef.current;
    const dialog = dialogRef.current;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced || !origin || !backdrop || !dialog) {
      onClose();
      return;
    }

    closingRef.current = true;
    const dest = dialog.getBoundingClientRect();
    const scaleX = origin.width / dest.width;
    const scaleY = origin.height / dest.height;
    const dx = origin.left + origin.width / 2 - (dest.left + dest.width / 2);
    const dy = origin.top + origin.height / 2 - (dest.top + dest.height / 2);

    const outro = gsap.timeline({
      onComplete: onClose,
    });
    outro.to(backdrop, { opacity: 0, duration: 0.28, ease: 'power2.in' }, 0);
    outro.to(
      dialog,
      {
        x: dx,
        y: dy,
        scaleX,
        scaleY,
        clipPath: 'inset(12% 12% 12% 12%)',
        duration: 0.4,
        ease: 'power3.in',
      },
      0,
    );
  };

  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      ref={backdropRef}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) dismiss();
      }}
    >
      <section
        className="brief-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="brief-title"
        data-testid="dialog-project-brief"
        ref={dialogRef}
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
            onClick={dismiss}
          >
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <div className="dialog-body">
          <div className="project-sub mono">{project.scope}</div>
          <div className="project-sub mono">{project.subtitle}</div>
          <p>{project.detail}</p>
          <div className="detail-grid">
            <div className="detail-box">
              <strong>What this is</strong>
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
            <a
              className="text-link"
              href={withBasePath(`/work/${project.slug}/`)}
              data-testid="link-brief-case-study"
              data-cursor="hover"
              data-cursor-label="open"
              onClick={dismiss}
            >
              Read the full case study <ArrowUpRight size={14} aria-hidden="true" />
            </a>
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
              onClick={dismiss}
            >
              Contact <ArrowUpRight size={14} aria-hidden="true" />
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}
