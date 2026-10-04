"use client";

import { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import Link from "next/link";
import { ArrowUpRight, Github } from "lucide-react";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { projects } from "@/lib/content";
import { withBasePath } from "@/lib/base-path";
import { sceneState } from "@/lib/scene-state";
import { usePrefersReducedMotion } from "@/lib/use-reduced-motion";
import { SplitTitle } from "./split-title";
import { WorkDiagram } from "./work-diagram";

export function Work() {
  const [hovered, setHovered] = useState<number | null>(null);
  const [pinnedIndex, setPinnedIndex] = useState(0);
  const charAnimRef = useRef<gsap.core.Tween | null>(null);
  const reducedMotion = usePrefersReducedMotion();

  useEffect(
    () =>
      sceneState.subscribe((snapshot) => {
        setPinnedIndex(snapshot.workIndex);
      }),
    [],
  );

  /* ── Staggered title char morph on active card change ── */
  useEffect(() => {
    if (reducedMotion) return;
    const el = document.querySelector<HTMLElement>(
      `[data-work-index="${pinnedIndex}"]`,
    );
    if (!el) return;
    const chars = el.querySelectorAll<HTMLElement>(".work-title-char");
    if (!chars.length) return;
    charAnimRef.current?.kill();
    charAnimRef.current = gsap.fromTo(
      chars,
      { opacity: 0, y: 10 },
      {
        opacity: 1,
        y: 0,
        duration: 0.45,
        stagger: 0.022,
        ease: "power3.out",
        clearProps: "transform,opacity",
      },
    );
  }, [pinnedIndex, reducedMotion]);

  const goToCard = (index: number) => {
    const clamped = Math.max(0, Math.min(projects.length - 1, index));
    const trigger = ScrollTrigger.getById("work-pinned-track");
    if (!trigger) {
      const panel = document.querySelector(`[data-work-index="${clamped}"]`);
      panel?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
      return;
    }
    const progress = clamped / Math.max(1, projects.length - 1);
    const targetScroll = trigger.start + (trigger.end - trigger.start) * progress;
    const lenis = (window as unknown as { __lenis?: { scrollTo: (t: number) => void } }).__lenis;
    if (lenis) {
      lenis.scrollTo(targetScroll);
    } else {
      window.scrollTo({ top: targetScroll, behavior: "smooth" });
    }
  };

  return (
    <section className="section work" id="work" aria-labelledby="work-heading">
      <div className="work-pin">

        {/* ── Section header ── */}
        <div className="container work-head">
          <div>
            <div className="section-label mono">03 / selected work</div>
            <SplitTitle id="work-heading" lines={["Four pieces", "of work."]} />
          </div>
          <div className="work-head-right">
            <p className="section-intro">
              Three connected pieces of backend and platform work: services,
              on-prem clusters, and GitOps delivery. The fourth is my public
              Quarkus configuration tool. Proprietary details remain private.
            </p>
          </div>
        </div>

        {/* ── Horizontal Card stage ── */}
        <div className="work-stage">
          <div
            className="work-track"
            onFocusCapture={(event) => {
              if (window.matchMedia("(max-width: 899px)").matches) return;
              if (!(event.target instanceof HTMLElement)) return;
              const panel = event.target.closest<HTMLElement>(".work-panel");
              const trigger = ScrollTrigger.getById("work-pinned-track");
              if (!panel || !trigger) return;
              const index = Number(panel.dataset.workIndex ?? 0);
              if (sceneState.get().workIndex === index) return;
              const progress = index / Math.max(1, projects.length - 1);
              trigger.scroll(trigger.start + (trigger.end - trigger.start) * progress);
              ScrollTrigger.update();
            }}
          >
            {projects.map((project, index) => (
              <div className="work-panel-magnet" key={project.number}>
                <article
                  className={`work-panel${pinnedIndex === index ? " is-active" : ""}`}
                  data-testid={`card-project-${project.number}`}
                  data-work-index={index}
                  data-cursor="hover"
                  data-cursor-label={pinnedIndex === index ? "inspect" : "view"}
                  onClick={(e) => {
                    if ((e.target as HTMLElement).closest("a, button")) return;
                    if (pinnedIndex !== index) goToCard(index);
                  }}
                  onPointerEnter={(event) => {
                    if (event.pointerType !== "touch") setHovered(index);
                  }}
                  onPointerLeave={(event) => {
                    if (!event.currentTarget.matches(":focus-within")) setHovered(null);
                  }}
                  onFocusCapture={() => setHovered(index)}
                  onBlurCapture={(event) => {
                    const rel = event.relatedTarget;
                    if (!(rel instanceof Node) || !event.currentTarget.contains(rel)) {
                      setHovered(null);
                    }
                  }}
                >
                  {/* ── Terminal top bar ── */}
                  <div className="work-panel-terminal" aria-hidden="true">
                    <div className="terminal-dots">
                      <span className="terminal-dot td-red" />
                      <span className="terminal-dot td-yellow" />
                      <span className="terminal-dot td-green" />
                    </div>
                    <span className="terminal-label mono">
                      {project.number} — {project.scope}
                    </span>
                    <span className={`terminal-status mono${pinnedIndex === index ? " is-live" : ""}`}>
                      {pinnedIndex === index ? "● active" : "○ standby"}
                    </span>
                  </div>

                  {/* ── Split‑panel content body ── */}
                  <div className="work-panel-body">
                    <div className="work-panel-main">
                      <div className="work-panel-copy">
                        <div className="work-panel-head">
                          <span className="work-panel-no mono">{project.number}</span>
                          <span className="project-sub mono">{project.scope}</span>
                        </div>

                        {/* Title with individual char spans for morph animation */}
                        <h3 className="work-panel-title" aria-label={project.title}>
                          {project.title.split("").map((char, ci) => (
                            <span
                              key={ci}
                              className="work-title-char"
                              aria-hidden="true"
                            >
                              {char === " " ? "\u00a0" : char}
                            </span>
                          ))}
                        </h3>

                        <p data-testid={`text-project-copy-${project.number}`}>
                          {project.copy}
                        </p>
                      </div>

                      <div className="work-panel-visual" aria-hidden="true">
                        <WorkDiagram
                          number={project.number}
                          active={
                            hovered === index ||
                            (hovered === null && pinnedIndex === index)
                          }
                        />
                      </div>
                    </div>

                    <div className="work-panel-footer">
                      <div className="tags" aria-label={`Tech stack for ${project.title}`}>
                        {project.tags.map((tag) => (
                          <span className="tag mono" key={tag}>
                            {tag}
                          </span>
                        ))}
                      </div>

                      {project.metrics && project.metrics.length > 0 && (
                        <div
                          className="work-metrics"
                          aria-label={`Impact metrics for ${project.title}`}
                        >
                          {project.metrics.map((metric) => (
                            <span className="work-metric mono" key={metric}>
                              {metric}
                            </span>
                          ))}
                        </div>
                      )}

                      <div className="work-panel-actions">
                        {project.slug && (
                          <Link
                            className="work-case-link"
                            href={withBasePath(`/work/${project.slug}`)}
                            data-testid={`link-case-study-${project.number}`}
                            data-cursor="hover"
                            data-cursor-label="case study"
                          >
                            Read the case study <ArrowUpRight size={14} aria-hidden="true" />
                          </Link>
                        )}
                        {project.href && (
                          <a
                            className="text-link"
                            href={project.href}
                            target="_blank"
                            rel="noopener noreferrer"
                            data-testid={`link-project-documentation-${project.number}`}
                            data-cursor="hover"
                            data-cursor-label="docs"
                          >
                            {project.hrefLabel ?? "Documentation"} <ArrowUpRight size={12} aria-hidden="true" />
                          </a>
                        )}
                        {project.repoUrl && (
                          <a
                            className="text-link work-source-link"
                            href={project.repoUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            data-testid={`link-project-repository-${project.number}`}
                            data-cursor="hover"
                            data-cursor-label="source"
                          >
                            <Github size={14} aria-hidden="true" /> GitHub
                            <ArrowUpRight size={12} aria-hidden="true" />
                          </a>
                        )}
                      </div>
                    </div>
                  </div>
                </article>
              </div>
            ))}
          </div>
        </div>

        {/* ── Scroll‑progress strip (sleek line without label numbers) ── */}
        <div className="work-progress-strip" aria-hidden="true">
          <div className="work-progress-rail">
            <div className="work-progress-fill" id="work-progress-fill" />
          </div>
        </div>

      </div>
    </section>
  );
}
