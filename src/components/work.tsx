"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ArrowRight, ArrowUpRight, Check } from "lucide-react";
import { FaGithub } from "react-icons/fa";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { projects } from "@/lib/content";
import { withBasePath } from "@/lib/base-path";
import { sceneState } from "@/lib/scene-state";
import { SplitTitle } from "./split-title";
import { WorkDiagram } from "./work-diagram";

export function Work() {
  const [hovered, setHovered] = useState<number | null>(null);
  const [pinnedIndex, setPinnedIndex] = useState(0);

  useEffect(
    () =>
      sceneState.subscribe((snapshot) => {
        setPinnedIndex(snapshot.workIndex);
      }),
    [],
  );

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
            <SplitTitle id="work-heading" lines={["Work and", "open source projects."]} />
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
                  data-cursor-label={pinnedIndex === index ? "open" : "view"}
                  onClick={(e) => {
                    if ((e.target as HTMLElement).closest("a, button")) return;
                    if (pinnedIndex !== index) goToCard(index);
                  }}
                  onPointerEnter={(event) => {
                    if (event.pointerType !== "touch") setHovered(index);
                  }}
                  onPointerMove={(event) => {
                    if (event.pointerType === "touch") return;
                    const el = event.currentTarget;
                    const rect = el.getBoundingClientRect();
                    // Normalise for GSAP scale so the glow tracks the cursor exactly.
                    const sx = el.offsetWidth / rect.width || 1;
                    const sy = el.offsetHeight / rect.height || 1;
                    el.style.setProperty("--mx", `${(event.clientX - rect.left) * sx}px`);
                    el.style.setProperty("--my", `${(event.clientY - rect.top) * sy}px`);
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
                  <span className="work-card-glow" aria-hidden="true" />
                  <span className="work-card-spot" aria-hidden="true" />

                  <div className="work-card-main">
                    <div className="work-card-copy">
                      <h3 className="work-card-title">{project.title}</h3>
                      <p
                        className="work-card-desc"
                        data-testid={`text-project-copy-${project.number}`}
                      >
                        {project.copy}
                      </p>
                      {project.metrics.length > 0 && (
                        <ul
                          className="work-card-highlights"
                          aria-label={`Highlights for ${project.title}`}
                        >
                          {project.metrics.map((metric) => (
                            <li key={metric}>
                              <Check size={13} aria-hidden="true" />
                              {metric}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>

                    <div className="work-card-visual" aria-hidden="true">
                      <WorkDiagram
                        slug={project.slug}
                        active={
                          hovered === index ||
                          (hovered === null && pinnedIndex === index)
                        }
                      />
                    </div>
                  </div>

                  <footer className="work-card-foot">
                    <ul className="work-card-tags" aria-label={`Tech stack for ${project.title}`}>
                      {project.tags.map((tag) => (
                        <li className="work-card-tag mono" key={tag}>
                          {tag}
                        </li>
                      ))}
                    </ul>
                    <div className="work-card-actions">
                      <Link
                        className="work-card-cta"
                        href={withBasePath(`/work/${project.slug}/`)}
                        data-testid={`link-case-study-${project.number}`}
                        data-cursor="hover"
                        data-cursor-label="case study"
                      >
                        Case study <ArrowUpRight size={14} aria-hidden="true" />
                      </Link>
                      {project.href && (
                        <a
                          className="work-card-link"
                          href={project.href}
                          target="_blank"
                          rel="noopener noreferrer"
                          data-testid={`link-project-documentation-${project.number}`}
                          data-cursor="hover"
                          data-cursor-label="docs"
                        >
                          {project.hrefLabel ?? "Docs"} <ArrowUpRight size={12} aria-hidden="true" />
                        </a>
                      )}
                      {project.repoUrl && (
                        <a
                          className="work-card-link"
                          href={project.repoUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          aria-label={`${project.title} on GitHub`}
                          data-testid={`link-project-repository-${project.number}`}
                          data-cursor="hover"
                          data-cursor-label="source"
                        >
                          <FaGithub size={14} aria-hidden="true" /> GitHub
                        </a>
                      )}
                    </div>
                  </footer>
                </article>
              </div>
            ))}
          </div>
        </div>

        {/* ── Track position + prev/next (desktop pinned track only) ── */}
        <div className="container work-controls" aria-label="Selected work navigation">
          <div className="work-arrows">
            <button
              type="button"
              className="work-arrow"
              onClick={() => goToCard(pinnedIndex - 1)}
              disabled={pinnedIndex === 0}
              aria-label="Previous project"
              data-cursor="hover"
              data-cursor-label="prev"
            >
              <ArrowLeft size={16} aria-hidden="true" />
            </button>
            <button
              type="button"
              className="work-arrow"
              onClick={() => goToCard(pinnedIndex + 1)}
              disabled={pinnedIndex === projects.length - 1}
              aria-label="Next project"
              data-cursor="hover"
              data-cursor-label="next"
            >
              <ArrowRight size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
