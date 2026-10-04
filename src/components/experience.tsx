"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Minus, Plus } from "lucide-react";
import { experience } from "@/lib/content";
import { SplitTitle } from "./split-title";

type ExperienceJob = (typeof experience)[number];

function ExperienceEntry({
  job,
  expanded,
  onToggle,
}: {
  job: ExperienceJob;
  expanded: boolean;
  onToggle: () => void;
}) {
  const [renderDetails, setRenderDetails] = useState(expanded);
  const [closing, setClosing] = useState(false);
  const detailsRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const panel = detailsRef.current;
    if (!panel) return;

    const measure = () => {
      panel.style.setProperty(
        "--experience-details-height",
        `${panel.scrollHeight}px`,
      );
    };
    measure();

    const content = panel.firstElementChild;
    if (!content) return;
    const observer = new ResizeObserver(measure);
    observer.observe(content);
    return () => observer.disconnect();
  }, [renderDetails]);

  useEffect(() => {
    if (expanded) {
      setRenderDetails(true);
      setClosing(false);
      return;
    }
    if (!renderDetails) return;

    setClosing(true);
    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const closeDuration = reducedMotion ? 0 : 420;
    const timeout = window.setTimeout(() => {
      setRenderDetails(false);
      setClosing(false);
    }, closeDuration);
    return () => window.clearTimeout(timeout);
  }, [expanded, renderDetails]);

  return (
    <li className="experience-item" data-testid={`card-experience-${job.id}`}>
      <div className="experience-when mono">
        <span>{job.dates}</span>
        <span>{job.location}</span>
      </div>
      <div className={`experience-body${expanded ? " is-expanded" : ""}`}>
        <div className="experience-heading-row">
          <h3 className="experience-title">
            <a
              href={job.companyUrl}
              target="_blank"
              rel="noopener noreferrer"
              data-cursor="hover"
              data-cursor-label="open"
            >
              {job.company}
            </a>
          </h3>
          <button
            className="experience-toggle"
            id={`experience-trigger-${job.id}`}
            type="button"
            aria-label={`${expanded ? "Collapse" : "Expand"} ${job.company} details`}
            aria-expanded={expanded}
            aria-controls={`experience-panel-${job.id}`}
            data-testid={`button-experience-toggle-${job.id}`}
            onClick={onToggle}
          >
            <span className="experience-toggle-icon" aria-hidden="true">
              {expanded ? <Minus size={17} /> : <Plus size={17} />}
            </span>
          </button>
        </div>
        <p className="experience-role">{job.role}</p>
        {renderDetails && (
          <div
            className={`experience-details${expanded ? " is-open" : closing ? " is-closing" : ""}`}
            id={`experience-panel-${job.id}`}
            role="region"
            aria-labelledby={`experience-trigger-${job.id}`}
            aria-hidden={!expanded}
            inert={!expanded}
            ref={detailsRef}
            data-testid={`panel-experience-${job.id}`}
          >
            <ul>
              {job.bullets.map((bullet) => (
                <li key={bullet}>{bullet}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </li>
  );
}

export function Experience() {
  const [expandedId, setExpandedId] = useState<string | null>(
    experience[0]?.id ?? null,
  );

  return (
    <section
      className="section experience"
      id="experience"
      aria-labelledby="experience-heading"
    >
      <div className="container">
        <div className="experience-head">
          <div>
            <div className="section-label mono">02 / experience</div>
            <SplitTitle
              id="experience-heading"
              lines={["A career", "in systems."]}
            />
          </div>
          <p className="section-intro">
            From Node.js backends to Java services, integrations, and platform
            delivery.
          </p>
        </div>
        <ol className="experience-list" data-testid="list-experience">
          {experience.map((job) => (
            <ExperienceEntry
              key={job.id}
              job={job}
              expanded={expandedId === job.id}
              onToggle={() =>
                setExpandedId((current) => (current === job.id ? null : job.id))
              }
            />
          ))}
        </ol>
      </div>
    </section>
  );
}
