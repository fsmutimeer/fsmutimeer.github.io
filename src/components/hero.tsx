"use client";

import Link from "next/link";
import { ArrowDownRight, FileText } from "lucide-react";
import { profile } from "@/lib/profile";
import { Magnetic } from "./magnetic";

export function Hero({ ready }: { ready: boolean }) {
  const nameParts = profile.name.split(" ");
  const firstName = nameParts.slice(0, -1).join(" ");
  const lastName = nameParts[nameParts.length - 1] ?? profile.name;

  return (
    <section
      className="hero"
      id="top"
      aria-labelledby="hero-heading"
      data-ready={ready}
    >
      <div className="container hero-inner">
        <p className="eyebrow mono">{profile.hero.eyebrow}</p>
        <h1 id="hero-heading">
          <span className="line-mask">
            <span className="line">
              {firstName} <em>{lastName}</em>
            </span>
          </span>
        </h1>
        <p className="hero-copy">{profile.hero.copy}</p>
        <p className="hero-stack mono">{profile.hero.stack}</p>
        <div className="actions">
          <Magnetic>
            <Link
              className="button"
              href="#work"
              data-testid="link-selected-work"
              data-cursor="hover"
              data-cursor-label="work"
            >
              See selected work <ArrowDownRight size={16} aria-hidden="true" />
            </Link>
          </Magnetic>
          <a
            className="text-link"
            href={profile.cvUrl}
            target="_blank"
            rel="noopener noreferrer"
            data-testid="link-cv-hero"
            data-cursor="hover"
            data-cursor-label="cv"
          >
            Download CV <FileText size={14} aria-hidden="true" />
          </a>
          <a
            className="text-link"
            href={`mailto:${profile.email}`}
            data-testid="link-contact-hero"
            data-cursor="hover"
            data-cursor-label="contact"
          >
            Get in touch
          </a>
        </div>
      </div>
      <div className="scroll-cue mono">
        <span />
        scroll
      </div>
    </section>
  );
}
