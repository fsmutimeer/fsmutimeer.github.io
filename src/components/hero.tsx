'use client';

import { useEffect, useRef } from 'react';
import { ArrowDownRight, FileText, Github } from 'lucide-react';
import { FaLinkedinIn } from 'react-icons/fa';
import { navItems } from '@/lib/content';
import { profile } from '@/lib/profile';
import { scrambleTo } from '@/lib/scramble';
import { Magnetic } from './magnetic';

const indexTotal = String(navItems.length + 1).padStart(2, '0');

export function Hero({ ready }: { ready: boolean }) {
  const whoamiRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const node = whoamiRef.current;
    if (!ready || !node) return;
    return scrambleTo(node, profile.whoami, 0.85);
  }, [ready]);

  return (
    <section className="hero" id="top" aria-labelledby="hero-heading" data-ready={ready}>
      <div className="container hero-inner">
        <p className="eyebrow mono">{profile.hero.eyebrow}</p>
        <h1 id="hero-heading">
          <span className="line-mask">
            <span className="line">{profile.hero.lines[0]}</span>
          </span>
          <span className="line-mask">
            <span className="line">{profile.hero.lines[1]}</span>
          </span>
          <span className="line-mask">
            <span className="line">
              <em>{profile.hero.lines[2]}</em>
            </span>
          </span>
        </h1>
        <p className="hero-copy">
          I’m <strong>{profile.name}</strong> — {profile.hero.copy}
        </p>
        <p className="hero-stack mono">{profile.hero.stack}</p>
        <div className="actions">
          <Magnetic>
            <a
              className="button"
              href="#work"
              data-testid="link-selected-work"
              data-cursor="hover"
              data-cursor-label="work"
            >
              See selected work <ArrowDownRight size={16} aria-hidden="true" />
            </a>
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
            href={profile.githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            data-testid="link-github-hero"
            data-cursor="hover"
            data-cursor-label="github"
          >
            GitHub <Github size={14} aria-hidden="true" />
          </a>
          <a
            className="text-link"
            href={profile.linkedinUrl}
            target="_blank"
            rel="noopener noreferrer"
            data-testid="link-linkedin-hero"
            data-cursor="hover"
            data-cursor-label="linkedin"
          >
            LinkedIn <FaLinkedinIn size={13} aria-hidden="true" />
          </a>
        </div>
      </div>
      <aside className="hero-hud mono" aria-label={`${profile.name} telemetry`}>
        <span>
          <span className="prompt">$</span> whoami
        </span>
        <span className="hud-result" ref={whoamiRef} aria-label={profile.whoami} />
        <span className="hud-signal">{profile.timezone}</span>
        <span>index 01 / {indexTotal}</span>
      </aside>
      <div className="scroll-cue mono">
        <span />
        scroll
      </div>
    </section>
  );
}
