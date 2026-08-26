'use client';

import { useEffect, useRef } from 'react';
import { ArrowDownRight, ArrowUpRight } from 'lucide-react';
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
        <p className="eyebrow mono">software engineer / IT22 B.V. · islamabad</p>
        <h1 id="hero-heading">
          <span className="line-mask">
            <span className="line">Code that</span>
          </span>
          <span className="line-mask">
            <span className="line">survives</span>
          </span>
          <span className="line-mask">
            <span className="line">
              <em>production.</em>
            </span>
          </span>
        </h1>
        <p className="hero-copy">
          I’m <strong>{profile.name}</strong> — a software engineer at {profile.company} turning
          Java services into systems that survive the cluster. From a Quarkus commit to a healthy
          pod on OpenShift, I own the path in between.
        </p>
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
            href="#contact"
            data-testid="link-start-conversation"
            data-cursor="hover"
            data-cursor-label="talk"
          >
            Start a conversation <ArrowUpRight size={14} aria-hidden="true" />
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
        scroll the control plane
      </div>
    </section>
  );
}
