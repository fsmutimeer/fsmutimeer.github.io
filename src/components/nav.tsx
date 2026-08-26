'use client';

import { useEffect, useRef, useState } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { navItems } from '@/lib/content';
import { profile } from '@/lib/profile';
import { Magnetic } from './magnetic';
import { scrambleTo } from '@/lib/scramble';
import { requestInspectFrame } from './inspect-grid';

const sections = [
  { id: 'top', label: 'Home', preview: 'Code that survives production.' },
  ...navItems,
] as const;

function OverlayLabel({ text, scramble }: { text: string; scramble: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!scramble) {
      node.textContent = text;
      return;
    }
    return scrambleTo(node, text, 0.45);
  }, [scramble, text]);

  return <span ref={ref} className="nav-overlay-label">{text}</span>;
}

export function Nav({
  menuOpen,
  onToggle,
  onClose,
}: {
  menuOpen: boolean;
  onToggle: () => void;
  onClose: () => void;
}) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLElement>(null);
  const maskRef = useRef<HTMLDivElement>(null);
  const [activeId, setActiveId] = useState('top');
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const previewId = hoveredId ?? activeId;
  const previewItem = sections.find((item) => item.id === previewId) ?? sections[0];
  const previewIndex = sections.findIndex((item) => item.id === previewItem.id) + 1;
  const activeIndex = Math.max(
    1,
    sections.findIndex((item) => item.id === activeId) + 1,
  );

  useEffect(() => {
    const nodes = sections
      .map(({ id }) => document.getElementById(id))
      .filter((node): node is HTMLElement => Boolean(node));
    if (!nodes.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target.id) setActiveId(visible.target.id);
      },
      { rootMargin: '-28% 0px -55% 0px', threshold: [0.1, 0.35, 0.6] },
    );

    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const update = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? Math.min(1, window.scrollY / max) : 0);
    };
    update();
    window.addEventListener('scroll', update, { passive: true });
    return () => window.removeEventListener('scroll', update);
  }, []);

  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    closeRef.current?.focus();
    return () => window.removeEventListener('keydown', onKey);
  }, [menuOpen, onClose]);

  useEffect(() => {
    if (!menuOpen) setHoveredId(null);
  }, [menuOpen]);

  useEffect(() => {
    const list = listRef.current;
    const mask = maskRef.current;
    if (!menuOpen || !list || !mask) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (window.matchMedia('(pointer: coarse)').matches) {
      gsap.set(mask, { autoAlpha: 0 });
      return;
    }

    const target = list.querySelector<HTMLElement>(
      `.nav-overlay-link[href="#${previewId}"]`,
    );
    if (!target) return;

    const place = (duration: number) => {
      const listRect = list.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      gsap.set(mask, { autoAlpha: 1 });
      gsap.to(mask, {
        x: targetRect.left - listRect.left,
        y: targetRect.top - listRect.top,
        width: targetRect.width,
        height: targetRect.height,
        duration,
        ease: 'power3.out',
        overwrite: 'auto',
        onComplete: () => {
          mask.dataset.placed = 'true';
        },
      });
    };

    const instant = mask.dataset.placed !== 'true';
    if (instant) {
      const delayed = gsap.delayedCall(0.62, () => place(0));
      return () => {
        delayed.kill();
      };
    }
    place(0.45);
  }, [menuOpen, previewId]);

  useEffect(() => {
    if (menuOpen) return;
    const mask = maskRef.current;
    if (!mask) return;
    mask.dataset.placed = '';
    gsap.set(mask, { autoAlpha: 0, width: 0, height: 0, x: 0, y: 0 });
  }, [menuOpen]);

  useGSAP(
    () => {
      if (!menuOpen || !overlayRef.current) return;
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
      gsap.fromTo(
        '.nav-overlay-link',
        { y: 36, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.55, stagger: 0.07, ease: 'power3.out', delay: 0.08 },
      );
      gsap.fromTo(
        '.nav-overlay-meta > *',
        { y: 16, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.5, stagger: 0.08, ease: 'power3.out', delay: 0.28 },
      );
    },
    { dependencies: [menuOpen] },
  );

  return (
    <header className="hud" data-open={menuOpen}>
      <div className="hud-progress" aria-hidden="true">
        <span style={{ transform: `scaleX(${progress})` }} />
      </div>
      <div className="container hud-bar">
        <div className="hud-start">
          <a
            href="#top"
            className="brand"
            data-testid="link-brand"
            data-cursor="hover"
            data-cursor-label="home"
            onClick={onClose}
          >
            <span className="brand-mark">{profile.initials}</span>
            <span>
              {profile.name}
              <span className="brand-dot">.</span>
            </span>
          </a>
          <button
            className="inspect-btn mono"
            type="button"
            aria-label="Frame the current heading"
            data-cursor="hover"
            data-cursor-label="inspect"
            onClick={(event) => {
              event.preventDefault();
              requestInspectFrame();
            }}
          >
            inspect
          </button>
        </div>
        <div className="hud-end">
          <div className="status" hidden={menuOpen}>
            <span className="status-dot" />
            <span>
              {String(activeIndex).padStart(2, '0')} / {String(sections.length).padStart(2, '0')}
            </span>
            <span className="status-copy">shipping at {profile.company}</span>
          </div>
          <button
            className="index-btn"
            type="button"
            ref={closeRef}
            aria-label={menuOpen ? 'Close site index' : 'Open site index'}
            aria-expanded={menuOpen}
            aria-controls="site-index"
            data-testid="button-mobile-menu"
            data-cursor="hover"
            data-cursor-label={menuOpen ? 'close' : 'index'}
            onClick={onToggle}
          >
            <span className="index-btn-label mono">{menuOpen ? 'Close' : 'Index'}</span>
            <span className="index-burger" aria-hidden="true">
              <i />
              <i />
            </span>
          </button>
        </div>
      </div>

      <div
        className="nav-overlay"
        id="site-index"
        ref={overlayRef}
        hidden={!menuOpen}
        role="dialog"
        aria-modal="true"
        aria-label="Site index"
      >
        <div className="container nav-overlay-grid">
          <nav
            className="nav-overlay-list"
            aria-label="Primary navigation"
            ref={listRef}
            onPointerLeave={() => setHoveredId(null)}
          >
            <div className="nav-overlay-mask" ref={maskRef} aria-hidden="true" />
            {sections.map(({ id, label }, index) => (
              <Magnetic key={id} strength={0.08}>
                <a
                  className={`nav-overlay-link${previewItem.id === id ? ' is-preview' : ''}`}
                  href={`#${id}`}
                  data-testid={id === 'top' ? 'link-nav-home' : `link-nav-${id}`}
                  data-cursor="hover"
                  data-cursor-label="open"
                  aria-current={activeId === id ? 'location' : undefined}
                  onPointerEnter={() => setHoveredId(id)}
                  onMouseEnter={() => setHoveredId(id)}
                  onFocus={() => setHoveredId(id)}
                  onClick={onClose}
                >
                  <span className="nav-overlay-no mono">{String(index + 1).padStart(2, '0')}</span>
                  <OverlayLabel text={label} scramble={hoveredId === id} />
                </a>
              </Magnetic>
            ))}
          </nav>
          <aside className="nav-overlay-meta">
            <div className="nav-overlay-preview" aria-live="polite">
              <span className="nav-overlay-preview-no mono">
                {String(previewIndex).padStart(2, '0')}
              </span>
              <p className="nav-overlay-preview-label">{previewItem.label}</p>
              <p>{previewItem.preview}</p>
            </div>
            <p className="mono nav-overlay-kicker">signal</p>
            <p>
              Software engineer at {profile.company}
              <br />
              {profile.location}
            </p>
            <p className="mono nav-overlay-inspect">ALT+G inspect</p>
            <a href={`mailto:${profile.email}`} data-cursor="hover" data-cursor-label="mail">
              {profile.email}
            </a>
            <a
              href={profile.cvUrl}
              target="_blank"
              rel="noopener noreferrer"
              data-cursor="hover"
              data-cursor-label="cv"
            >
              Download CV
            </a>
          </aside>
        </div>
      </div>
    </header>
  );
}
