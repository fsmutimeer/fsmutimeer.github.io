'use client';

import { useEffect, useRef, useState } from 'react';

import Image from 'next/image';
import Link from 'next/link';
import Lenis from 'lenis';
import { aboutStory } from '@/lib/about';
import { usePrefersReducedMotion } from '@/lib/use-reduced-motion';
import { Nav } from './nav';
import { Cursor } from './cursor';

import 'lenis/dist/lenis.css';



function PhotoPlaceholder({
  slot,
  caption,
  aspectHint,
  objectPosition,
}: {
  slot: string;
  caption: string;
  aspectHint?: string;
  objectPosition?: string;
}) {
  const [status, setStatus] = useState<'loading' | 'loaded' | 'error'>('loading');
  const imgSrc = `/about/${slot}.jpg`;

  /* Resolve the CSS aspect-ratio for the loaded photo vs the placeholder */
  const loadedAspect = aspectHint ?? '4 / 5';
  const placeholderAspect = '4 / 5';

  return (
    <figure className="story-figure">
      {status !== 'error' ? (
        <Image
          className="story-photo"
          src={imgSrc}
          alt={caption}
          width={960}
          height={640}
          onLoad={() => setStatus('loaded')}
          onError={() => setStatus('error')}
          style={{
            ...(status === 'loading'
              ? { visibility: 'hidden', position: 'absolute' }
              : {
                  aspectRatio: loadedAspect,
                  objectPosition: objectPosition ?? 'center center',
                }),
          }}
        />
      ) : null}

      {status !== 'loaded' ? (
        <div
          className="story-photo-slot"
          data-photo-slot={slot}
          role="img"
          aria-label={`${caption}. Placeholder until the photo is added.`}
          style={{ aspectRatio: placeholderAspect }}
        >
          <span className="story-photo-mark mono">photo placeholder</span>
          <span className="story-photo-hint mono">public/about/{slot}.jpg</span>
        </div>
      ) : null}

      {/* Show the real caption only when the photo is loaded */}
      {status === 'loaded' ? (
        <figcaption className="mono">{caption}</figcaption>
      ) : null}
    </figure>
  );
}

export function AboutStory() {
  const [menuOpen, setMenuOpen] = useState(false);
  const reducedMotion = usePrefersReducedMotion();
  const lenisRef = useRef<Lenis | null>(null);

  /* ── Lenis smooth-scroll (mirrors portfolio.tsx) ── */
  useEffect(() => {
    if (reducedMotion) return;

    const lenis = new Lenis({
      duration: 1.2,
      smoothWheel: true,
      touchMultiplier: 1.15,
    });
    lenisRef.current = lenis;

    const raf = (time: number) => {
      lenis.raf(time);
      requestAnimationFrame(raf);
    };
    const id = requestAnimationFrame(raf);

    /* Anchor-link click handling */
    const onClick = (event: MouseEvent) => {
      const target = (event.target as HTMLElement | null)?.closest(
        'a[href^="#"]',
      ) as HTMLAnchorElement | null;
      if (!target) return;
      const hash = target.getAttribute('href');
      if (!hash || hash === '#') return;
      const el = document.querySelector(hash);
      if (!el) return;
      event.preventDefault();
      lenis.scrollTo(el as HTMLElement, { offset: -72 });
    };
    document.addEventListener('click', onClick);

    return () => {
      cancelAnimationFrame(id);
      document.removeEventListener('click', onClick);
      lenis.destroy();
      lenisRef.current = null;
    };
  }, [reducedMotion]);

  /* ── Pause / resume on menu open ── */
  useEffect(() => {
    const lenis = lenisRef.current;
    if (menuOpen) {
      lenis?.stop();
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
      lenis?.start();
    }
  }, [menuOpen]);

  return (
    <main className="portfolio-shell about-page">
      <div className="grain" aria-hidden="true" />
      <Cursor />

      <Nav
        menuOpen={menuOpen}
        onToggle={() => setMenuOpen((v) => !v)}
        onClose={() => setMenuOpen(false)}
        minimal
      />

      <article className="story">
        <div className="container study-body">
          <p className="eyebrow mono">{aboutStory.eyebrow}</p>
          <h1 className="study-title">{aboutStory.title}</h1>
          <p className="study-lede">{aboutStory.lede}</p>

          {aboutStory.sections.map((section, index) => (
            <section
              className="story-section"
              id={section.id}
              key={section.id}
              aria-labelledby={`${section.id}-heading`}
              data-align={index % 2 === 0 ? 'image-left' : 'image-right'}
            >
              <PhotoPlaceholder
                slot={section.photoSlot}
                caption={section.photoCaption}
                aspectHint={section.aspectHint}
                objectPosition={section.objectPosition}
              />
              <div className="story-copy">
                <span className="story-no mono">{section.number}</span>
                <h2 id={`${section.id}-heading`}>{section.title}</h2>
                {section.paragraphs.map((paragraph) => (
                  <p key={paragraph.slice(0, 40)}>{paragraph}</p>
                ))}
              </div>
            </section>
          ))}

          <p className="story-signoff">{aboutStory.signoff}</p>

          <div className="study-footer">
            <Link className="text-link" href="/">
              Back to the portfolio
            </Link>
            <Link className="text-link" href="/#work">
              Selected work
            </Link>
          </div>
        </div>
      </article>
    </main>
  );
}
