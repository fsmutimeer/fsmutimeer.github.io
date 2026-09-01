'use client';

import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';

const STORAGE_KEY = 'inspect-grid';

const headings = [
  '#hero-heading',
  '#about-heading',
  '#experience-heading',
  '#work-heading',
  '#approach-heading',
  '#platform-heading',
  '#now-heading',
] as const;

export function requestInspectFrame() {
  window.dispatchEvent(new Event('inspect-frame'));
}

function currentHeading(): HTMLElement | null {
  const mid = window.innerHeight * 0.38;
  let best: HTMLElement | null = null;
  let bestDist = Infinity;
  for (const selector of headings) {
    const node = document.querySelector<HTMLElement>(selector);
    if (!node) continue;
    const rect = node.getBoundingClientRect();
    if (rect.bottom < 80 || rect.top > window.innerHeight - 40) continue;
    const dist = Math.abs(rect.top - mid);
    if (dist < bestDist) {
      best = node;
      bestDist = dist;
    }
  }
  return best ?? document.querySelector<HTMLElement>('#hero-heading');
}

export function InspectGrid({ ready }: { ready: boolean }) {
  const [gridOn, setGridOn] = useState(false);
  const [guides, setGuides] = useState<{ top: number; bottom: number; left: number; right: number } | null>(
    null,
  );
  const [reduced, setReduced] = useState(false);
  const frameRef = useRef<ReturnType<typeof gsap.to> | null>(null);

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(media.matches);
    if (media.matches) return;

    try {
      setGridOn(sessionStorage.getItem(STORAGE_KEY) === '1');
    } catch {
      setGridOn(false);
    }

    const onKey = (event: KeyboardEvent) => {
      if (!event.altKey || (event.key !== 'g' && event.key !== 'G')) return;
      if (event.repeat) return;
      event.preventDefault();
      setGridOn((prev) => {
        const next = !prev;
        try {
          sessionStorage.setItem(STORAGE_KEY, next ? '1' : '0');
        } catch {
          /* ignore */
        }
        return next;
      });
    };

    const onFrame = () => {
      const target = currentHeading();
      if (!target) return;
      const rect = target.getBoundingClientRect();
      setGuides({
        top: rect.top,
        bottom: rect.bottom,
        left: rect.left,
        right: rect.right,
      });
    };

    window.addEventListener('keydown', onKey);
    window.addEventListener('inspect-frame', onFrame);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('inspect-frame', onFrame);
    };
  }, []);

  useEffect(() => {
    if (!guides) return;
    const nodes = document.querySelectorAll<HTMLElement>('.inspect-guide');
    if (!nodes.length) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    frameRef.current?.kill();
    gsap.set(nodes, { opacity: 0 });
    frameRef.current = gsap.to(nodes, { opacity: 1, duration: 0.45, stagger: 0.04, ease: 'power2.out' });
    const hide = window.setTimeout(() => {
      gsap.to(nodes, {
        opacity: 0,
        duration: 0.6,
        ease: 'power2.in',
        onComplete: () => setGuides(null),
      });
    }, 2400);
    return () => {
      window.clearTimeout(hide);
      frameRef.current?.kill();
    };
  }, [guides]);

  if (!ready || reduced) return null;

  return (
    <>
      {gridOn && (
        <div className="inspect-grid" aria-hidden="true">
          <div className="inspect-baselines" />
          <div className="inspect-columns">
            {Array.from({ length: 12 }, (_, index) => (
              <span key={index} />
            ))}
          </div>
        </div>
      )}
      {guides && (
        <div className="inspect-guides" aria-hidden="true">
          <span className="inspect-guide inspect-guide--h" style={{ top: guides.top }} />
          <span className="inspect-guide inspect-guide--h" style={{ top: guides.bottom }} />
          <span className="inspect-guide inspect-guide--v" style={{ left: guides.left }} />
          <span className="inspect-guide inspect-guide--v" style={{ left: guides.right }} />
        </div>
      )}
    </>
  );
}
