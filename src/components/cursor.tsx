'use client';

import { useEffect, useRef } from 'react';
import gsap from 'gsap';

function verbFromTarget(target: EventTarget | null) {
  const el = target instanceof Element ? target : null;
  const node = el?.closest<HTMLElement>('a, button, [data-cursor="hover"], [data-cursor-label]');
  if (!node) return '';
  const labeled = node.getAttribute('data-cursor-label');
  if (labeled) return labeled;
  const href = node.getAttribute('href') ?? '';
  if (href.startsWith('mailto:')) return 'mail';
  if (href.includes('.pdf')) return 'cv';
  if (href.startsWith('tel:')) return 'call';
  return 'open';
}

export function Cursor() {
  const rootRef = useRef<HTMLDivElement>(null);
  const dotRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const labelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (window.matchMedia('(pointer: coarse)').matches) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    document.documentElement.classList.add('has-cursor');

    const dot = dotRef.current;
    const ring = ringRef.current;
    const label = labelRef.current;
    if (!dot || !ring || !label) return;

    const pos = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    const ringPos = { x: pos.x, y: pos.y };
    gsap.set([dot, ring, label], { x: pos.x, y: pos.y });

    const xTo = gsap.quickTo(dot, 'x', { duration: 0.12, ease: 'power3.out' });
    const yTo = gsap.quickTo(dot, 'y', { duration: 0.12, ease: 'power3.out' });

    const onMove = (event: MouseEvent) => {
      pos.x = event.clientX;
      pos.y = event.clientY;
      xTo(pos.x);
      yTo(pos.y);
      const verb = verbFromTarget(event.target);
      label.textContent = verb || '';
      gsap.to(ring, {
        scale: verb ? 2.35 : 1,
        duration: 0.35,
        ease: 'power3.out',
      });
      gsap.to(label, {
        opacity: verb ? 1 : 0,
        duration: 0.22,
        ease: 'power2.out',
      });
    };

    const tick = () => {
      ringPos.x += (pos.x - ringPos.x) * 0.16;
      ringPos.y += (pos.y - ringPos.y) * 0.16;
      gsap.set([ring, label], { x: ringPos.x, y: ringPos.y });
    };

    gsap.ticker.add(tick);
    window.addEventListener('mousemove', onMove);
    return () => {
      document.documentElement.classList.remove('has-cursor');
      gsap.ticker.remove(tick);
      window.removeEventListener('mousemove', onMove);
    };
  }, []);

  return (
    <div ref={rootRef} className="cursor-layer" aria-hidden="true">
      <div ref={dotRef} className="cursor-dot" />
      <div ref={ringRef} className="cursor-ring" />
      <div ref={labelRef} className="cursor-label mono" />
    </div>
  );
}
