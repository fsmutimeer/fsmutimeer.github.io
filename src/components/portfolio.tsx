'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';
import Lenis from 'lenis';
import { ErrorBoundary } from '@/components/error-boundary';
import { projects, technologies, type Project, type Technology } from '@/lib/content';
import { sceneState } from '@/lib/scene-state';
import { usePrefersReducedMotion } from '@/lib/use-reduced-motion';
import { About } from './about';
import { Approach } from './approach';
import { Contact } from './contact';
import { Cursor } from './cursor';
import { Hero } from './hero';
import { InspectGrid } from './inspect-grid';
import { Nav } from './nav';
import { Preloader } from './preloader';
import { SceneFallback } from './scene/fallback';
import { BriefDialog, Work } from './work';
import 'lenis/dist/lenis.css';

gsap.registerPlugin(ScrollTrigger, useGSAP);

const SceneCanvas = dynamic(
  () => import('./scene/canvas').then((module) => module.SceneCanvas),
  { ssr: false, loading: () => <SceneFallback /> },
);

export function Portfolio() {
  const reducedMotion = usePrefersReducedMotion();
  const [menuOpen, setMenuOpen] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [briefOrigin, setBriefOrigin] = useState<DOMRect | null>(null);
  const [selectedTechnology, setSelectedTechnology] = useState<Technology>(technologies[0]);
  const [selectedStage, setSelectedStage] = useState(0);
  const [ready, setReady] = useState(false);
  const lenisRef = useRef<Lenis | null>(null);
  const rootRef = useRef<HTMLElement>(null);

  const closeMenu = () => setMenuOpen(false);

  useEffect(() => {
    const lenis = lenisRef.current;
    if (menuOpen) {
      lenis?.stop();
      document.body.style.overflow = 'hidden';
    } else {
      lenis?.start();
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [menuOpen]);
  const onPreloaderDone = useCallback(() => {
    setReady(true);
    sceneState.set({ ready: true });
    ScrollTrigger.refresh();
  }, []);

  useEffect(() => {
    if (reducedMotion) {
      setReady(true);
      sceneState.set({ ready: true, reducedMotion: true });
    }
  }, [reducedMotion]);

  useEffect(() => {
    const onMouse = (event: MouseEvent) => {
      sceneState.set({
        mouseX: (event.clientX / window.innerWidth) * 2 - 1,
        mouseY: -(event.clientY / window.innerHeight) * 2 + 1,
      });
    };
    window.addEventListener('mousemove', onMouse, { passive: true });
    return () => window.removeEventListener('mousemove', onMouse);
  }, []);

  useEffect(() => {
    if (reducedMotion) return;

    const lenis = new Lenis({
      duration: 1.2,
      smoothWheel: true,
      touchMultiplier: 1.15,
    });
    lenisRef.current = lenis;
    lenis.on('scroll', () => {
      ScrollTrigger.update();
      sceneState.set({ scrollVelocity: lenis.velocity });
    });

    const ticker = (time: number) => {
      lenis.raf(time * 1000);
    };
    gsap.ticker.add(ticker);
    gsap.ticker.lagSmoothing(0);

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
      lenis.start();
      closeMenu();
      lenis.scrollTo(el as HTMLElement, { offset: -72 });
    };
    document.addEventListener('click', onClick);

    return () => {
      gsap.ticker.remove(ticker);
      document.removeEventListener('click', onClick);
      sceneState.set({ scrollVelocity: 0 });
      lenis.destroy();
      lenisRef.current = null;
    };
  }, [reducedMotion]);

  useEffect(() => {
    if (!selectedProject) {
      document.body.style.overflow = '';
      lenisRef.current?.start();
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelectedProject(null);
    };
    document.body.style.overflow = 'hidden';
    lenisRef.current?.stop();
    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = '';
      lenisRef.current?.start();
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [selectedProject]);

  useGSAP(
    () => {
      if (reducedMotion || !ready) return;

      const setSection = (section: 'hero' | 'about' | 'work' | 'approach' | 'contact') => {
        sceneState.set({ section });
      };

      ScrollTrigger.create({
        trigger: document.documentElement,
        start: 'top top',
        end: 'bottom bottom',
        onUpdate: (self) => sceneState.set({ progress: self.progress }),
      });

      ScrollTrigger.create({
        trigger: '.hero',
        start: 'top center',
        end: 'bottom center',
        onEnter: () => setSection('hero'),
        onEnterBack: () => setSection('hero'),
      });
      ScrollTrigger.create({
        trigger: '#about',
        start: 'top center',
        end: 'bottom center',
        onEnter: () => setSection('about'),
        onEnterBack: () => setSection('about'),
      });
      ScrollTrigger.create({
        trigger: '#approach',
        start: 'top center',
        end: 'bottom center',
        onEnter: () => setSection('approach'),
        onEnterBack: () => setSection('approach'),
      });
      ScrollTrigger.create({
        trigger: '#now',
        start: 'top center',
        end: 'bottom bottom',
        onEnter: () => setSection('contact'),
        onEnterBack: () => setSection('contact'),
      });

      gsap.from('.hero .line', {
        yPercent: 110,
        duration: 1.15,
        stagger: 0.12,
        ease: 'power4.out',
      });
      gsap.from('.hero-copy, .actions, .hero-hud, .scroll-cue, .eyebrow', {
        opacity: 0,
        y: 18,
        duration: 0.9,
        stagger: 0.08,
        delay: 0.35,
        ease: 'power3.out',
      });

      gsap.utils.toArray<HTMLElement>('.section-title').forEach((title) => {
        const lines = title.querySelectorAll('.line');
        if (!lines.length) return;
        gsap.from(lines, {
          yPercent: 110,
          duration: 0.95,
          stagger: 0.08,
          ease: 'power4.out',
          scrollTrigger: { trigger: title, start: 'top 88%' },
        });
      });

      gsap.utils.toArray<HTMLElement>('.section-intro, .about-focus-item, .stack-item, .principle, .technology-card, .lifecycle-step').forEach((node, index) => {
        gsap.from(node, {
          opacity: 0,
          y: 28,
          duration: 0.85,
          delay: (index % 4) * 0.05,
          ease: 'power3.out',
          scrollTrigger: { trigger: node, start: 'top 88%' },
        });
      });

      ScrollTrigger.create({
        trigger: '#work',
        start: 'top center',
        end: 'bottom center',
        onEnter: () => setSection('work'),
        onEnterBack: () => setSection('work'),
      });

      const mm = gsap.matchMedia();
      mm.add('(min-width: 900px)', () => {
        const track = document.querySelector('.work-track') as HTMLElement | null;
        const pin = document.querySelector('.work-pin') as HTMLElement | null;
        if (!track || !pin) return;
        const distance = () => Math.max(0, track.scrollWidth - window.innerWidth + 160);
        gsap.to(track, {
          x: () => -distance(),
          ease: 'none',
          scrollTrigger: {
            trigger: pin,
            start: 'top 88px',
            end: () => `+=${distance()}`,
            pin: true,
            scrub: 0.7,
            anticipatePin: 1,
            onEnter: () => setSection('work'),
            onEnterBack: () => setSection('work'),
            onUpdate: (self) => {
              sceneState.set({
                section: 'work',
                workIndex: Math.min(
                  projects.length - 1,
                  Math.floor(self.progress * projects.length),
                ),
              });
            },
          },
        });
      });
      mm.add('(max-width: 899px)', () => {
        const panels = gsap.utils.toArray<HTMLElement>('.work-panel');
        if (!panels.length) return;
        const observer = new IntersectionObserver(
          (entries) => {
            const visible = entries
              .filter((entry) => entry.isIntersecting)
              .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
            const index = visible
              ? Number((visible.target as HTMLElement).dataset.workIndex ?? 0)
              : 0;
            sceneState.set({
              section: 'work',
              workIndex: Math.min(projects.length - 1, Math.max(0, index)),
            });
          },
          { rootMargin: '-30% 0px -40% 0px', threshold: [0.25, 0.5, 0.75] },
        );
        panels.forEach((panel) => observer.observe(panel));
        return () => observer.disconnect();
      });

      ScrollTrigger.refresh();
    },
    { scope: rootRef, dependencies: [ready, reducedMotion] },
  );

  const onSelectTechnology = (technology: Technology) => {
    setSelectedTechnology(technology);
    sceneState.set({ techIndex: technologies.findIndex((item) => item.name === technology.name) });
  };

  const onSelectStage = (index: number) => {
    setSelectedStage(index);
    sceneState.set({ techIndex: index });
  };

  return (
    <main className="portfolio-shell" id="site" ref={rootRef}>
      {!reducedMotion && !ready && <Preloader onDone={onPreloaderDone} />}
      {!reducedMotion && <Cursor />}
      {!reducedMotion && (
        <ErrorBoundary FallbackComponent={() => <SceneFallback />}>
          <SceneCanvas />
        </ErrorBoundary>
      )}
      <div className="grain" aria-hidden="true" />
      <InspectGrid ready={ready || reducedMotion} />
      <Nav menuOpen={menuOpen} onToggle={() => setMenuOpen((open) => !open)} onClose={closeMenu} />
      <Hero ready={ready || reducedMotion} />
      <About />
      <Work
        onOpenBrief={(project, origin) => {
          setBriefOrigin(origin);
          setSelectedProject(project);
        }}
      />
      <Approach
        selectedTechnology={selectedTechnology}
        selectedStage={selectedStage}
        onSelectTechnology={onSelectTechnology}
        onSelectStage={onSelectStage}
      />
      <Contact />
      {selectedProject && (
        <BriefDialog
          project={selectedProject}
          origin={briefOrigin}
          onClose={() => {
            setSelectedProject(null);
            setBriefOrigin(null);
          }}
        />
      )}
    </main>
  );
}
