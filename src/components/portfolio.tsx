"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import Lenis from "lenis";
import { projects } from "@/lib/content";
import { sceneState, type SceneSection } from "@/lib/scene-state";
import { usePrefersReducedMotion } from "@/lib/use-reduced-motion";
import { About } from "./about";
import { Contact } from "./contact";
import { Cursor } from "./cursor";
import { Experience } from "./experience";
import { Hero } from "./hero";
import { InspectGrid } from "./inspect-grid";
import { Nav } from "./nav";
import { Preloader } from "./preloader";
import { Work } from "./work";
import "lenis/dist/lenis.css";

gsap.registerPlugin(ScrollTrigger, useGSAP);

let hasPreloadedInSession = false;

function subscribe() {
  return () => {};
}

function getSnapshot() {
  if (hasPreloadedInSession) return true;
  if (typeof window !== "undefined") {
    try {
      if (sessionStorage.getItem("portfolio_preloaded") === "true") {
        hasPreloadedInSession = true;
        return true;
      }
    } catch {
      // Ignore storage errors
    }
  }
  return false;
}

function getServerSnapshot() {
  return false;
}


export function Portfolio() {
  const reducedMotion = usePrefersReducedMotion();
  const alreadyPreloaded = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const [menuOpen, setMenuOpen] = useState(false);
  const [ready, setReady] = useState(false);
  const isReady = ready || alreadyPreloaded || reducedMotion;
  const lenisRef = useRef<Lenis | null>(null);
  const rootRef = useRef<HTMLElement>(null);

  const closeMenu = () => setMenuOpen(false);

  useEffect(() => {
    const lenis = lenisRef.current;
    if (menuOpen) {
      lenis?.stop();
      document.body.style.overflow = "hidden";
    } else {
      lenis?.start();
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [menuOpen]);

  const onPreloaderDone = useCallback(() => {
    hasPreloadedInSession = true;
    try {
      sessionStorage.setItem("portfolio_preloaded", "true");
    } catch {}
    setReady(true);
    sceneState.set({ ready: true });
    ScrollTrigger.refresh();
  }, []);

  useEffect(() => {
    if (alreadyPreloaded && !ready) {
      setReady(true);
      sceneState.set({ ready: true });
      ScrollTrigger.refresh();
    }
  }, [alreadyPreloaded, ready]);

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
    window.addEventListener("mousemove", onMouse, { passive: true });
    return () => window.removeEventListener("mousemove", onMouse);
  }, []);

  useEffect(() => {
    if (reducedMotion) return;

    const lenis = new Lenis({
      duration: 1.2,
      smoothWheel: true,
      touchMultiplier: 1.15,
    });
    lenisRef.current = lenis;
    if (typeof window !== "undefined") {
      (window as unknown as { __lenis?: Lenis }).__lenis = lenis;
    }
    lenis.on("scroll", () => {
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
      const hash = target.getAttribute("href");
      if (!hash || hash === "#") return;
      const el = document.querySelector(hash);
      if (!el) return;
      event.preventDefault();
      lenis.start();
      closeMenu();
      lenis.scrollTo(el as HTMLElement, { offset: -72 });
    };
    document.addEventListener("click", onClick);

    return () => {
      gsap.ticker.remove(ticker);
      document.removeEventListener("click", onClick);
      sceneState.set({ scrollVelocity: 0 });
      if (typeof window !== "undefined") {
        delete (window as unknown as { __lenis?: Lenis }).__lenis;
      }
      lenis.destroy();
      lenisRef.current = null;
    };
  }, [reducedMotion]);


  useGSAP(
    () => {
      if (reducedMotion || !isReady) return;

      const setSection = (section: SceneSection) => {
        sceneState.set({ section });
      };

      ScrollTrigger.create({
        trigger: document.documentElement,
        start: "top top",
        end: "bottom bottom",
        onUpdate: (self) => sceneState.set({ progress: self.progress }),
      });

      ScrollTrigger.create({
        trigger: ".hero",
        start: "top center",
        end: "bottom center",
        onEnter: () => setSection("hero"),
        onEnterBack: () => setSection("hero"),
      });
      ScrollTrigger.create({
        trigger: "#about",
        start: "top center",
        end: "bottom center",
        onEnter: () => setSection("about"),
        onEnterBack: () => setSection("about"),
      });
      ScrollTrigger.create({
        trigger: "#experience",
        start: "top center",
        end: "bottom center",
        onEnter: () => setSection("experience"),
        onEnterBack: () => setSection("experience"),
      });
      ScrollTrigger.create({
        trigger: "#now",
        start: "top 85%",
        end: "bottom bottom",
        onEnter: () => setSection("contact"),
        onEnterBack: () => setSection("contact"),
      });

      gsap.from(".hero .line", {
        yPercent: 110,
        duration: 1.15,
        stagger: 0.12,
        ease: "power4.out",
      });
      gsap.from(
        ".hero-copy, .hero-stack, .actions, .hero-hud, .scroll-cue, .eyebrow",
        {
          opacity: 0,
          y: 18,
          duration: 0.9,
          stagger: 0.08,
          delay: 0.35,
          ease: "power3.out",
        },
      );

      gsap.utils.toArray<HTMLElement>(".section-title").forEach((title) => {
        const lines = title.querySelectorAll(".line");
        if (!lines.length) return;
        gsap.from(lines, {
          yPercent: 110,
          duration: 0.95,
          stagger: 0.08,
          ease: "power4.out",
          scrollTrigger: { trigger: title, start: "top 88%" },
        });
      });

      gsap.utils
        .toArray<HTMLElement>(
          ".section-intro, .about-cap-row, .experience-item",
        )
        .forEach((node, index) => {
          gsap.from(node, {
            opacity: 0,
            y: 28,
            duration: 0.85,
            delay: (index % 4) * 0.05,
            ease: "power3.out",
            scrollTrigger: { trigger: node, start: "top 88%" },
          });
        });

      ScrollTrigger.create({
        trigger: "#work",
        start: "top center",
        end: "bottom center",
        onEnter: () => setSection("work"),
        onEnterBack: () => setSection("work"),
      });

      // ── Desktop: Horizontal scroll track with smooth centered rise ────────
      const mm = gsap.matchMedia();
      mm.add("(min-width: 900px)", () => {
        const pin = document.querySelector(".work-pin") as HTMLElement | null;
        const track = document.querySelector(".work-track") as HTMLElement | null;
        const cards = gsap.utils.toArray<HTMLElement>(".work-panel-magnet");
        if (!pin || !track || cards.length < 2) return;

        const steps = cards.length - 1;
        const getStepDistance = () => window.innerHeight * 0.9;
        const rise = 90;

        // Initial state: card 0 centred; the rest wait below, transparent
        gsap.set(cards[0], { y: 0, scale: 1, opacity: 1 });
        for (let i = 1; i < cards.length; i++) {
          gsap.set(cards[i], { y: rise, scale: 0.97, opacity: 0 });
        }

        const timeline = gsap.timeline({
          defaults: { ease: "none" },
          scrollTrigger: {
            trigger: pin,
            start: "top 88px",
            end: () => `+=${Math.round(getStepDistance() * steps * 1.15)}`,
            pin: true,
            scrub: 0.8,
            invalidateOnRefresh: true,
            id: "work-pinned-track",
            onEnter: () => setSection("work"),
            onEnterBack: () => setSection("work"),
            onLeave: () => setSection("contact"),
            onLeaveBack: () => setSection("experience"),
            onUpdate: (self) => {
              const activeIdx = Math.min(
                projects.length - 1,
                Math.max(0, Math.round(self.progress * steps)),
              );
              if (self.isActive) {
                sceneState.set({ section: "work", workIndex: activeIdx });
              } else {
                sceneState.set({ workIndex: activeIdx });
              }
              cards.forEach((c, idx) => {
                const panel = c.querySelector(".work-panel");
                if (panel) {
                  if (idx === activeIdx) {
                    panel.classList.add("is-active");
                  } else {
                    panel.classList.remove("is-active");
                  }
                }
              });
            },
          },
        });

        // Each step: hold, current card fades away upward, then the next rises in.
        // Tweens span [step + 0.15, step + 0.85] so each card rests at integer progress.
        for (let step = 0; step < steps; step++) {
          timeline.to(
            cards[step],
            { y: -rise, scale: 0.97, opacity: 0, ease: "power2.in", duration: 0.4 },
            step + 0.15,
          );
          timeline.to(
            cards[step + 1],
            { y: 0, scale: 1, opacity: 1, ease: "power2.out", duration: 0.4 },
            step + 0.45,
          );
        }
        // Pad the timeline so its duration equals `steps` (progress = index / steps)
        timeline.set({}, {}, steps);

        return () => {
          timeline.scrollTrigger?.kill();
          timeline.kill();
        };
      });

      // ── Mobile: IntersectionObserver for active card ──────────────────────
      mm.add("(max-width: 899px)", () => {
        const panels = gsap.utils.toArray<HTMLElement>(".work-panel");
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
              section: "work",
              workIndex: Math.min(projects.length - 1, Math.max(0, index)),
            });
          },
          { rootMargin: "-30% 0px -40% 0px", threshold: [0.25, 0.5, 0.75] },
        );
        panels.forEach((panel) => observer.observe(panel));
        return () => observer.disconnect();
      });

      ScrollTrigger.refresh();
    },
    { scope: rootRef, dependencies: [isReady, reducedMotion] },
  );

  useEffect(() => {
    if (!isReady || reducedMotion) return;
    const hash = window.location.hash;
    if (hash && hash.length > 1) {
      const el = document.querySelector(hash);
      if (el && lenisRef.current) {
        const timer = setTimeout(() => {
          lenisRef.current?.scrollTo(el as HTMLElement, { offset: -72, immediate: true });
        }, 120);
        return () => clearTimeout(timer);
      }
    }
  }, [isReady, reducedMotion]);

  return (
    <main className="portfolio-shell" id="site" ref={rootRef}>
      {!reducedMotion && !isReady && <Preloader onDone={onPreloaderDone} />}
      {!reducedMotion && <Cursor />}
      <div className="grain" aria-hidden="true" />
      <InspectGrid ready={isReady} />
      <Nav
        menuOpen={menuOpen}
        onToggle={() => setMenuOpen((open) => !open)}
        onClose={closeMenu}
      />
      <Hero ready={isReady} />
      <About />
      <Experience />
      <Work />
      <Contact />
    </main>
  );
}
