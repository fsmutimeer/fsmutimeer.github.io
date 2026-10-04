"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { navItems } from "@/lib/content";
import { profile } from "@/lib/profile";
import { sceneState, type SceneSection } from "@/lib/scene-state";
import { withBasePath } from "@/lib/base-path";
import { useModalFocus } from "@/lib/use-modal-focus";
import { Magnetic } from "./magnetic";
import { scrambleTo } from "@/lib/scramble";

const homeSection = {
  id: "top",
  label: "Home",
  preview: "Backend and platform engineer. Java, Quarkus, Kafka, and OpenShift.",
  href: undefined,
} as const;
const sections = [homeSection, ...navItems] as const;

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

  return (
    <span ref={ref} className="nav-overlay-label">
      {text}
    </span>
  );
}

export function Nav({
  menuOpen,
  onToggle,
  onClose,
  minimal = false,
}: {
  menuOpen: boolean;
  onToggle: () => void;
  onClose: () => void;
  /** When true, strip portfolio HUD chrome (counter and company). */
  minimal?: boolean;
}) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLElement>(null);
  const maskRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const isHomePage = pathname === "/" || pathname === "";
  const brandHref = isHomePage ? "#top" : withBasePath("/");
  const [activeId, setActiveId] = useState("top");
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const previewId = hoveredId ?? activeId;
  const previewItem =
    sections.find((item) => item.id === previewId) ?? sections[0];
  const previewIndex = Math.max(
    0,
    navItems.findIndex((item) => item.id === previewItem.id) + 1,
  );
  const activeIndex = Math.max(
    0,
    navItems.findIndex((item) => item.id === activeId) + 1,
  );

  useModalFocus(dialogRef, menuOpen, onClose);

  useEffect(() => {
    return sceneState.subscribe((snapshot) => {
      const sectionMap: Record<SceneSection, string> = {
        hero: "top",
        about: "about",
        experience: "experience",
        work: "work",
        approach: "work",
        contact: "now",
      };
      if (snapshot.section && sectionMap[snapshot.section]) {
        setActiveId(sectionMap[snapshot.section]);
      }
    });
  }, []);

  useEffect(() => {
    if (!isHomePage) {
      setActiveId("about");
      return;
    }
    const updateActiveSection = () => {
      const focalY = 160;
      for (let i = sections.length - 1; i >= 0; i--) {
        const id = sections[i].id;
        const el = document.getElementById(id);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= focalY && rect.bottom > focalY) {
            setActiveId(id);
            return;
          }
        }
      }
    };
    updateActiveSection();
    window.addEventListener("scroll", updateActiveSection, { passive: true });
    return () => window.removeEventListener("scroll", updateActiveSection);
  }, [isHomePage]);

  useEffect(() => {
    const update = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? Math.min(1, window.scrollY / max) : 0);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, []);

  useEffect(() => {
    if (!menuOpen) setHoveredId(null);
  }, [menuOpen]);

  useEffect(() => {
    const list = listRef.current;
    const mask = maskRef.current;
    if (!menuOpen || !list || !mask) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (window.matchMedia("(pointer: coarse)").matches) {
      gsap.set(mask, { autoAlpha: 0 });
      return;
    }

    const target = list.querySelector<HTMLElement>(
      `.nav-overlay-link[data-nav-id="${previewId}"]`,
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
        ease: "power3.out",
        overwrite: "auto",
        onComplete: () => {
          mask.dataset.placed = "true";
        },
      });
    };

    const instant = mask.dataset.placed !== "true";
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
    mask.dataset.placed = "";
    gsap.set(mask, { autoAlpha: 0, width: 0, height: 0, x: 0, y: 0 });
  }, [menuOpen]);

  useGSAP(
    () => {
      if (!menuOpen || !overlayRef.current) return;
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      gsap.fromTo(
        ".nav-overlay-link",
        { y: 36, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.55,
          stagger: 0.07,
          ease: "power3.out",
          delay: 0.08,
        },
      );
      gsap.fromTo(
        ".nav-overlay-meta > *",
        { y: 16, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.5,
          stagger: 0.08,
          ease: "power3.out",
          delay: 0.28,
        },
      );
    },
    { dependencies: [menuOpen] },
  );

  return (
    <header
      className="hud"
      data-open={menuOpen}
      ref={dialogRef}
      role={menuOpen ? "dialog" : undefined}
      aria-modal={menuOpen ? true : undefined}
      aria-label={menuOpen ? "Site index" : undefined}
    >
      {!minimal && (
        <div className="hud-progress" aria-hidden="true">
          <span style={{ transform: `scaleX(${progress})` }} />
        </div>
      )}
      <div className="container hud-bar">
        <div className="hud-start">
          <Link
            href={brandHref}
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
          </Link>
        </div>
        <div className="hud-end">
          {!minimal && (
            <div className="status" hidden={menuOpen}>
              <span className="status-dot" />
              <span>
                {String(activeIndex).padStart(2, "0")} /{" "}
                {String(navItems.length).padStart(2, "0")}
              </span>
              <span className="status-copy">backend & platform</span>
            </div>
          )}
          <button
            className="index-btn"
            type="button"
            ref={closeRef}
            aria-label={menuOpen ? "Close site menu" : "Open site menu"}
            aria-expanded={menuOpen}
            aria-controls="site-index"
            data-testid="button-mobile-menu"
            data-cursor="hover"
            data-cursor-label={menuOpen ? "close" : "menu"}
            onClick={onToggle}
          >
            <span className="index-btn-label mono">
              {menuOpen ? "Close" : "MENU"}
            </span>
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
      >
        <div className="container nav-overlay-grid">
          <nav
            className="nav-overlay-list"
            aria-label="Primary navigation"
            ref={listRef}
            onPointerLeave={() => setHoveredId(null)}
          >
            <div
              className="nav-overlay-mask"
              ref={maskRef}
              aria-hidden="true"
            />
            {sections.map((section, index) => {
              const { id, label } = section;
              const href =
                "href" in section && section.href
                  ? withBasePath(section.href)
                  : id === "top"
                    ? isHomePage
                      ? "#top"
                      : withBasePath("/")
                    : isHomePage
                      ? `#${id}`
                      : `${withBasePath("/")}#${id}`;
              const isPageLink = href.startsWith("/") && !href.startsWith("#");
              return (
                <Magnetic key={id} strength={0.08}>
                  <Link
                    className={`nav-overlay-link${previewItem.id === id ? " is-preview" : ""}`}
                    href={href}
                    data-nav-id={id}
                    data-testid={
                      id === "top" ? "link-nav-home" : `link-nav-${id}`
                    }
                    data-modal-autofocus={index === 0 ? "true" : undefined}
                    data-cursor="hover"
                    data-cursor-label={isPageLink ? "visit" : "open"}
                    aria-current={activeId === id ? "location" : undefined}
                    onPointerEnter={() => setHoveredId(id)}
                    onMouseEnter={() => setHoveredId(id)}
                    onFocus={() => setHoveredId(id)}
                    onClick={onClose}
                  >
                    <span className="nav-overlay-no mono">
                      {String(index === 0 ? 0 : index).padStart(2, "0")}
                    </span>
                    <OverlayLabel text={label} scramble={hoveredId === id} />
                  </Link>
                </Magnetic>
              );
            })}
          </nav>
          <aside className="nav-overlay-meta">
            <div className="nav-overlay-preview" aria-live="polite">
              <span className="nav-overlay-preview-no mono">
                {String(previewIndex).padStart(2, "0")}
              </span>
              <p className="nav-overlay-preview-label">{previewItem.label}</p>
              <p>{previewItem.preview}</p>
            </div>
            <p className="mono nav-overlay-kicker">currently</p>
            <p>
              Backend & platform engineer
              <br />
              {profile.location}
            </p>
            <Link
              href={`mailto:${profile.email}`}
              data-cursor="hover"
              data-cursor-label="mail"
            >
              {profile.email}
            </Link>
            <Link
              href={profile.cvUrl}
              target="_blank"
              rel="noopener noreferrer"
              data-cursor="hover"
              data-cursor-label="cv"
            >
              Download CV
            </Link>
          </aside>
        </div>
      </div>
    </header>
  );
}
