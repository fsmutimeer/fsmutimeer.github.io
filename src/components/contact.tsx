"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowUp, ArrowUpRight, FileText, Github, Mail, Phone } from "lucide-react";
import { FaLinkedinIn } from "react-icons/fa";
import { profile } from "@/lib/profile";
import { scrambleTo } from "@/lib/scramble";
import { Magnetic } from "./magnetic";
import { SplitTitle } from "./split-title";

export function Contact() {
  const copyrightRef = useRef<HTMLSpanElement>(null);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > 300);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  useEffect(() => {
    const node = copyrightRef.current;
    if (!node) return;
    const finalText = node.textContent ?? "";
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries[0]?.isIntersecting) return;
        scrambleTo(node, finalText, 0.7);
        observer.disconnect();
      },
      { threshold: 0.55 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <section className="section now" id="now" aria-labelledby="now-heading">
      <span className="now-watermark" aria-hidden="true">
        Feroz
      </span>
      <div className="container now-grid">
        <div>
          <div className="section-label mono">04 / contact</div>
          <SplitTitle id="now-heading" lines={["Let's talk."]} />

          <div className="availability" data-testid="status-availability">
            <i /> Based in {profile.location}
            <span className="availability-tz" data-testid="text-location">
              · {profile.timezone}
            </span>
          </div>
        </div>

        <ul className="contact" id="contact" aria-label="Contact details">
          <li>
            <Link
              className="contact-row"
              href={`mailto:${profile.email}`}
              data-testid="link-email-contact"
              data-cursor="hover"
              data-cursor-label="mail"
            >
              <Mail size={16} className="contact-row-icon" aria-hidden="true" />
              <span className="contact-row-label mono">email</span>
              <span className="contact-row-value">{profile.email}</span>
              <ArrowUpRight size={16} className="contact-row-arrow" aria-hidden="true" />
            </Link>
          </li>
          <li>
            <a
              className="contact-row"
              href={profile.phoneHref}
              data-testid="link-phone-contact"
              data-cursor="hover"
              data-cursor-label="call"
            >
              <Phone size={16} className="contact-row-icon" aria-hidden="true" />
              <span className="contact-row-label mono">phone</span>
              <span className="contact-row-value">{profile.phone}</span>
              <ArrowUpRight size={16} className="contact-row-arrow" aria-hidden="true" />
            </a>
          </li>
        </ul>
      </div>

      <div className="container">
        <footer className="footer-bar">
          <span
            className="footer-copy"
            data-testid="text-footer-copyright"
            ref={copyrightRef}
          >
            © 2026 {profile.name} · {profile.role}
          </span>

          <nav className="footer-nav" aria-label="Page navigation">
            {[
              { label: "Home", href: "#top" },
              { label: "About", href: "#about" },
              { label: "Experience", href: "#experience" },
              { label: "Work", href: "#work" },
            ].map(({ label, href }) => (
              <a key={href} className="footer-nav-link" href={href}>
                {label}
              </a>
            ))}
          </nav>

          <nav className="footer-socials" aria-label="Profiles">
            <Magnetic strength={0.15}>
              <Link
                className="dock-item"
                href={profile.githubUrl}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={`GitHub @${profile.handle}`}
                data-testid="link-github"
                data-cursor="hover"
                data-cursor-label="github"
              >
                <span className="dock-tooltip mono">GitHub</span>
                <Github size={17} aria-hidden="true" />
              </Link>
            </Magnetic>
            <Magnetic strength={0.15}>
              <Link
                className="dock-item"
                href={profile.linkedinUrl}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={`LinkedIn @${profile.handle}`}
                data-testid="link-linkedin"
                data-cursor="hover"
                data-cursor-label="linkedin"
              >
                <span className="dock-tooltip mono">LinkedIn</span>
                <FaLinkedinIn size={15} aria-hidden="true" />
              </Link>
            </Magnetic>
            <Magnetic strength={0.15}>
              <Link
                className="dock-item"
                href={profile.cvUrl}
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Download CV"
                data-testid="link-cv"
                data-cursor="hover"
                data-cursor-label="cv"
              >
                <span className="dock-tooltip mono">Resume / CV</span>
                <FileText size={17} aria-hidden="true" />
              </Link>
            </Magnetic>
          </nav>
        </footer>
      </div>

      <button
        type="button"
        className={`scroll-to-top-btn ${scrolled ? "is-visible" : ""}`}
        onClick={scrollToTop}
        aria-label="Scroll to top"
        data-cursor="hover"
        data-cursor-label="top"
      >
        <span className="scroll-to-top-tooltip mono">Back to top</span>
        <ArrowUp size={22} strokeWidth={2.5} aria-hidden="true" />
      </button>
    </section>
  );
}
