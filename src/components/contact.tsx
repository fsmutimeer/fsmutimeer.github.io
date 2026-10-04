"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowUp,
  ExternalLink,
  Facebook,
  FileText,
  Github,
  Instagram,
  Mail,
  MapPin,
  Phone,
} from "lucide-react";
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
          </div>
        </div>
        <div className="contact" id="contact">
          <div className="contact-col">
            <div className="mono section-label contact-label">email</div>
            <div className="contact-emails-list">
              {profile.emails.map((email) => (
                <Magnetic strength={0.12} key={email}>
                  <Link
                    className="contact-email"
                    href={`mailto:${email}`}
                    data-testid={`link-email-${email}`}
                    data-cursor="hover"
                    data-cursor-label="mail"
                  >
                    {email}{" "}
                    <ExternalLink
                      size={17}
                      style={{ verticalAlign: "-2px", color: "var(--acid)" }}
                      aria-hidden="true"
                    />
                  </Link>
                </Magnetic>
              ))}
            </div>
          </div>

          <div className="contact-col">
            <div className="mono section-label contact-label">phone</div>
            <Magnetic strength={0.12}>
              <a
                className="contact-email"
                href="tel:+923337022773"
                data-testid="link-phone-contact"
                data-cursor="hover"
                data-cursor-label="call"
              >
                +92 333 7022773{" "}
                <ExternalLink
                  size={17}
                  style={{ verticalAlign: "-2px", color: "var(--acid)" }}
                  aria-hidden="true"
                />
              </a>
            </Magnetic>
          </div>
        </div>
      </div>

      <div className="container">
        <footer>
          <nav className="footer-nav" aria-label="Page navigation">
            {[
              { label: "Home", href: "#hero" },
              { label: "About", href: "#about" },
              { label: "Experience", href: "#experience" },
              { label: "Work", href: "#work" },
            ].map(({ label, href }) => (
              <a key={href} className="footer-nav-link" href={href}>
                {label}
              </a>
            ))}
          </nav>

          <div className="footer-dock-wrap">
            <nav className="footer-dock" aria-label="Social and professional links dock">
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
                  <Github size={20} aria-hidden="true" />
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
                  <FaLinkedinIn size={18} aria-hidden="true" />
                </Link>
              </Magnetic>

              <Magnetic strength={0.15}>
                <Link
                  className="dock-item"
                  href={profile.facebookUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={`Facebook @${profile.handle}`}
                  data-testid="link-facebook"
                  data-cursor="hover"
                  data-cursor-label="facebook"
                >
                  <span className="dock-tooltip mono">Facebook</span>
                  <Facebook size={20} aria-hidden="true" />
                </Link>
              </Magnetic>

              <Magnetic strength={0.15}>
                <Link
                  className="dock-item"
                  href={profile.instagramUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={`Instagram @${profile.handle}`}
                  data-testid="link-instagram"
                  data-cursor="hover"
                  data-cursor-label="instagram"
                >
                  <span className="dock-tooltip mono">Instagram</span>
                  <Instagram size={20} aria-hidden="true" />
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
                  <FileText size={20} aria-hidden="true" />
                </Link>
              </Magnetic>

              <div className="dock-divider" aria-hidden="true" />

              <Magnetic strength={0.15}>
                <Link
                  className="dock-item"
                  href={`mailto:${profile.email}`}
                  aria-label={profile.email}
                  data-testid="link-email-footer"
                  data-cursor="hover"
                  data-cursor-label="mail"
                >
                  <span className="dock-tooltip mono">Email</span>
                  <Mail size={20} aria-hidden="true" />
                </Link>
              </Magnetic>

              <Magnetic strength={0.15}>
                <Link
                  className="dock-item"
                  href={profile.phoneHref}
                  aria-label={profile.phone}
                  data-testid="link-phone"
                  data-cursor="hover"
                  data-cursor-label="call"
                >
                  <span className="dock-tooltip mono">Call</span>
                  <Phone size={20} aria-hidden="true" />
                </Link>
              </Magnetic>
            </nav>
          </div>

          <div className="footer-bottom-bar">
            <span data-testid="text-footer-copyright" ref={copyrightRef}>
              © 2026 {profile.name} · {profile.role}
            </span>
            <div className="footer-meta-group">
              <span
                className="footer-meta"
                data-testid="text-location"
                title={profile.location}
              >
                <MapPin size={15} aria-hidden="true" />
                <span>{profile.timezone}</span>
              </span>

            </div>
          </div>
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
