'use client';

import { useEffect, useRef } from 'react';
import { ExternalLink, FileText, Github, Mail, MapPin, Phone, Terminal } from 'lucide-react';
import { FaLinkedinIn } from 'react-icons/fa';
import { profile } from '@/lib/profile';
import { scrambleTo } from '@/lib/scramble';
import { Magnetic } from './magnetic';

export function Contact() {
  const copyrightRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const node = copyrightRef.current;
    if (!node) return;
    const finalText = node.textContent ?? '';
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
      <div className="container now-grid">
        <div>
          <div className="section-label mono">05 / current signal</div>
          <h2 className="section-title" id="now-heading">
            Currently shipping at {profile.company}
          </h2>
          <p className="section-intro">
            Backend and platform work in {profile.location} — Quarkus services, OpenShift clusters, and
            the GitOps path between them. If you have a hard service or cluster problem, tell me
            what’s breaking.
          </p>
          <div className="availability" data-testid="status-availability">
            <i /> {profile.company} · {profile.location}
          </div>
        </div>
        <div className="contact" id="contact">
          <div>
            <div className="mono section-label" style={{ marginBottom: 12 }}>
              route open
            </div>
            <Magnetic strength={0.12}>
              <a
                className="contact-email"
                href={`mailto:${profile.email}`}
                data-testid="link-email-contact"
                data-cursor="hover"
                data-cursor-label="mail"
              >
                {profile.email}{' '}
                <ExternalLink
                  size={17}
                  style={{ verticalAlign: '-2px', color: 'var(--acid)' }}
                  aria-hidden="true"
                />
              </a>
            </Magnetic>
          </div>
          <p className="contact-note">
            Have a hard service or cluster problem, or a team building its first one? Tell me what’s
            breaking.{' '}
            <a
              href={profile.cvUrl}
              target="_blank"
              rel="noopener noreferrer"
              data-testid="link-cv-contact"
              data-cursor="hover"
              data-cursor-label="cv"
            >
              Download the CV
            </a>
            .
          </p>
        </div>
      </div>
      <div className="container">
        <footer>
          <span data-testid="text-footer-copyright" ref={copyrightRef}>
            © 2026 {profile.name} · {profile.role}
          </span>
          <div className="footer-links">
            <a
              href="#top"
              aria-label="Back to top"
              title="Back to top"
              data-testid="link-back-to-top"
              data-cursor="hover"
              data-cursor-label="home"
            >
              <Terminal size={24} aria-hidden="true" />
            </a>
            <a
              href={profile.githubUrl}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`GitHub ${profile.handle}`}
              title={profile.handle}
              data-testid="link-github"
              data-cursor="hover"
            >
              <Github size={24} aria-hidden="true" />
            </a>
            <a
              href={profile.linkedinUrl}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`LinkedIn ${profile.handle}`}
              title={profile.handle}
              data-testid="link-linkedin"
              data-cursor="hover"
            >
              <FaLinkedinIn size={24} aria-hidden="true" />
            </a>
            <a
              href={profile.cvUrl}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Download CV"
              title="CV"
              data-testid="link-cv"
              data-cursor="hover"
              data-cursor-label="cv"
            >
              <FileText size={24} aria-hidden="true" />
            </a>
            <a
              href={`mailto:${profile.email}`}
              aria-label={profile.email}
              title={profile.email}
              data-testid="link-email-footer"
              data-cursor="hover"
              data-cursor-label="mail"
            >
              <Mail size={24} aria-hidden="true" />
            </a>
            <a
              href={profile.phoneHref}
              aria-label={profile.phone}
              title={profile.phone}
              data-testid="link-phone"
              data-cursor="hover"
              data-cursor-label="call"
            >
              <Phone size={24} aria-hidden="true" />
            </a>
            <span className="footer-meta" data-testid="text-location" title={profile.location}>
              <MapPin size={24} aria-hidden="true" />
              <span>{profile.timezone}</span>
            </span>
          </div>
        </footer>
      </div>
    </section>
  );
}
