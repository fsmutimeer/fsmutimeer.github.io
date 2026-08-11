import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  ArrowDownRight,
  ArrowUpRight,
  Braces,
  Check,
  ChevronRight,
  Cloud,
  Container,
  Cpu,
  ExternalLink,
  Github,
  Globe2,
  Mail,
  MapPin,
  Menu,
  Network,
  Radio,
  ShieldCheck,
  Terminal,
  X,
  type LucideIcon,
} from 'lucide-react';
import './index.css';

type Project = {
  number: string;
  title: string;
  subtitle: string;
  copy: string;
  metrics: string[];
  tags: string[];
  detail: string;
  outcome: string;
  role: string;
};

const projects: Project[] = [
  {
    number: '01',
    title: 'The service that stopped paging us',
    subtitle: 'Quarkus → OpenShift / production platform',
    copy: 'Rebuilt a noisy Java integration layer as small, observable Quarkus services. The result was a safer deploy path, clearer ownership, and fewer 2am surprises.',
    metrics: ['−68% deploy time', '0 rollback guesswork'],
    tags: ['Quarkus', 'OpenShift', 'Grafana'],
    detail: 'A legacy Java integration layer had grown into a shared blast radius. I split the critical paths into focused Quarkus services, introduced health and readiness contracts, and moved delivery behind a repeatable OpenShift promotion flow.',
    outcome: 'Deploys became a routine, observable event instead of an incident-adjacent ritual.',
    role: 'Platform lead · service design · delivery ownership',
  },
  {
    number: '02',
    title: 'A cluster with a point of view',
    subtitle: 'OKD / Kubernetes platform enablement',
    copy: 'Created opinionated namespaces, golden paths, and runbooks for teams shipping into OKD. Good infrastructure should make the right thing the easy thing.',
    metrics: ['14 teams enabled', '31 runbooks shipped'],
    tags: ['OKD', 'Kubernetes', 'Argo CD'],
    detail: 'Teams were each solving the same Kubernetes questions in a different way. I shaped a small platform product: namespace defaults, service templates, GitOps promotion, policy guardrails, and runbooks that make the happy path obvious.',
    outcome: 'Fourteen teams moved from bespoke manifests to a shared operating language.',
    role: 'Platform product owner · Kubernetes · developer enablement',
  },
  {
    number: '03',
    title: 'Making failure legible',
    subtitle: 'DevOps / reliability engineering',
    copy: 'Connected logs, traces, alerts, and release signals into one operational story—so an incident starts with context instead of a scavenger hunt.',
    metrics: ['−42% MTTR', '99.96% service health'],
    tags: ['Prometheus', 'OpenTelemetry', 'GitLab CI'],
    detail: 'The team had plenty of telemetry and very little shared context. I connected OpenTelemetry traces to structured logs, tightened Prometheus alert intent, and surfaced release markers in the same operational views.',
    outcome: 'The first question in an incident changed from “where do I look?” to “what changed?”',
    role: 'Reliability engineer · observability · CI/CD',
  },
];

const stack: [string, string, string, LucideIcon][] = [
  ['01', 'Java & Quarkus', 'Services that start fast, stay small, and expose their seams.', Braces],
  ['02', 'Containers & clusters', 'Kubernetes primitives with the operational detail filled in.', Container],
  ['03', 'Delivery systems', 'Pipelines, promotion, policy, and a rollback you can trust.', Radio],
  ['04', 'Observability', 'Signals that explain what happened, not just that something did.', Network],
];

function Reveal({ children, delay = 0, className = '' }: { children: ReactNode; delay?: number; className?: string }) {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = ref.current;
    if (!node || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setVisible(true);
        observer.disconnect();
      }
    }, { threshold: 0.08 });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return <div ref={ref} className={`reveal ${visible ? 'is-visible' : ''} ${className}`} style={{ transitionDelay: `${delay}ms` }}>{children}</div>;
}

function Pill({ children }: { children: ReactNode }) {
  return <span className="pill pill-accent" data-testid={`tag-${String(children).toLowerCase().replace(/\s/g, '-')}`}>{children}</span>;
}

function Portfolio() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);

  const closeMenu = () => setMenuOpen(false);
  const openBrief = (project: Project) => setSelectedProject(project);

  useEffect(() => {
    if (!selectedProject) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelectedProject(null);
    };
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [selectedProject]);

  return (
    <main className="portfolio-shell" id="top">
      <header className="topbar">
        <div className="container topbar-inner">
          <a href="#top" className="brand" data-testid="link-brand" onClick={closeMenu}>
            <span className="brand-mark">AM</span>
            <span>Alex Morgan<span style={{ color: 'var(--acid)' }}>.</span></span>
          </a>
          <nav className="nav" aria-label="Primary navigation">
            {['work', 'approach', 'now', 'contact'].map((item) => (
              <a key={item} href={`#${item}`} data-testid={`link-nav-${item}`}>{item}</a>
            ))}
          </nav>
          <div className="status"><span className="status-dot" /> open to the right problem</div>
          <button className="menu-btn" type="button" aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'} aria-expanded={menuOpen} aria-controls="mobile-navigation" data-testid="button-mobile-menu" onClick={() => setMenuOpen((open) => !open)}>
            {menuOpen ? <X size={21} aria-hidden="true" /> : <Menu size={21} aria-hidden="true" />}
          </button>
          {menuOpen && (
            <nav className="mobile-nav" id="mobile-navigation" aria-label="Mobile navigation">
              {['work', 'approach', 'now', 'contact'].map((item) => (
                <a key={item} href={`#${item}`} data-testid={`link-mobile-${item}`} onClick={closeMenu}>{item}</a>
              ))}
            </nav>
          )}
        </div>
      </header>

      <section className="container hero" aria-labelledby="hero-heading">
        <Reveal>
          <div>
            <div className="eyebrow mono">platform engineer / available now</div>
            <h1 id="hero-heading">Code that survives<br /><em>production.</em></h1>
            <p className="hero-copy">I’m <strong>Alex Morgan</strong> — a hands-on engineer turning Java services into reliable, cloud-native systems. From a Quarkus commit to a healthy pod on OpenShift, I own the path in between.</p>
            <div className="actions">
              <a className="button" href="#work" data-testid="link-selected-work">See selected work <ArrowDownRight size={16} aria-hidden="true" /></a>
              <a className="text-link" href="#contact" data-testid="link-start-conversation">Start a conversation <ArrowUpRight size={14} aria-hidden="true" /></a>
            </div>
          </div>
        </Reveal>
        <Reveal delay={160}>
          <div className="terminal" aria-label="Alex Morgan platform status terminal">
            <div className="terminal-head"><i className="term-dot" /><i className="term-dot" /><i className="term-dot" /><span className="term-title mono">alex@platform ~/signal</span></div>
            <div className="terminal-body mono">
              <span className="terminal-line"><span className="prompt">$</span> <span className="command">whoami</span></span>
              <span className="terminal-line terminal-result">alex.morgan — platform engineer</span><br />
              <span className="terminal-line"><span className="prompt">$</span> <span className="command">cat operating_principles.md</span></span>
              <span className="terminal-line terminal-result">01 / reduce the distance to production</span>
              <span className="terminal-line terminal-result">02 / make failure useful</span>
              <span className="terminal-line terminal-result">03 / leave the system clearer</span><br />
              <span className="terminal-line"><span className="prompt">$</span> <span className="command">kubectl get signal</span></span>
              <span className="terminal-line terminal-result" style={{ color: 'var(--acid)' }}>platform&nbsp;&nbsp; online&nbsp;&nbsp; 99.96%&nbsp;&nbsp; <Check size={12} style={{ verticalAlign: '-2px' }} aria-hidden="true" /></span><br />
              <span className="terminal-line"><span className="prompt">$</span> <span className="cursor" aria-hidden="true" /></span>
            </div>
          </div>
        </Reveal>
        <div className="scroll-cue mono"><span /> scroll to inspect the system</div>
      </section>

      <section className="section" id="work" aria-labelledby="work-heading">
        <div className="container">
          <Reveal>
            <div className="work-head">
              <div><div className="section-label mono">01 / selected systems</div><h2 className="section-title" id="work-heading">The work behind<br />the uptime.</h2></div>
              <p className="section-intro">Not a museum of logos. A field guide to the moments where good engineering made the system—and the team—calmer.</p>
            </div>
          </Reveal>
          <div className="work-list">
            {projects.map((project, index) => (
              <Reveal delay={index * 100} key={project.number}>
                <article className="project" data-testid={`card-project-${project.number}`}>
                  <div className="project-no mono">{project.number}</div>
                  <div>
                    <div className="project-sub mono">{project.subtitle}</div>
                    <h3>{project.title}</h3>
                    <p data-testid={`text-project-copy-${project.number}`}>{project.copy}</p>
                    <div className="tags">{project.tags.map((tag) => <Pill key={tag}>{tag}</Pill>)}</div>
                  </div>
                  <div className="project-side">
                    <div className="metric mono">{project.metrics[0]}</div>
                    <div className="metric mono">{project.metrics[1]}</div>
                    <button className="text-link brief-button" type="button" data-testid={`button-read-brief-${project.number}`} onClick={() => openBrief(project)}>Read the brief <ChevronRight size={12} aria-hidden="true" /></button>
                  </div>
                </article>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="section" id="approach" aria-labelledby="approach-heading">
        <div className="container">
          <div className="split">
            <Reveal>
              <div>
                <div className="section-label mono">02 / the capability story</div>
                <h2 className="section-title" id="approach-heading">A stack is only useful when it tells a story.</h2>
                <p className="section-intro">I work across the seam between application code and platform reality. That means knowing what a service needs, what a cluster can promise, and where the two will disagree at 03:17.</p>
              </div>
            </Reveal>
            <Reveal delay={120}>
              <div className="stack-list" data-testid="list-capabilities">
                {stack.map(([number, title, copy, Icon]) => (
                  <div className="stack-item" key={number} data-testid={`row-capability-${number}`}>
                    <span className="stack-no mono">{number}</span><div><h3>{title}</h3><p>{copy}</p></div><Icon className="stack-arrow" size={18} aria-hidden="true" />
                  </div>
                ))}
              </div>
            </Reveal>
          </div>
          <div className="principles">
            <Reveal delay={80}><div className="principle" data-testid="card-principle-platform"><Cloud className="principle-icon" size={22} aria-hidden="true" /><h3>Platform as product</h3><p>Clear paths, useful defaults, and documentation that respects the person on call.</p></div></Reveal>
            <Reveal delay={160}><div className="principle" data-testid="card-principle-reliability"><ShieldCheck className="principle-icon" size={22} aria-hidden="true" /><h3>Reliability is a feature</h3><p>Health checks, graceful failure, and delivery signals are part of the design—not cleanup.</p></div></Reveal>
            <Reveal delay={240}><div className="principle" data-testid="card-principle-practical"><Cpu className="principle-icon" size={22} aria-hidden="true" /><h3>Curious, then practical</h3><p>I like new tools. I like them more when they make tomorrow’s incident smaller.</p></div></Reveal>
          </div>
        </div>
      </section>

      <section className="section now" id="now" aria-labelledby="now-heading">
        <div className="container now-grid">
          <Reveal>
            <div>
              <div className="section-label mono">03 / current signal</div>
              <h2 className="section-title" id="now-heading">Looking for a team that ships thoughtfully.</h2>
              <p className="section-intro">I’m open to platform engineering and senior backend roles where I can work close to the code, the cluster, and the people operating both.</p>
              <div className="availability" data-testid="status-availability"><i /> available for conversations · remote / hybrid</div>
            </div>
          </Reveal>
          <Reveal delay={130}>
            <div className="contact" id="contact">
              <div><div className="mono section-label" style={{ marginBottom: 12 }}>route open</div><a className="contact-email" href="mailto:alex.morgan@signal.dev" data-testid="link-email-contact">alex.morgan@signal.dev <ExternalLink size={17} style={{ verticalAlign: '-2px', color: 'var(--acid)' }} aria-hidden="true" /></a></div>
              <p className="contact-note">Have a hard platform problem or a team building its first one? Tell me what’s breaking.</p>
            </div>
          </Reveal>
        </div>
      </section>

      <div className="container">
        <footer>
          <span data-testid="text-footer-copyright">© 2024 Alex Morgan · platform engineer</span>
          <div className="footer-links">
            <a href="#top" data-testid="link-back-to-top"><Terminal size={12} style={{ verticalAlign: '-2px' }} aria-hidden="true" /> back to top</a>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" data-testid="link-github"><Github size={12} style={{ verticalAlign: '-2px' }} aria-hidden="true" /> github</a>
            <a href="mailto:alex.morgan@signal.dev" data-testid="link-email-footer"><Mail size={12} style={{ verticalAlign: '-2px' }} aria-hidden="true" /> email</a>
            <span data-testid="text-location"><MapPin size={12} style={{ verticalAlign: '-2px' }} aria-hidden="true" /> UTC−05</span>
            <Globe2 size={12} aria-label="Remote and hybrid work" />
          </div>
        </footer>
      </div>

      {selectedProject && (
        <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setSelectedProject(null); }}>
          <section className="brief-dialog" role="dialog" aria-modal="true" aria-labelledby="brief-title" data-testid="dialog-project-brief">
            <div className="dialog-head">
              <div><div className="dialog-number">{selectedProject.number} / SYSTEM BRIEF</div><h2 className="dialog-title" id="brief-title">{selectedProject.title}</h2></div>
              <button className="dialog-close" type="button" aria-label="Close project brief" data-testid="button-close-brief" onClick={() => setSelectedProject(null)}><X size={18} aria-hidden="true" /></button>
            </div>
            <div className="dialog-body">
              <div className="project-sub mono">{selectedProject.subtitle}</div>
              <p>{selectedProject.detail}</p>
              <div className="detail-grid">
                <div className="detail-box"><strong>Signal</strong><span>{selectedProject.metrics.join(' · ')}</span></div>
                <div className="detail-box"><strong>Role</strong><span>{selectedProject.role}</span></div>
                <div className="detail-box"><strong>Outcome</strong><span>{selectedProject.outcome}</span></div>
                <div className="detail-box"><strong>Tools in the path</strong><span>{selectedProject.tags.join(' · ')}</span></div>
              </div>
              <div className="dialog-footer"><a className="text-link" href="#contact" data-testid="link-brief-contact" onClick={() => setSelectedProject(null)}>Talk through a similar problem <ArrowUpRight size={14} aria-hidden="true" /></a></div>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

export default Portfolio;