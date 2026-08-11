import { useEffect, useState, type ReactNode } from "react";
import {
  ArrowDownRight,
  ArrowUpRight,
  Braces,
  Check,
  ChevronRight,
  Cloud,
  Code2,
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
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

const projects = [
  {
    number: "01",
    title: "The service that stopped paging us",
    subtitle: "Quarkus → OpenShift / production platform",
    copy: "Rebuilt a noisy Java integration layer as small, observable Quarkus services. The result was a safer deploy path, clearer ownership, and fewer 2am surprises.",
    metrics: ["−68% deploy time", "0 rollback guesswork"],
    tags: ["Quarkus", "OpenShift", "Grafana"],
    tone: "lime",
  },
  {
    number: "02",
    title: "A cluster with a point of view",
    subtitle: "OKD / Kubernetes platform enablement",
    copy: "Created opinionated namespaces, golden paths, and runbooks for teams shipping into OKD. Good infrastructure should make the right thing the easy thing.",
    metrics: ["14 teams enabled", "31 runbooks shipped"],
    tags: ["OKD", "Kubernetes", "Argo CD"],
    tone: "cyan",
  },
  {
    number: "03",
    title: "Making failure legible",
    subtitle: "DevOps / reliability engineering",
    copy: "Connected logs, traces, alerts, and release signals into one operational story—so an incident starts with context instead of a scavenger hunt.",
    metrics: ["−42% MTTR", "99.96% service health"],
    tags: ["Prometheus", "OpenTelemetry", "GitLab CI"],
    tone: "amber",
  },
];

const stack: [string, string, string, LucideIcon][] = [
  ["01", "Java & Quarkus", "Services that start fast, stay small, and expose their seams.", Braces],
  ["02", "Containers & clusters", "Kubernetes primitives with the operational detail filled in.", Container],
  ["03", "Delivery systems", "Pipelines, promotion, policy, and a rollback you can trust.", Radio],
  ["04", "Observability", "Signals that explain what happened, not just that something did.", Network],
];

function Reveal({ children, delay = 0, className = "" }: { children: ReactNode; delay?: number; className?: string }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const node = document.getElementById(`reveal-${delay}-${className.replace(/\W/g, "")}`);
    if (!node) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setVisible(true);
        observer.disconnect();
      }
    }, { threshold: 0.08 });
    observer.observe(node);
    return () => observer.disconnect();
  }, [delay, className]);
  return <div id={`reveal-${delay}-${className.replace(/\W/g, "")}`} className={`reveal ${visible ? "is-visible" : ""} ${className}`} style={{ transitionDelay: `${delay}ms` }}>{children}</div>;
}

function Pill({ children, accent = false }: { children: ReactNode; accent?: boolean }) {
  return <span className={`pill ${accent ? "pill-accent" : ""}`}>{children}</span>;
}

export function Portfolio() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [sent, setSent] = useState(false);

  const nav = ["work", "approach", "now"];
  return (
    <main className="portfolio-shell">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        .portfolio-shell { --ink:#e6f0ec; --muted:#8ca39d; --line:#213932; --panel:#10231f; --deep:#091511; --acid:#c8f36a; --cyan:#83e4dd; --amber:#ffc878; background:var(--deep); color:var(--ink); font-family:'Manrope',sans-serif; overflow:hidden; }
        .portfolio-shell * { box-sizing:border-box; } .portfolio-shell a { color:inherit; text-decoration:none; }
        .mono { font-family:'DM Mono',monospace; } .container { width:min(1160px,calc(100% - 48px)); margin:auto; }
        .topbar { position:sticky; top:0; z-index:20; border-bottom:1px solid rgba(56,94,80,.5); background:rgba(9,21,17,.86); backdrop-filter:blur(16px); }
        .topbar-inner { height:76px; display:flex; align-items:center; justify-content:space-between; } .brand { display:flex; align-items:center; gap:12px; font-weight:800; letter-spacing:-.04em; }
        .brand-mark { color:var(--acid); border:1px solid var(--acid); width:28px; height:28px; display:grid; place-items:center; font-size:13px; border-radius:50%; }
        .nav { display:flex; gap:30px; align-items:center; color:var(--muted); font-size:13px; } .nav a:hover { color:var(--acid); }
        .status { display:flex; align-items:center; gap:8px; color:var(--acid); font-size:11px; letter-spacing:.08em; text-transform:uppercase; } .status-dot { width:7px; height:7px; background:var(--acid); border-radius:50%; box-shadow:0 0 0 4px rgba(200,243,106,.12); }
        .menu-btn { display:none; background:none; border:0; color:var(--ink); }
        .hero { min-height:690px; display:grid; grid-template-columns:1.12fr .88fr; align-items:center; gap:70px; padding:96px 0 90px; position:relative; }
        .hero:before { content:''; position:absolute; width:540px; height:540px; right:-170px; top:30px; border-radius:50%; background:radial-gradient(circle,rgba(89,176,147,.13),transparent 66%); pointer-events:none; }
        .eyebrow { display:flex; align-items:center; gap:10px; color:var(--acid); font-size:11px; letter-spacing:.12em; text-transform:uppercase; margin-bottom:25px; } .eyebrow:before { content:''; height:1px; width:30px; background:var(--acid); }
        h1 { font-size:clamp(3.2rem,7vw,6.9rem); line-height:.94; letter-spacing:-.085em; margin:0; max-width:760px; font-weight:700; } h1 em { color:var(--acid); font-style:normal; }
        .hero-copy { color:#a7bab4; font-size:17px; line-height:1.75; max-width:570px; margin:31px 0 32px; } .hero-copy strong { color:var(--ink); font-weight:600; }
        .actions { display:flex; align-items:center; gap:22px; flex-wrap:wrap; } .button { display:inline-flex; align-items:center; gap:10px; padding:14px 18px; font-size:13px; font-weight:700; border:1px solid var(--acid); color:var(--deep); background:var(--acid); transition:.25s ease; } .button:hover { transform:translateY(-3px); box-shadow:0 12px 30px rgba(200,243,106,.13); } .text-link { font-size:13px; color:var(--muted); border-bottom:1px solid var(--line); padding-bottom:5px; } .text-link:hover { color:var(--acid); border-color:var(--acid); }
        .terminal { background:#0d1c19; border:1px solid #2b5145; box-shadow:20px 24px 0 rgba(19,43,36,.45); position:relative; transform:rotate(1.5deg); } .terminal-head { border-bottom:1px solid #29473e; padding:14px 16px; display:flex; gap:7px; align-items:center; } .term-dot { width:8px; height:8px; border-radius:50%; background:#38584e; } .term-title { margin-left:auto; color:#55756a; font-size:10px; }
        .terminal-body { padding:25px 24px 28px; min-height:340px; font-size:12px; line-height:2; color:#99b7ac; } .prompt { color:var(--acid); } .command { color:#d9e7df; } .terminal-line { display:block; } .terminal-result { color:#6e9787; padding-left:18px; } .cursor { display:inline-block; width:8px; height:15px; background:var(--acid); vertical-align:-2px; animation:blink 1.1s steps(2) infinite; }
        @keyframes blink { 50% { opacity:0; } } .scroll-cue { position:absolute; bottom:25px; left:0; display:flex; gap:12px; align-items:center; color:#56756a; font-size:10px; letter-spacing:.1em; text-transform:uppercase; } .scroll-cue span { width:30px; height:1px; background:#56756a; }
        .section { padding:116px 0; border-top:1px solid var(--line); } .section-label { color:var(--acid); font-size:11px; letter-spacing:.13em; text-transform:uppercase; margin-bottom:22px; } .section-title { font-size:clamp(2.1rem,4vw,4rem); letter-spacing:-.07em; line-height:1; margin:0; max-width:690px; } .section-intro { color:var(--muted); max-width:510px; line-height:1.75; margin-top:22px; }
        .work-head { display:flex; justify-content:space-between; gap:30px; align-items:end; margin-bottom:60px; } .work-list { border-top:1px solid var(--line); } .project { display:grid; grid-template-columns:70px 1fr 260px; gap:34px; align-items:start; padding:38px 0; border-bottom:1px solid var(--line); transition:.3s; } .project:hover { padding-left:12px; background:linear-gradient(90deg,rgba(200,243,106,.035),transparent 55%); } .project-no { color:#4e7064; font-size:12px; padding-top:5px; } .project-sub { color:var(--acid); text-transform:uppercase; letter-spacing:.08em; font-size:10px; margin-bottom:12px; } .project h3 { font-size:clamp(1.45rem,2.7vw,2.25rem); letter-spacing:-.055em; margin:0 0 14px; } .project p { color:var(--muted); line-height:1.7; max-width:600px; margin:0; font-size:14px; } .project-side { border-left:1px solid var(--line); padding-left:24px; } .metric { color:var(--ink); font-size:13px; margin-bottom:9px; } .metric:before { content:'↳'; color:var(--acid); margin-right:8px; } .tags { display:flex; flex-wrap:wrap; gap:7px; margin-top:20px; } .pill { border:1px solid #315247; padding:6px 9px; font:10px 'DM Mono',monospace; color:#92aaa1; } .pill-accent { color:var(--acid); border-color:#5e783e; }
        .split { display:grid; grid-template-columns:.9fr 1.1fr; gap:100px; } .stack-list { border-top:1px solid var(--line); } .stack-item { display:grid; grid-template-columns:45px 1fr 30px; gap:20px; padding:23px 0; border-bottom:1px solid var(--line); align-items:center; } .stack-item:hover .stack-arrow { color:var(--acid); transform:translateX(4px); } .stack-no { color:#547369; font-size:11px; } .stack-item h3 { font-size:16px; margin:0 0 6px; letter-spacing:-.03em; } .stack-item p { color:var(--muted); font-size:12px; line-height:1.5; margin:0; } .stack-arrow { color:#4d6b61; transition:.2s; }
        .principles { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin-top:65px; } .principle { background:var(--panel); padding:28px; border:1px solid #1d3a31; min-height:185px; } .principle:nth-child(2) { margin-top:26px; } .principle:nth-child(3) { margin-top:52px; } .principle-icon { color:var(--cyan); margin-bottom:25px; } .principle h3 { margin:0 0 10px; font-size:17px; } .principle p { margin:0; color:var(--muted); line-height:1.65; font-size:13px; }
        .now { background:#0e211c; position:relative; overflow:hidden; } .now:after { content:'OPEN'; position:absolute; right:-15px; bottom:-55px; font-size:190px; line-height:1; font-weight:800; color:rgba(200,243,106,.035); letter-spacing:-.12em; } .now-grid { display:grid; grid-template-columns:1.1fr .9fr; gap:90px; align-items:end; } .availability { display:flex; gap:13px; align-items:center; color:var(--acid); font-size:12px; text-transform:uppercase; letter-spacing:.08em; margin-top:31px; } .availability i { width:8px; height:8px; background:var(--acid); border-radius:50%; }
        .contact { display:flex; justify-content:space-between; align-items:center; gap:30px; border-top:1px solid #315447; padding-top:25px; position:relative; z-index:1; } .contact-email { font-size:clamp(1rem,2vw,1.35rem); } .contact-email:hover { color:var(--acid); } .contact-note { color:var(--muted); font-size:12px; max-width:225px; line-height:1.6; }
        footer { padding:28px 0; color:#5f7b71; font:10px 'DM Mono',monospace; display:flex; justify-content:space-between; } .footer-links { display:flex; gap:20px; } .footer-links a:hover { color:var(--acid); }
        .reveal { opacity:0; transform:translateY(24px); transition:opacity .7s ease,transform .7s ease; } .reveal.is-visible { opacity:1; transform:none; }
        @media (max-width:760px) { .container { width:min(100% - 34px,600px); } .topbar-inner { height:66px; } .nav,.topbar .status { display:none; } .menu-btn { display:block; } .mobile-nav { display:flex; position:absolute; top:65px; left:0; right:0; padding:18px; background:#0d1c19; border-bottom:1px solid var(--line); flex-direction:column; align-items:flex-start; gap:18px; color:var(--muted); font-size:13px; } .hero { display:block; padding:80px 0 88px; min-height:auto; } h1 { font-size:clamp(3.2rem,16vw,5rem); } .hero-copy { font-size:15px; } .terminal { margin-top:58px; transform:none; } .scroll-cue { display:none; } .section { padding:78px 0; } .work-head,.split,.now-grid { display:block; } .work-head { margin-bottom:38px; } .project { grid-template-columns:34px 1fr; gap:12px; padding:28px 0; } .project-side { grid-column:2; border-left:0; padding:15px 0 0; } .principles { display:block; margin-top:42px; } .principle,.principle:nth-child(2),.principle:nth-child(3) { margin:0 0 12px; min-height:auto; } .stack-list { margin-top:50px; } .now-grid .contact { margin-top:48px; display:block; } .contact-note { margin-top:20px; } footer { display:block; line-height:2; } .footer-links { margin-top:10px; } }
      `}</style>

      <header className="topbar">
        <div className="container topbar-inner">
          <a href="#top" className="brand"><span className="brand-mark">AM</span><span>Alex Morgan<span style={{ color: "var(--acid)" }}>.</span></span></a>
          <nav className="nav">{nav.map((item) => <a key={item} href={`#${item}`}>{item}</a>)}<a href="#contact">contact</a></nav>
          <div className="status"><span className="status-dot" /> open to the right problem</div>
          <button className="menu-btn" aria-label="Toggle navigation" onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X size={21} /> : <Menu size={21} />}</button>
          {menuOpen && <nav className="mobile-nav">{[...nav, "contact"].map((item) => <a onClick={() => setMenuOpen(false)} key={item} href={`#${item}`}>{item}</a>)}</nav>}
        </div>
      </header>

      <section className="container hero" id="top">
        <Reveal><div><div className="eyebrow mono">platform engineer / available now</div><h1>Code that survives<br /><em>production.</em></h1><p className="hero-copy">I’m <strong>Alex Morgan</strong> — a hands-on engineer turning Java services into reliable, cloud-native systems. From a Quarkus commit to a healthy pod on OpenShift, I own the path in between.</p><div className="actions"><a className="button" href="#work">See selected work <ArrowDownRight size={16} /></a><a className="text-link" href="#contact">Start a conversation <ArrowUpRight size={14} /></a></div></div></Reveal>
        <Reveal delay={160}><div className="terminal"><div className="terminal-head"><i className="term-dot" /><i className="term-dot" /><i className="term-dot" /><span className="term-title mono">alex@platform ~/signal</span></div><div className="terminal-body mono"><span className="terminal-line"><span className="prompt">$</span> <span className="command">whoami</span></span><span className="terminal-line terminal-result">alex.morgan — platform engineer</span><br /><span className="terminal-line"><span className="prompt">$</span> <span className="command">cat operating_principles.md</span></span><span className="terminal-line terminal-result">01 / reduce the distance to production</span><span className="terminal-line terminal-result">02 / make failure useful</span><span className="terminal-line terminal-result">03 / leave the system clearer</span><br /><span className="terminal-line"><span className="prompt">$</span> <span className="command">kubectl get signal</span></span><span className="terminal-line terminal-result" style={{ color: "var(--acid)" }}>platform   online   99.96%   <Check size={12} style={{ verticalAlign: "-2px" }} /></span><br /><span className="terminal-line"><span className="prompt">$</span> <span className="cursor" /></span></div></div></Reveal>
        <div className="scroll-cue mono"><span /> scroll to inspect the system</div>
      </section>

      <section className="section" id="work"><div className="container"><Reveal><div className="work-head"><div><div className="section-label mono">01 / selected systems</div><h2 className="section-title">The work behind<br />the uptime.</h2></div><p className="section-intro">Not a museum of logos. A field guide to the moments where good engineering made the system—and the team—calmer.</p></div></Reveal><div className="work-list">{projects.map((project, i) => <Reveal delay={i * 100} className={`project-r${i}`} key={project.number}><article className="project"><div className="project-no mono">{project.number}</div><div><div className="project-sub mono">{project.subtitle}</div><h3>{project.title}</h3><p>{project.copy}</p><div className="tags">{project.tags.map((tag) => <Pill key={tag} accent>{tag}</Pill>)}</div></div><div className="project-side"><div className="metric mono">{project.metrics[0]}</div><div className="metric mono">{project.metrics[1]}</div><a href="#contact" className="text-link" style={{ display: "inline-block", marginTop: 18 }}>Read the brief <ChevronRight size={12} style={{ verticalAlign: "-2px" }} /></a></div></article></Reveal>)}</div></div></section>

      <section className="section" id="approach"><div className="container"><div className="split"><Reveal><div><div className="section-label mono">02 / the capability story</div><h2 className="section-title">A stack is only useful when it tells a story.</h2><p className="section-intro">I work across the seam between application code and platform reality. That means knowing what a service needs, what a cluster can promise, and where the two will disagree at 03:17.</p></div></Reveal><Reveal delay={120}><div className="stack-list">{stack.map(([no, title, copy, Icon]) => <div className="stack-item" key={no}><span className="stack-no mono">{no}</span><div><h3>{title}</h3><p>{copy}</p></div><Icon className="stack-arrow" size={18} /></div>)}</div></Reveal></div><div className="principles"><Reveal delay={80}><div className="principle"><Cloud className="principle-icon" size={22} /><h3>Platform as product</h3><p>Clear paths, useful defaults, and documentation that respects the person on call.</p></div></Reveal><Reveal delay={160}><div className="principle"><ShieldCheck className="principle-icon" size={22} /><h3>Reliability is a feature</h3><p>Health checks, graceful failure, and delivery signals are part of the design—not cleanup.</p></div></Reveal><Reveal delay={240}><div className="principle"><Cpu className="principle-icon" size={22} /><h3>Curious, then practical</h3><p>I like new tools. I like them more when they make tomorrow’s incident smaller.</p></div></Reveal></div></div></section>

      <section className="section now" id="now"><div className="container now-grid"><Reveal><div><div className="section-label mono">03 / current signal</div><h2 className="section-title">Looking for a team that ships thoughtfully.</h2><p className="section-intro">I’m open to platform engineering and senior backend roles where I can work close to the code, the cluster, and the people operating both.</p><div className="availability"><i /> available for conversations · remote / hybrid</div></div></Reveal><Reveal delay={130}><div className="contact" id="contact"><div><div className="mono section-label" style={{ marginBottom: 12 }}>route open</div><a className="contact-email" href="mailto:alex.morgan@signal.dev">alex.morgan@signal.dev <ExternalLink size={17} style={{ verticalAlign: "-2px", color: "var(--acid)" }} /></a></div><p className="contact-note">Have a hard platform problem or a team building its first one? Tell me what’s breaking.</p></div></Reveal></div></section>
      <div className="container"><footer><span>© 2024 Alex Morgan · portfolio mockup</span><div className="footer-links"><a href="#top"><Terminal size={12} style={{ verticalAlign: "-2px" }} /> back to top</a><a href="https://github.com" target="_blank" rel="noreferrer"><Github size={12} style={{ verticalAlign: "-2px" }} /> github</a><a href="mailto:alex.morgan@signal.dev" onClick={() => setSent(true)}><Mail size={12} style={{ verticalAlign: "-2px" }} /> {sent ? "mail client opened" : "email"}</a><span><MapPin size={12} style={{ verticalAlign: "-2px" }} /> UTC−05</span><Globe2 size={12} /></div></footer></div>
    </main>
  );
}