# Portfolio Improvement Roadmap
## Goal: Position this portfolio for big-company Backend / Platform Engineering roles

Source reviewed: https://fsmutimeer.github.io/
Review date: 2026-08-31

---

## 0. North-star positioning

### Current positioning

The site currently leads with:

> Code that survives production.

and:

> I’m Feroz Shah — a software engineer at IT22 B.V. turning Java services into systems that survive the cluster.

This is distinctive, but it makes the visitor infer your role and seniority.

### Target positioning

Position yourself primarily as:

**Backend / Platform Engineer**

with a technical specialization in:

**Java · Distributed Systems · Cloud Native · Quarkus · Kafka · Kubernetes/OpenShift**

Quarkus should be an important technology, not the entire identity.

### Suggested positioning statement

> **Backend & Platform Engineer**
>
> I build production-grade distributed systems with Java, Quarkus, Kafka and Kubernetes/OpenShift — from service design and event-driven architecture to GitOps delivery, security and observability.

Keep **“Code that survives production.”** as a secondary tagline if you like it.

---

# 1. Priority order

Work through the improvements in this order.

## 🔴 P0 — Do these first

- [ ] Rewrite the hero section around Backend / Platform Engineering
- [ ] Make your target role obvious
- [ ] Add measurable impact to professional work
- [ ] Turn the strongest projects into detailed case studies
- [ ] Make quarkus-doctor a flagship project
- [ ] Add a clear experience timeline
- [ ] Add prominent CV / GitHub / LinkedIn links
- [ ] Make the site immediately understandable in 10–15 seconds
- [ ] Keep hero type readable over the 3D scene (one canvas; no second WebGL context)
- [ ] Align page metadata, Open Graph, and JSON-LD with the new role
- [ ] Fix the known SplitTitle hydration overlay before new pages go live

## 🟠 P1 — Do next

- [ ] Add architecture diagrams
- [ ] Explain engineering decisions
- [ ] Organize technologies by capability
- [ ] Add technical writing
- [ ] Add sanitized details for proprietary production work
- [ ] Add stronger project outcomes
- [ ] Improve project navigation

## 🟢 P2 — Polish

- [ ] Visual refinements
- [ ] Better animations/transitions
- [ ] More micro-interactions
- [ ] SEO metadata
- [ ] Open Graph/social preview
- [ ] Accessibility audit
- [ ] Performance audit
- [ ] Mobile review

Do not spend most of your time on P2 until P0/P1 are finished.

---

# 2. Hero section

## Problem

The current hero is memorable but abstract.

Current:

> Code that survives production.

The visitor should not have to figure out whether you are a Java developer, DevOps engineer, platform engineer, or architect.

## Target

The first screen should answer:

1. Who are you?
2. What do you do?
3. What technologies/systems do you specialize in?
4. What kind of opportunity are you looking for?
5. Where can I see your work?

## Suggested structure

### Eyebrow

> BACKEND / PLATFORM ENGINEER

### Heading

> I build backend systems that survive production.

### Description

> Software engineer at IT22 B.V. specializing in Java, Quarkus, Kafka and Kubernetes/OpenShift. I work across service architecture, event-driven systems, GitOps delivery, security and cloud-native platforms.

### Technology line

> Java · Quarkus · Kafka · Kubernetes · OpenShift · GitOps

### CTAs

- View selected work
- Download CV
- GitHub
- LinkedIn

### Optional availability line

If true:

> Open to Backend / Platform Engineering opportunities.

If you are not currently looking, do not add this.

---

# 3. About section

## Current strength

The current About section already communicates an unusually broad backend + platform profile.

Current message:

> Engineer at the seam of services and platforms.

Keep the idea, but make the actual professional value clearer.

## Suggested structure

### Heading

> Backend engineer with platform ownership.

### Copy

> I’m a software engineer at IT22 B.V. working across backend services and the platforms that run them.
>
> My core work is Java and Quarkus microservices, with Apache Camel, Kafka, MongoDB and Keycloak. I also work on OpenShift/OKD and Kubernetes infrastructure, GitOps delivery with Argo CD and Helm, CI/CD with Tekton, and security tooling such as Trivy and Wazuh.
>
> I’m most interested in the boundary between application engineering and platform engineering: designing services that are easy to operate, making delivery repeatable, and reducing the number of production problems that require manual intervention.

Keep your education below this rather than mixing it into the main paragraph.

---

# 4. Experience timeline

## Problem

Your current site communicates your current role and previous company, but a recruiter would benefit from a dedicated timeline.

## Add

### IT22 B.V.
**Software Engineer / Backend & Platform Engineer**  
Islamabad, Pakistan · [START YEAR] – Present

Add 4–6 achievement bullets.

Example structure:

- Led backend development for [NUMBER] Quarkus microservices supporting [SYSTEM/DOMAIN].
- Designed Kafka-based event flows between [SERVICES/DOMAINS].
- Implemented Keycloak-based RBAC across [NUMBER] modules/services.
- Built or maintained OpenShift/OKD infrastructure with [DETAIL].
- Automated delivery through Tekton, Argo CD and Helm.
- Introduced [TOOL/PRACTICE], reducing [PROBLEM] by [METRIC].

Only use numbers and claims you can verify.

### ESOLS Technologies
**Software Engineer**  
[LOCATION] · [DATES]

Add 2–4 bullets focused on outcomes, not duties.

---

# 5. Numbers / measurable impact

## This is one of the most important improvements.

The portfolio currently contains technically impressive work but relatively little measurable impact.

Go through every project and professional achievement and find real numbers.

Look for:

- Number of microservices
- Number of Kafka topics
- Events/messages per day
- Requests per second
- Number of users
- Database size
- Number of environments
- Number of clusters
- Number of nodes
- Deployment frequency
- CI pipeline duration
- Build time
- Deployment time
- Startup time
- Memory usage
- Container image size
- Test coverage
- Vulnerabilities caught
- Incidents reduced
- Manual steps removed
- Engineering hours saved
- Number of developers supported

## Example

Weak:

> Built Kafka event paths for ERP/PMS microservices.

Stronger:

> Designed Kafka event flows across X backend services, removing synchronous dependencies between [services].

Strongest, if true:

> Designed Kafka event flows across X Quarkus services processing approximately Y events/day, reducing synchronous dependencies and improving failure isolation.

Never invent metrics.

If you cannot quantify something, describe the concrete engineering outcome instead.

---

# 6. Selected work

Your selected-work section is already a strong part of the site.

Keep the four main projects:

1. Quarkus / Camel / Kafka services
2. OpenShift / OKD
3. Tekton / Argo CD / Helm / DevSecOps
4. quarkus-doctor

But turn each into a real case study.

Each case study should answer:

- What problem existed?
- What was your responsibility?
- What architecture did you use?
- Why did you choose it?
- What did you implement?
- What went wrong?
- How did you solve it?
- What was the result?
- What would you change today?

---

# 7. Make quarkus-doctor the flagship project

This is one of the strongest differentiators in the portfolio.

Current description:

> I built quarkus-doctor: a Java CLI and Maven plugin that compares application.properties with Kubernetes, Helm, and Kustomize manifests in CI—before the image ships.

That is already good.

Give it much more prominence.

## Recommended case-study structure

### quarkus-doctor

**Static validation for Quarkus configuration before deployment**

#### Problem

Explain the production failure mode.

Example:

> Configuration can be correct inside the application while being inconsistent with Kubernetes, Helm or Kustomize deployment configuration. These mismatches are often discovered only after deployment.

#### Solution

> quarkus-doctor validates the relationship between application configuration and deployment manifests before an image is deployed.

#### Architecture

Show:

```text
Quarkus application
        |
        +---- application.properties
        |
        v
   quarkus-doctor
        |
   +----+----+----------------+
   |         |                |
Kubernetes  Helm          Kustomize
manifests   values        overlays
   |         |                |
   +---------+----------------+
             |
             v
        validation result
             |
             v
        CI / Maven build
```

#### Technical details

Explain:

- Java
- Maven plugin
- CLI
- Maven verify goal
- Configuration parsing
- Kubernetes manifests
- Helm
- Kustomize
- Validation rules
- CI integration
- No-cluster-required design

#### Engineering decisions

Explain why you chose:

- Static validation
- Maven integration
- CI-time validation
- No live cluster dependency

#### Evidence

Link to:

- GitHub
- Documentation
- Releases
- Tests
- Example output

This should become a project you can discuss for 20–30 minutes in an interview.

---

# 8. OpenShift / OKD case study

Current claim:

> Deployed a Red Hat OpenShift cluster via Assisted Installer and an OKD cluster on KVM—each with three control-plane nodes and one worker—on premises.

This is valuable.

Expand it.

## Show

### Architecture

```text
                 Git / Config
                     |
                     v
             Cluster configuration
                     |
        +------------+-------------+
        |                          |
        v                          v
   OpenShift                     OKD
   Assisted                      KVM
   Installer
        |                          |
        v                          v
  3 control-plane              3 control-plane
  + 1 worker                   + 1 worker
```

Adjust this to your real architecture.

## Explain

- Why on-prem?
- Why OpenShift?
- Why OKD?
- Why KVM?
- How networking was handled
- Storage approach
- DNS
- Ingress
- Registry
- Authentication
- Cluster upgrades
- Monitoring
- Troubleshooting
- What failed during installation
- What you learned

The failures and lessons are often more interesting than the happy path.

---

# 9. GitOps / DevSecOps case study

Current project:

> The path from commit to a scanned image

This is strong.

Make the pipeline visual.

## Suggested architecture

```text
Developer
   |
   v
Git repository
   |
   v
Tekton
   |
   +--> Build
   |
   +--> Test
   |
   +--> Trivy scan
   |
   v
Container registry
   |
   v
Argo CD
   |
   v
Helm
   |
   v
OpenShift / OKD
   |
   v
Runtime monitoring
```

Add the actual tools you use.

Explain:

- What triggers the pipeline
- How images are built
- Where tests run
- How Trivy gates delivery
- How environments differ
- How Helm values are managed
- How Argo CD App-of-Apps works
- How rollback works
- How secrets are handled
- What happens when a scan fails
- What happens when deployment fails

---

# 10. Backend microservices case study

Current:

> Services that talk without a shared blast radius

Good concept. Make it concrete.

## Show

```text
              API / Client
                   |
                   v
             Quarkus service
                   |
          +--------+--------+
          |                 |
          v                 v
        Kafka            MongoDB
          |
    +-----+------+
    |            |
    v            v
 Service A    Service B

          Keycloak
              |
              v
             RBAC
```

Adapt this to your real architecture.

## Explain

- Service boundaries
- REST APIs
- Kafka topics
- Event contracts
- Retry behavior
- Idempotency
- Failure handling
- Database ownership
- RBAC
- Authentication
- Observability
- Testing

---

# 11. Technology section

Your current technology list is broad.

Keep it, but organize it by capability.

## Backend

Java · Quarkus · Apache Camel

## Messaging

Kafka

## Data

MongoDB

Only list technologies you genuinely use. PostgreSQL and Redis are not on the live profile stack in this repo — do not add them unless they appear on the CV.

## Platform

Kubernetes · OpenShift · OKD · Docker

## Delivery

Tekton · Argo CD · Helm · GitOps

## Security

Keycloak · Trivy · Wazuh

This communicates your engineering model better than one long technology list.

---

# 12. Add engineering decisions

This can make the site feel much more senior.

For every major case study, add:

## Why Kafka?

Explain the actual requirement.

## Why Quarkus?

Explain startup time, footprint, development model, ecosystem, or other real reasons.

## Why MongoDB?

Explain the data model and access patterns.

## Why GitOps?

Explain reproducibility, auditability and rollback.

## Why OpenShift?

Explain the organizational/infrastructure requirement.

## What would I change?

This is particularly important.

Show that you can critically evaluate your own architecture.

Example:

> If I rebuilt this today, I would reconsider X because Y. The original decision made sense because Z.

---

# 13. Add technical writing

Create a `Writing` section.

Start with 3–5 high-quality articles rather than 20 shallow posts.

Potential topics:

1. Building a Maven plugin for Quarkus configuration validation
2. Detecting configuration drift before Kubernetes deployment
3. Running OKD on KVM: lessons learned
4. Building a GitOps delivery path with Tekton and Argo CD
5. Quarkus + Kafka: designing failure-tolerant event flows
6. Adding Trivy security gates to CI/CD
7. Keycloak RBAC patterns for microservices
8. What production taught me about cloud-native Java

Each article should contain:

- Problem
- Context
- Architecture
- Implementation
- Failure/edge cases
- Lessons learned
- Code/examples where possible

---

# 14. GitHub

Make GitHub much more visible.

For every public project:

- GitHub
- Documentation
- Architecture
- Demo/example
- Release/version

For private company projects:

> Production system — source code private

Then provide:

- Sanitized architecture
- Your contribution
- Technologies
- Engineering decisions
- Outcomes

Never expose proprietary code, credentials, internal URLs, customer information or confidential architecture.

---

# 15. CV

The CV should use the same positioning as the website.

## Header

> Feroz Shah  
> **Backend / Platform Engineer**

## Summary

> Backend and platform engineer specializing in Java, Quarkus, Kafka and Kubernetes/OpenShift. Experienced in distributed services, event-driven architecture, GitOps delivery, DevSecOps and production infrastructure.

Keep the summary short.

## Skills

Group by capability rather than one giant list.

## Experience

Use achievement bullets.

## Projects

Feature:

**quarkus-doctor**

as a serious engineering project.

## Education

Keep your M.Sc. information.

---

# 16. LinkedIn alignment

Your website, CV and LinkedIn should tell the same story.

Recommended LinkedIn headline:

> Backend / Platform Engineer | Java | Quarkus | Kafka | Kubernetes | OpenShift | Distributed Systems

Possible About opening:

> I’m a backend and platform engineer focused on Java, Quarkus, distributed systems and cloud-native infrastructure...

Your portfolio should link to LinkedIn, and LinkedIn should link back to the portfolio.

---

# 17. Recruiter 15-second test

After the changes, a recruiter should be able to answer these questions immediately:

### Who is this?

Backend / Platform Engineer.

### What does he specialize in?

Java, Quarkus, Kafka, Kubernetes/OpenShift.

### Does he build production systems?

Yes.

### Does he understand infrastructure?

Yes.

### Does he understand distributed systems?

Yes.

### Does he have evidence?

Case studies + architecture + GitHub + measurable outcomes.

### Can I contact him?

Yes.

If any answer requires scrolling through several sections, simplify the page.

---

# 18. Big-company positioning

Do not market yourself as only:

> Quarkus Developer

Market yourself as:

> Backend / Platform Engineer

Then use:

> Java · Distributed Systems · Cloud Native

as your broader technical identity.

Quarkus becomes a strong implementation detail.

This protects you from companies that do not use Quarkus while still making your Quarkus expertise valuable.

---

# 19. What NOT to do

## Don't

- Add 30 more technologies just to make the stack look bigger
- Claim Kubernetes expertise without explaining what you actually operated
- Add fake metrics
- Publish confidential company architecture
- Turn every project into a generic CRUD demo
- Make the site visually flashy at the expense of clarity
- Put every technology in giant logo grids
- Call yourself an architect if your experience doesn't support it
- Make Quarkus your only identity
- Spend weeks polishing animations before fixing content
- Treat the 3D dragon, fire, custom cursor, or a second WebGL canvas as P0 work
- Add PostgreSQL, Redis, or any other tool that is not on the CV

## Do

- Show difficult problems
- Explain tradeoffs
- Show architecture
- Quantify outcomes
- Show real code where possible
- Explain failures
- Explain what you learned
- Show ownership
- Keep the site fast and readable

---

# 20. Suggested final site structure

```text
HOME
│
├── Hero
│   ├── Backend / Platform Engineer
│   ├── Short positioning statement
│   ├── Core technologies
│   └── CV / GitHub / LinkedIn / Work
│
├── ABOUT
│   ├── Professional summary
│   ├── What I specialize in
│   └── Education
│
├── EXPERIENCE
│   ├── IT22 B.V.
│   └── ESOLS Technologies
│
├── SELECTED WORK
│   ├── quarkus-doctor
│   ├── Backend / Kafka systems
│   ├── OpenShift / OKD
│   └── GitOps / DevSecOps
│
├── ENGINEERING
│   ├── Architecture
│   ├── Engineering decisions
│   ├── Reliability
│   └── Security
│
├── WRITING
│   ├── Article 1
│   ├── Article 2
│   └── Article 3
│
├── NOW
│   └── Current focus
│
└── CONTACT
    ├── Email
    ├── GitHub
    ├── LinkedIn
    └── CV
```

---

# 21. Suggested implementation milestones

## Milestone 1 — Positioning

- [ ] Hero rewrite
- [ ] About rewrite
- [ ] Target-role language
- [ ] CTA cleanup
- [ ] CV/GitHub/LinkedIn links
- [ ] Metadata / Open Graph / JSON-LD jobTitle
- [ ] Hero type vs 3D scene (scrim / containment)

## Milestone 2 — Evidence

- [ ] Add experience timeline
- [ ] Collect real metrics
- [ ] Rewrite experience bullets
- [ ] Rewrite project descriptions

## Milestone 3 — Case studies

- [ ] quarkus-doctor case study
- [ ] Kafka/microservices case study
- [ ] OpenShift/OKD case study
- [ ] GitOps/DevSecOps case study
- [ ] Add architecture diagrams

## Milestone 4 — Senior-level signal

- [ ] Engineering decisions
- [ ] Tradeoffs
- [ ] Failure modes
- [ ] Lessons learned
- [ ] What I would change

## Milestone 5 — Public engineering presence

- [ ] GitHub cleanup
- [ ] README improvements
- [ ] 3–5 technical articles
- [ ] LinkedIn alignment
- [ ] CV alignment

## Milestone 6 — Polish

- [ ] Mobile
- [ ] Accessibility
- [ ] Performance
- [ ] SEO
- [ ] Social preview
- [ ] Visual refinement

---

# 22. Definition of done

Consider the portfolio ready when:

- [ ] A recruiter understands your role in <15 seconds
- [ ] Your first screen says Backend / Platform Engineer
- [ ] Java + distributed systems + cloud-native are obvious
- [ ] Your current experience is easy to scan
- [ ] You have at least 3 detailed technical case studies
- [ ] quarkus-doctor is a flagship project
- [ ] Home selected-work cards link to dedicated static case-study pages
- [ ] Page metadata matches the on-page role
- [ ] At least one case study contains an architecture diagram
- [ ] At least one case study contains measurable impact
- [ ] Engineering tradeoffs are documented
- [ ] GitHub is easy to find
- [ ] CV is easy to find
- [ ] LinkedIn is easy to find
- [ ] Proprietary information is protected
- [ ] The site works well on mobile
- [ ] The site loads quickly
- [ ] Your CV, LinkedIn and website use the same positioning

---

# 23. The career message behind the portfolio

The portfolio should move the perception from:

> “Java/Quarkus developer who knows some DevOps.”

to:

> **“Backend/platform engineer who understands the entire production path: service design → messaging → security → CI/CD → containers → Kubernetes/OpenShift → GitOps → operations.”**

That second profile is much more compelling for large engineering organizations.

---

# 24. Your strongest differentiators

Based on the current portfolio, lean into these:

1. **Java + Quarkus**
2. **Event-driven systems with Kafka**
3. **Kubernetes/OpenShift/OKD**
4. **GitOps with Argo CD**
5. **Tekton CI/CD**
6. **DevSecOps**
7. **Keycloak/security**
8. **Building developer tooling**
9. **quarkus-doctor**
10. **Application + platform ownership**

You do not need another giant list of technologies.

You need stronger evidence around the technologies you already use.

---

# 25. Final principle

Your portfolio should not say:

> “Look how many technologies I know.”

It should say:

> **“Here are hard production problems I've solved, here is how I designed the system, here is why I made those decisions, here is what happened, and here is what I learned.”**

That is the story to optimize for.

---

# 26. Site-specific notes (this repo)

These notes bind the roadmap to [fsmutimeer.github.io](https://fsmutimeer.github.io/) as it exists in this codebase. They do not change the first three things to work on.

## Content source of truth

All public copy lives in `src/lib/profile.ts` and `src/lib/content.ts`. Edit the PDF CV and LinkedIn in the same pass as the hero. Do not invent dates, counts, or tools.

## Hero vs 3D scene (P0, not P2)

The first screen must still be readable with the canvas on. The recruiter test fails if the coiled dragon covers “Backend / Platform Engineer.” Treat type hierarchy, scrim, and scene containment as part of Milestone 1, not later polish. Do not start a second WebGL context.

## Case studies need routes

A 20–30 minute quarkus-doctor write-up will not fit a pinned card. Add static pages such as `/work/quarkus-doctor` (and the other three). Keep home “Selected work” as a scan layer that links in. This repo is a Next static export, so MDX/pages must stay static-export friendly. GitHub and docs links for quarkus-doctor already exist in `src/lib/content.ts`.

## Machine-readable identity

`src/app/layout.tsx` still has `jobTitle: Software Engineer` in metadata, Open Graph, and JSON-LD. Change those when the hero changes, or Google/LinkedIn previews will disagree with the page.

## Private metrics worksheet

Keep candidate numbers (services, nodes, pipeline times) in a gitignored or uncommitted notes file. Nothing from it goes on the site until verified. This unblocks section 5 without publishing fake metrics.

## ESOLS is supporting evidence, not the lead

Node.js at ESOLS stays on the timeline. It should not compete with the Java / platform story on the first screen.

## Optional logistics line (only if true)

Timezone is already PKT. Add remote / hybrid / relocation / work-authorization only if you want recruiters to know. Do not guess.

## P0 engineering hygiene

There is a known Next hydration error from `src/components/split-title.tsx` (`Date` / locale / `typeof window`). Fix that before case-study pages, or the overlay will sit on top of the new copy.

## Writing is P1 / Milestone 5, not a blocker

Three articles after the quarkus-doctor case study exist. Do not delay the hero rewrite for a blog.

## Availability

Keep the rule in section 2: add “Open to Backend / Platform Engineering opportunities” only if that is true. Confirm before writing it into the hero.

---

## First three things to work on

If you want the fastest improvement, do these first:

### 1. Rewrite the hero

**Backend / Platform Engineer**

Java · Quarkus · Kafka · Kubernetes/OpenShift

### 2. Collect metrics

Open a document and write down every real number you can find from your work.

### 3. Build the quarkus-doctor case study

Make this your flagship public engineering artifact.

Once those three are done, the rest of the portfolio becomes much easier to rewrite.
