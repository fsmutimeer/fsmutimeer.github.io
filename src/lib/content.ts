import {
  Braces,
  Cloud,
  Container,
  Cpu,
  GitBranch,
  Network,
  Radio,
  Rocket,
  ShieldCheck,
  type LucideIcon,
} from 'lucide-react';
import { SiApachekafka, SiKubernetes, SiQuarkus, SiRedhatopenshift } from 'react-icons/si';
import type { IconType } from 'react-icons';

export type CaseStudyDecision = {
  title: string;
  copy: string;
};

export type CaseStudy = {
  problem: string;
  design: string;
  path: string;
  constraints: string[];
  decisions: CaseStudyDecision[];
  limits: string[];
  proprietary?: boolean;
};

export type Project = {
  number: string;
  slug: string;
  scope: string;
  title: string;
  subtitle: string;
  copy: string;
  metrics: string[];
  tags: string[];
  detail: string;
  outcome: string;
  role: string;
  href?: string;
  hrefLabel?: string;
  repoUrl?: string;
  caseStudy: CaseStudy;
};

export type ExperienceRole = {
  id: string;
  company: string;
  companyUrl: string;
  role: string;
  location: string;
  dates: string;
  bullets: string[];
};

export const projects: Project[] = [
  {
    number: '01',
    slug: 'quarkus-kafka',
    scope: 'Backend services',
    title: 'Quarkus microservices with Kafka and Keycloak',
    subtitle: 'Quarkus · Camel · Kafka',
    copy: 'Backend architecture for Quarkus microservices connected through Apache Camel and Kafka, with Keycloak for access control.',
    metrics: ['Event-driven services', 'Kafka event path'],
    tags: ['Quarkus', 'Apache Camel', 'Kafka', 'Keycloak', 'Event-Driven'],
    detail:
      'Backend engineering for Quarkus microservices. Apache Camel and Kafka handle event distribution between independent domain services. Keycloak provides centralized role-based access control. Downstream notification, relay, and messaging services consume from the event backbone.',
    outcome: 'Decoupled service communication through event contracts rather than direct HTTP dependencies.',
    role: 'Software engineer · Backend · Jan 2023–present',
    caseStudy: {
      proprietary: true,
      problem:
        'Synchronous inter-service coupling causes cascading failures and tight deployment interdependencies across domain boundaries.',
      design:
        'Quarkus microservices with Apache Camel and Apache Kafka orchestrating asynchronous domain events. Centralized Keycloak RBAC handles authorization across services. Core and downstream consumers operate on a shared event backbone.',
      path:
        'Domain services emit business events to Kafka topics. Downstream services (notification, integration relay, email) independently consume and process events according to explicit contracts.',
      constraints: [
        'Production configuration details, customer data, and specific topic names are omitted.',
      ],
      decisions: [
        {
          title: 'Asynchronous events over direct RPC',
          copy: 'Kafka event streams keep domain communication decoupled, auditable, and replayable.',
        },
        {
          title: 'Camel for integration workflows',
          copy: 'Integration flows exist as declarative contracts, allowing new consumers to subscribe without modifying producers.',
        },
        {
          title: 'Centralized Keycloak RBAC',
          copy: 'Dynamic role-based access control managed centrally rather than reimplemented inside individual services.',
        },
      ],
      limits: [
        'Event-driven architectures introduce operational overhead (broker maintenance, schema evolution) in exchange for decoupling and resilience.',
      ],
    },
  },
  {
    number: '02',
    slug: 'openshift-okd',
    scope: 'On-prem platforms',
    title: 'OpenShift and OKD on premises',
    subtitle: 'OpenShift / OKD · on-prem KVM',
    copy: 'Architected and deployed enterprise OpenShift and OKD Kubernetes clusters on on-premise KVM infrastructure.',
    metrics: ['Enterprise clusters', 'High-availability control planes'],
    tags: ['OpenShift', 'OKD', 'KVM', 'Kubernetes'],
    detail:
      'Provisioned enterprise Red Hat OpenShift and OKD container platforms on on-premise KVM virtualization. Implemented automated installation, control-plane high availability, software-defined networking, and platform storage.',
    outcome: 'Production-ready on-premise container platforms with enterprise reliability.',
    role: 'Software engineer · Platform · Jan 2023–present',
    caseStudy: {
      proprietary: true,
      problem:
        'Enterprise microservices required a reliable on-premise container platform with automated ingress, RBAC, high availability, and operational visibility.',
      design:
        'Engineered high-availability Red Hat OpenShift and OKD container platforms on KVM infrastructure. Automated cluster provisioning via Assisted Installer, established multi-node control planes, and configured platform monitoring and storage.',
      path:
        'Base virtualization provisioning, automated installer execution, control-plane orchestration, worker pool allocation, and GitOps pipeline integration.',
      constraints: [
        'Hostnames, IP ranges, and internal capacity metrics are omitted.',
      ],
      decisions: [
        {
          title: 'Automated bare-metal/KVM installation',
          copy: 'Streamlined cluster bootstrapping using automated installation workflows for consistent, reproducible node provisioning.',
        },
        {
          title: 'Standardized OKD & OpenShift topology',
          copy: 'Maintained parity between enterprise and upstream distributions for consistent deployment manifests and policies.',
        },
        {
          title: 'Platform-service synergy',
          copy: 'Designed cluster configuration directly around workload requirements, ensuring proper health probes, resource limits, and ingress routing.',
        },
      ],
      limits: [
        'Architected specifically for on-premise infrastructure constraints and dedicated virtualization environments.',
      ],
    },
  },
  {
    number: '03',
    slug: 'gitops-tekton',
    scope: 'Delivery and security',
    title: 'Tekton, Argo CD, and Trivy',
    subtitle: 'Tekton · Argo CD · Helm · DevSecOps',
    copy: 'Designed automated Tekton build pipelines, Trivy container security scans, and Argo CD GitOps delivery.',
    metrics: ['Tekton CI pipelines', 'Argo CD + Helm GitOps'],
    tags: ['Tekton', 'Argo CD', 'Helm', 'Trivy', 'Wazuh'],
    detail:
      'Engineered automated CI/CD pipelines using Tekton for containerized compilation, resource-isolated builds, and automated vulnerability scanning via Trivy. Configured Argo CD App-of-Apps and per-service applications with Helm for continuous declarative delivery on OpenShift. Integrated cluster-level security monitoring via Wazuh.',
    outcome: 'Automated GitOps pipeline enabling declarative deployments and continuous security scanning from Git.',
    role: 'Software engineer · Delivery & Security · Jan 2023–present',
    caseStudy: {
      proprietary: true,
      problem:
        'Manual deployments and unstandardized build environments caused build fragility, configuration drift, and late-stage security discovery.',
      design:
        'Built resilient Tekton pipeline tasks for containerized Java builds with memory isolation and single-threaded compilation. Integrated Trivy image scanning at build time. Established Argo CD GitOps architecture with environmental Helm values, webhook triggers, and Wazuh cluster monitoring.',
      path:
        'Git commit → automated webhook → Tekton build & Trivy scan → Argo CD declarative sync to OpenShift via Helm.',
      constraints: [
        'Internal repository URLs, pipeline timing benchmarks, and specific security finding logs are omitted.',
      ],
      decisions: [
        {
          title: 'Containerized, isolated build environments',
          copy: 'Isolated build tasks eliminate host-level dependency leaks and stabilize compilation memory consumption.',
        },
        {
          title: 'Declarative GitOps with Argo CD',
          copy: 'Git remains the single source of truth for desired state, replacing manual kubectl/oc commands with automated reconciliation.',
        },
        {
          title: 'Shift-left security scanning',
          copy: 'Trivy scans container images before registry promotion, while Wazuh provides continuous runtime cluster monitoring.',
        },
      ],
      limits: [
        'GitOps ensures declarative state consistency but relies on rigorous Helm chart linting and environmental property validations.',
      ],
    },
  },
  {
    number: '04',
    slug: 'quarkus-doctor',
    scope: 'Personal project · GitHub',
    title: 'quarkus-doctor',
    subtitle: 'quarkus-doctor · Maven plugin · CLI',
    copy: 'A Java CLI and Maven plugin that compares Quarkus config with Kubernetes, Helm, and Kustomize YAML in CI. No live cluster required.',
    metrics: ['Public · early preview', 'Not on Maven Central'],
    tags: ['Quarkus', 'Maven', 'Kubernetes', 'Helm', 'Kustomize'],
    detail:
      'KubeLinter, Checkov, and Trivy never read Quarkus config. Quarkus itself only fails at startup. quarkus-doctor is a Java CLI plus a Maven plugin—JDK 11, no Node, no live cluster—that diffs application.properties / application.yaml against this repo’s Deployment, Helm, and Kustomize YAML in CI. Bind the scan to verify and the build fails on build-time ${VAR} with no default, secret defaults in Git, localhost JDBC in a manifest, trust-all TLS, CORS * with credentials, Swagger in prod. Green means no hits in the current rule set, not an audit. Early preview; not on Maven Central yet.',
    outcome: 'Config mistakes can fail the Maven build instead of failing at pod startup.',
    role: 'Author · public project',
    href: 'https://quarkusdoctor.github.io/',
    hrefLabel: 'Documentation',
    repoUrl: 'https://github.com/fsmutimeer/quarkus-doctor',
    caseStudy: {
      problem:
        'KubeLinter, Checkov, and Trivy never open application.properties. Quarkus quarkus.config.build-time-mismatch-at-runtime fails at startup, not in CI. The crash is a CrashLoop, not a red build.',
      design:
        'quarkus-doctor is a Java CLI and a Maven plugin (JDK 11+, no Node, no live cluster). It diffs application.properties / application.yaml against this repo’s Deployment, Helm, and Kustomize YAML. Bind the scan to Maven verify and the build fails on the current rule set: build-time ${VAR} with no default, secret defaults in Git, localhost JDBC in a manifest, trust-all TLS, CORS * with credentials, Swagger in prod, and the rest of the published rules.',
      path:
        'Clone, mvn install, then mvn verify in the app module—or run the CLI JAR, or the GitHub Action that builds the CLI from source and scans a path. --ci fails on warnings. --json and a report file are available. The plugin writes target/quarkus-doctor.json. It never talks to a cluster.',
      constraints: [
        'Not on Maven Central. This is an early public preview.',
        'A green scan means no hits in the current rule set, not a full security audit.',
        'Build-time detection is a prefix list in the source, not Quarkus’s official lock-icon catalog.',
      ],
      decisions: [
        {
          title: 'Config vs YAML in CI',
          copy: 'The gap is between Quarkus config and the manifests in Git. The tool reads both. It does not need oc, kubectl, or a running API server.',
        },
        {
          title: 'Maven verify as the default gate',
          copy: 'If the scan is a separate ritual, it will be skipped. Binding scan to verify makes the failure look like any other test failure.',
        },
        {
          title: 'Fail closed on the known-bad set',
          copy: 'Trust-all TLS, secret literals in defaults, loopback JDBC in a manifest, and Swagger forced into prod are errors. Missing env in YAML can warn or info when there is no local manifest.',
        },
      ],
      limits: [
        'If env is injected by another chart, ignore ENV_VAR_NOT_IN_MANIFEST.',
        'A scan with no findings is not a guarantee the app is production-safe.',
        'Helm --fix skips templates. Ignore lists live in .quarkus-doctor.yml.',
      ],
    },
  },
];

export const experience: ExperienceRole[] = [
  {
    id: 'it22',
    company: 'IT22 B.V.',
    companyUrl: 'https://it22.nl/',
    role: 'Software engineer · backend & platform',
    location: 'Islamabad, Pakistan',
    dates: 'Jan 2023 – present',
    bullets: [
      'Led backend work on Quarkus microservices for core business platforms, using Apache Camel, Kafka, and Keycloak RBAC.',
      'Connected notification, relay, and email services through a decoupled Kafka event pipeline.',
      'Deployed on-prem OpenShift and OKD clusters on KVM, each with three control-plane nodes and one worker.',
      'Designed containerized Tekton / Maven builds and Argo CD / Helm GitOps delivery; integrated Git webhooks, Trivy scans, and Wazuh monitoring.',
    ],
  },
  {
    id: 'esols',
    company: 'ESOLS Technologies',
    companyUrl: 'https://esols.tech/',
    role: 'Software engineer',
    location: 'Islamabad, Pakistan',
    dates: 'Aug 2021 – Jan 2023',
    bullets: [
      'Built Node.js / Express backends for 6+ projects using MongoDB and Socket.IO, deployed on AWS Lightsail.',
      'Contributed to EGASI, Khebra, SmartBookings, Waves, and Brainbook; integrated AWS S3 file storage.',
    ],
  },
];

export function getProjectBySlug(slug: string): Project | undefined {
  return projects.find((project) => project.slug === slug);
}

export const doctorRules = [
  { code: 'BUILD_TIME_NULL_ENV', level: 'error', meaning: 'Build-time property references ${VAR} with no default.' },
  { code: 'SECRET_IN_ENV_DEFAULT', level: 'error', meaning: 'Secret key uses ${VAR:literal} — the default is the secret.' },
  { code: 'LOCALHOST_JDBC_IN_K8S', level: 'error', meaning: 'A client URL in a manifest points at loopback (localhost).' },
  { code: 'TLS_VERIFICATION_DISABLED', level: 'error', meaning: 'trust-all=true or tls.verification=none.' },
  { code: 'CORS_WILDCARD_WITH_CREDENTIALS', level: 'error', meaning: 'CORS origins=* with credentials enabled.' },
  { code: 'SWAGGER_IN_PROD', level: 'error', meaning: 'Swagger UI forced into prod.' },
  { code: 'PLAINTEXT_DB_PASSWORD', level: 'error', meaning: 'Literal password in config, not ${VAR}.' },
] as const;

export const stack: { number: string; title: string; copy: string; Icon: LucideIcon }[] = [
  {
    number: '01',
    title: 'One service at a time',
    copy: 'Quarkus services stay small enough that one module can ship without dragging the others.',
    Icon: Braces,
  },
  {
    number: '02',
    title: 'The cluster is operated, not assumed',
    copy: 'On-premise OpenShift and OKD clusters designed with high-availability control planes and automated operations.',
    Icon: Container,
  },
  {
    number: '03',
    title: 'Release from Git',
    copy: 'Tekton builds the commit. Argo CD syncs it. Rollback is another revision in Git.',
    Icon: Radio,
  },
  {
    number: '04',
    title: 'Scan before the pod runs',
    copy: 'Trivy runs in the pipeline. Keycloak and Wazuh handle access and monitoring.',
    Icon: Network,
  },
];

export type Technology = {
  name: string;
  label: string;
  copy: string;
  detail: string;
  Icon: IconType;
};

export const technologies: Technology[] = [
  {
    name: 'Quarkus',
    label: 'Java runtime',
    copy: 'Fast startup. Small footprint.',
    detail:
      'Quarkus is the runtime for the backend services: fast boot, small image, health endpoints the cluster can use.',
    Icon: SiQuarkus,
  },
  {
    name: 'OpenShift',
    label: 'Application platform',
    copy: 'Enterprise Kubernetes platforms.',
    detail:
      'OpenShift and OKD provide the enterprise container platform, orchestration, security controls, and GitOps delivery pipelines.',
    Icon: SiRedhatopenshift,
  },
  {
    name: 'Kubernetes',
    label: 'Cluster foundation',
    copy: 'The primitives behind OpenShift and OKD.',
    detail:
      'Kubernetes is what OpenShift and OKD are built on. Same API objects on both installs.',
    Icon: SiKubernetes,
  },
  {
    name: 'Kafka',
    label: 'Event backbone',
    copy: 'Events between services, not direct calls.',
    detail:
      'Kafka carries events between services so they communicate through contracts, not coupling.',
    Icon: SiApachekafka,
  },
];

export type LifecycleStage = {
  number: string;
  name: string;
  command: string;
  copy: string;
  outcome: string;
  Icon: LucideIcon;
};

export const lifecycle: LifecycleStage[] = [
  {
    number: '01',
    name: 'Design',
    command: 'git checkout --track',
    copy: 'Service boundary, Keycloak role, Kafka contract.',
    outcome: 'A Quarkus service with a clear owner.',
    Icon: GitBranch,
  },
  {
    number: '02',
    name: 'Build',
    command: './mvnw quarkus:build',
    copy: 'Compile, test, and scan with Trivy. Maven runs in a container with memory isolation.',
    outcome: 'An image plus a Trivy report.',
    Icon: Braces,
  },
  {
    number: '03',
    name: 'Promote',
    command: 'argocd app sync',
    copy: 'Argo CD App-of-Apps and Helm values sync that Git commit to OpenShift. Webhooks start Tekton.',
    outcome: 'A GitOps rollout that can be rolled back in Git.',
    Icon: Rocket,
  },
  {
    number: '04',
    name: 'Observe',
    command: 'oc logs -f wazuh',
    copy: 'Wazuh on the cluster for security monitoring and log analysis.',
    outcome: 'Cluster logs and alerts in one place.',
    Icon: Network,
  },
];

export const principles: { title: string; copy: string; Icon: LucideIcon }[] = [
  {
    title: 'Platform as product',
    copy: 'Standardized configurations and reproducible platform layouts designed for reliability and ease of operations.',
    Icon: Cloud,
  },
  {
    title: 'Reliability is a feature',
    copy: 'Health checks, GitOps, and image scanning belong in the design, not as cleanup.',
    Icon: ShieldCheck,
  },
  {
    title: 'Curious, then practical',
    copy: 'I try new tools when they make the next failure smaller, not because they are new.',
    Icon: Cpu,
  },
];

export const navItems = [
  {
    id: 'about',
    label: 'About',
    preview: 'Kalash, photography, IT, and the mountains still called home.',
    href: '/about/',
  },
  { id: 'experience', label: 'Experience', preview: 'Backend and platform engineering from Jan 2023. Node.js work before that.' },
  { id: 'work', label: 'Work', preview: 'Four selected pieces of work: backend services, on-prem platforms, GitOps delivery, and a public Quarkus tool.' },
  { id: 'now', label: 'Contact', preview: 'Email, phone, CV, GitHub, LinkedIn.' },
] as const;

export type NavItem = (typeof navItems)[number];
