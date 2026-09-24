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
    scope: 'IT22 B.V. · current job · not a separate employer',
    title: 'Quarkus microservices with Kafka and Keycloak',
    subtitle: 'Quarkus · Camel · Kafka · IT22 B.V.',
    copy: 'At IT22 I lead backend work on ERP and PMS microservices: Kafka between services, MongoDB for retrieval, Keycloak for RBAC.',
    metrics: ['IT22 · proprietary', 'Same job as Experience'],
    tags: ['Quarkus', 'Apache Camel', 'Kafka', 'MongoDB', 'Keycloak'],
    detail:
      'At IT22 I lead backend work on Quarkus microservices for ERP and PMS systems. Apache Camel and Kafka move events between services. MongoDB aggregation pipelines are used for retrieval. Keycloak provides RBAC across modules. Notification, relay, and email services consume the same event path.',
    outcome: 'Modules can grow without sharing a write path or a permission model.',
    role: 'Software engineer · IT22 B.V. · Jan 2023–present',
    caseStudy: {
      proprietary: true,
      problem:
        'ERP and PMS modules that share a database or call each other synchronously fail together. A change in one bounded context becomes an incident in another.',
      design:
        'At IT22 I lead backend work on Quarkus microservices for those systems. Apache Camel and Kafka carry events between services. MongoDB aggregation pipelines are used for retrieval. Keycloak holds RBAC across modules. Notification, relay, and email sit on the same event path.',
      path:
        'A module publishes an event. Other services consume it. Notification, relay, and email are consumers on that path, not hidden inside another module.',
      constraints: [
        'Customer names, tenant counts, and internal topic names are omitted.',
        'This is one part of the IT22 job, not a second company.',
      ],
      decisions: [
        {
          title: 'Events instead of a shared write path',
          copy: 'Kafka keeps service talk explicit. Replay and ownership stay possible. A shared database is not the integration.',
        },
        {
          title: 'Camel for the integration flow',
          copy: 'The flow is a first-class artifact. New consumers attach to the contract instead of reaching into another module.',
        },
        {
          title: 'Keycloak as the role source',
          copy: 'RBAC is dynamic and shared. A module does not invent its own permission model and hope it matches the next one.',
        },
      ],
      limits: [
        'Events add operational work (Kafka) in exchange for looser coupling. I am not publishing throughput or user counts.',
      ],
    },
  },
  {
    number: '02',
    slug: 'openshift-okd',
    scope: 'IT22 B.V. · current job · not a separate employer',
    title: 'OpenShift and OKD on premises',
    subtitle: 'OpenShift / OKD · on-prem KVM',
    copy: 'Deployed a Red Hat OpenShift cluster via Assisted Installer and an OKD cluster on KVM—each with three control-plane nodes and one worker—on premises.',
    metrics: ['3 control + 1 worker each', 'IT22 · proprietary'],
    tags: ['OpenShift', 'OKD', 'KVM', 'Kubernetes'],
    detail:
      'I deployed a Red Hat OpenShift cluster with Assisted Installer from console.redhat.com (three control-plane nodes, one worker, KVM, on premises) and an OKD cluster with the same node counts, also on KVM. The Quarkus services at IT22 run on this platform.',
    outcome: 'Two on-prem clusters, each 3 control + 1 worker.',
    role: 'Software engineer · IT22 B.V. · Jan 2023–present',
    caseStudy: {
      proprietary: true,
      problem:
        'The Java services need a cluster we operate. A laptop Kubernetes install is not that.',
      design:
        'I deployed a Red Hat OpenShift cluster via Assisted Installer from console.redhat.com: three control-plane nodes and one worker on KVM, on premises. I also deployed OKD on KVM with three control nodes and one worker. These are the clusters the IT22 Quarkus services run on.',
      path:
        'Installer and KVM first, then control plane and worker. The GitOps case study is the delivery path onto this platform.',
      constraints: [
        'Node counts on this page match the CV: 3 control + 1 worker, on both OpenShift and OKD.',
        'Hostnames and capacity numbers are omitted. This is the same IT22 job as the other IT22 cards.',
      ],
      decisions: [
        {
          title: 'Assisted Installer, on premises',
          copy: 'OpenShift comes from console.redhat.com Assisted Installer onto KVM we operate.',
        },
        {
          title: 'OKD with the same node counts',
          copy: 'OKD uses the same 3 control + 1 worker layout so the two platforms stay comparable.',
        },
        {
          title: 'Same person as the services',
          copy: 'Backend work and cluster work are the same IT22 role. I am not describing a separate platform team.',
        },
      ],
      limits: [
        '3 control + 1 worker is the installed footprint, not a claim about large multi-cluster estates.',
        'This page lists those two installs only.',
      ],
    },
  },
  {
    number: '03',
    slug: 'gitops-tekton',
    scope: 'IT22 B.V. · current job · not a separate employer',
    title: 'Tekton, Argo CD, and Trivy',
    subtitle: 'Tekton · Argo CD · Helm · DevSecOps',
    copy: 'At IT22 I designed Tekton pipelines, Argo CD App-of-Apps with Helm, Git webhooks, Trivy in CI, and Wazuh on the cluster.',
    metrics: ['IT22 · proprietary', 'Same job as Experience'],
    tags: ['Tekton', 'Argo CD', 'Helm', 'Trivy', 'Wazuh'],
    detail:
      'I designed Tekton pipelines that survived Maven’s “too many open files” class of failure: containerized builds with single-threaded compilation and JVM memory isolation, consistent across Java versions. Argo CD applications—App-of-Apps and per-service—wire Git and Helm for GitOps on OpenShift. Git webhooks kick pipelines and syncs. Trivy scans images before they ship; Wazuh watches the cluster for the rest.',
    outcome: 'A Git commit can build, scan, and sync to OpenShift.',
    role: 'Software engineer · IT22 B.V. · Jan 2023–present',
    caseStudy: {
      proprietary: true,
      problem:
        'Maven builds were failing with too-many-open-files errors. Deploys and image scans were easy to skip or do too late.',
      design:
        'I designed Tekton pipelines with containerized Maven builds: single-threaded compilation and JVM memory isolation, consistent across Java versions. Argo CD applications (App-of-Apps and per-service) sync Git and Helm to OpenShift. Git webhooks start pipelines and syncs. Helm values are per environment. Trivy scans images before they ship. Wazuh monitors the cluster.',
      path:
        'Git push → webhook → Tekton build and Trivy scan → Argo CD sync of that commit with Helm values. Rollback is another Git revision.',
      constraints: [
        'I am not publishing pipeline duration, image size, or CVE counts.',
        'Internal repo and registry names are omitted. Same IT22 job as cards 01 and 02.',
      ],
      decisions: [
        {
          title: 'Fix the build infrastructure, then automate it',
          copy: 'The pipeline has to survive Maven’s open-files failures. Memory isolation in the container is how that run stays stable.',
        },
        {
          title: 'App-of-Apps plus per-service apps',
          copy: 'Argo CD syncs Git. Helm values differ by environment. The cluster follows Git rather than a one-off oc apply.',
        },
        {
          title: 'Scan before the pod exists',
          copy: 'Trivy runs in CI. Wazuh runs on the cluster for monitoring and log analysis.',
        },
      ],
      limits: [
        'GitOps does not replace a correct Helm chart. I am not claiming a deployment-frequency or incident-reduction number.',
      ],
    },
  },
  {
    number: '04',
    slug: 'quarkus-doctor',
    scope: 'Public project · my GitHub · not an IT22 product page',
    title: 'quarkus-doctor',
    subtitle: 'quarkus-doctor · Maven plugin · CLI',
    copy: 'A Java CLI and Maven plugin I wrote. It compares Quarkus application.properties / application.yaml with Kubernetes, Helm, and Kustomize YAML in CI. It does not talk to a cluster.',
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
      'Led the backend team on Quarkus microservices for ERP and PMS systems, with Apache Camel, Kafka, MongoDB, and Keycloak RBAC across modules.',
      'Put notification, relay, and email services on the same Kafka event path so cross-platform communication is a contract.',
      'Deployed OpenShift (Assisted Installer) and OKD on-prem on KVM—each with three control-plane nodes and one worker.',
      'Designed Tekton pipelines with containerized Maven builds (open-files / JVM isolation) and Argo CD App-of-Apps plus Helm GitOps on OpenShift.',
      'Integrated Git webhooks for pipeline runs and syncs; Trivy in CI; Wazuh on the cluster for security monitoring.',
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
      'Built and maintained Node.js / Express backends for 6+ projects, with MongoDB, Socket.IO, AWS S3, and AWS Lightsail.',
      'Worked on EGASI, Khebra, SmartBookings, Waves, and Brainbook.',
      'Integrated AWS S3 for file storage and retrieval.',
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
    copy: 'OpenShift and OKD here are clusters I installed. The layout is 3 control + 1 worker each.',
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
    copy: 'Trivy runs in the pipeline. Keycloak and Wazuh are part of the same IT22 role, not extras.',
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
      'Quarkus is the runtime for the IT22 services: fast boot, small image, health endpoints the cluster can use.',
    Icon: SiQuarkus,
  },
  {
    name: 'OpenShift',
    label: 'Application platform',
    copy: 'On-prem OpenShift and OKD at IT22.',
    detail:
      'OpenShift and OKD are where the IT22 services run. GitOps, Helm, and Tekton deploy to those clusters.',
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
    copy: 'Events between IT22 services, not a shared database.',
    detail:
      'Kafka carries events between services so they do not integrate through one shared database.',
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
    outcome: 'A Quarkus service with an owner.',
    Icon: GitBranch,
  },
  {
    number: '02',
    name: 'Build',
    command: './mvnw quarkus:build',
    copy: 'Compile, test, and scan with Trivy. Maven runs in a container with memory isolation so open-files errors do not kill the build.',
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
    copy: 'Defaults and a cluster layout I can explain, because I installed it.',
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
  { id: 'experience', label: 'Experience', preview: 'IT22 from Jan 2023. ESOLS Aug 2021–Jan 2023.' },
  { id: 'work', label: 'Work', preview: 'Three views of the IT22 job, plus quarkus-doctor.' },
  { id: 'approach', label: 'Approach', preview: 'The IT22 path from Git to OpenShift. Not another job.' },
  { id: 'now', label: 'Now', preview: 'Employed at IT22 B.V. in Islamabad.' },
  { id: 'contact', label: 'Contact', preview: 'Email, phone, CV, GitHub, LinkedIn.' },
] as const;
