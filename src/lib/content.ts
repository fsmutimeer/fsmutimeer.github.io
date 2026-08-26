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

export type Project = {
  number: string;
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
};

export const projects: Project[] = [
  {
    number: '01',
    title: 'Services that talk without a shared blast radius',
    subtitle: 'Quarkus · Camel · Kafka · IT22 B.V.',
    copy: 'Backend lead on ERP and PMS microservices: event paths, fast retrieval, and RBAC that does not leak across modules.',
    metrics: ['Keycloak RBAC', 'Kafka event paths'],
    tags: ['Quarkus', 'Apache Camel', 'Kafka', 'MongoDB', 'Keycloak'],
    detail:
      'At IT22 I lead backend work on scalable microservices for ERP and PMS systems. The useful pieces are the seams: Camel and Kafka for event-driven communication, MongoDB aggregation pipelines for retrieval that stays fast, and Keycloak for dynamic role-based access across modules. Notification, relay, and email services sit on the same event path so cross-platform communication is a contract, not a side effect.',
    outcome: 'A backend that can grow a module without growing the blast radius.',
    role: 'Software engineer · backend lead · IT22 B.V. · 2023–present',
  },
  {
    number: '02',
    title: 'A cluster you can stand up on purpose',
    subtitle: 'OpenShift / OKD · on-prem KVM',
    copy: 'Deployed a Red Hat OpenShift cluster via Assisted Installer and an OKD cluster on KVM—each with three control-plane nodes and one worker—on premises.',
    metrics: ['3 control + 1 worker', 'OpenShift + OKD'],
    tags: ['OpenShift', 'OKD', 'KVM', 'Kubernetes'],
    detail:
      'The platform work is literal: an OpenShift cluster from console.redhat.com Assisted Installer, three control-plane nodes and one worker on KVM, and a matching OKD footprint. That is the foundation the Java services actually run on—not a slide, a rack.',
    outcome: 'Production-shaped OpenShift and OKD clusters running on-prem, under our own hands.',
    role: 'Software engineer · cluster build · IT22 B.V.',
  },
  {
    number: '03',
    title: 'The path from commit to a scanned image',
    subtitle: 'Tekton · Argo CD · Helm · DevSecOps',
    copy: 'Tekton pipelines, Argo CD App-of-Apps, Helm values per environment, and Trivy in CI—so a Git push can build, scan, and sync without a hero on the call.',
    metrics: ['GitOps App-of-Apps', 'Trivy in the pipeline'],
    tags: ['Tekton', 'Argo CD', 'Helm', 'Trivy', 'Wazuh'],
    detail:
      'I designed Tekton pipelines that survived Maven’s “too many open files” class of failure: containerized builds with single-threaded compilation and JVM memory isolation, consistent across Java versions. Argo CD applications—App-of-Apps and per-service—wire Git and Helm for GitOps on OpenShift. Git webhooks kick pipelines and syncs. Trivy scans images before they ship; Wazuh watches the cluster for the rest.',
    outcome: 'A promotion path that is GitOps-driven, scanned, and repeatable.',
    role: 'Software engineer · delivery & security · IT22 B.V.',
  },
  {
    number: '04',
    title: 'A plugin that fails the scan, not the pod',
    subtitle: 'quarkus-doctor · Maven plugin · CLI',
    copy: 'I built quarkus-doctor: a Java CLI and Maven plugin that compares application.properties with Kubernetes, Helm, and Kustomize manifests in CI—before the image ships.',
    metrics: ['Maven verify goal', 'No cluster required'],
    tags: ['Quarkus', 'Maven', 'Kubernetes', 'Helm', 'Kustomize'],
    detail:
      'KubeLinter, Checkov, and Trivy never read Quarkus config. Quarkus itself only fails at startup. quarkus-doctor is a Java CLI plus a Maven plugin—JDK 11, no Node, no live cluster—that diffs application.properties / application.yaml against this repo’s Deployment, Helm, and Kustomize YAML in CI. Bind the scan to verify and the build fails on build-time ${VAR} with no default, secret defaults in Git, localhost JDBC in a manifest, trust-all TLS, CORS * with credentials, Swagger in prod. Green means no hits in the current rule set, not an audit. Early preview; not on Maven Central yet.',
    outcome: 'Quarkus config vs Kubernetes YAML in CI, not a CrashLoop at startup.',
    role: 'Author · quarkus-doctor',
    href: 'https://quarkusdoctor.github.io/',
    hrefLabel: 'Documentation',
    repoUrl: 'https://github.com/fsmutimeer/quarkus-doctor',
  },
];

export const stack: { number: string; title: string; copy: string; Icon: LucideIcon }[] = [
  {
    number: '01',
    title: 'The service is the unit',
    copy: 'Quarkus stays small enough that a module can ship without dragging the rest of the estate with it.',
    Icon: Braces,
  },
  {
    number: '02',
    title: 'The cluster is a product',
    copy: 'A happy path people can explain on call—not a pile of YAML that only one engineer understands.',
    Icon: Container,
  },
  {
    number: '03',
    title: 'Promotion is a Git event',
    copy: 'A known commit moves. A meeting does not. Rollback is a revision, not a war room.',
    Icon: Radio,
  },
  {
    number: '04',
    title: 'Security is in the path',
    copy: 'Roles, scans, and cluster signal happen before the pod exists—not after the incident.',
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
      'Quarkus is the runtime for the IT22 services: fast boot, small image, health that the cluster can actually use.',
    Icon: SiQuarkus,
  },
  {
    name: 'OpenShift',
    label: 'Application platform',
    copy: 'On-prem clusters that teams can actually ship to.',
    detail:
      'OpenShift is where those services live. GitOps, Helm values, and pipelines treat the cluster as a product—not a one-off rack.',
    Icon: SiRedhatopenshift,
  },
  {
    name: 'Kubernetes',
    label: 'Cluster foundation',
    copy: 'The primitives behind OpenShift and OKD.',
    detail:
      'Kubernetes is the shared language under OpenShift and OKD. Same primitives, same intent, fewer surprises when the platform name changes.',
    Icon: SiKubernetes,
  },
  {
    name: 'Kafka',
    label: 'Event backbone',
    copy: 'Services that communicate without a shared database.',
    detail:
      'Kafka keeps service talk explicit: events you can own, replay, and reason about—without a shared database becoming the integration.',
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
    copy: 'Start with a service boundary, a Keycloak role, and the Kafka contract worth making visible.',
    outcome: 'A small, testable Quarkus service with an owner.',
    Icon: GitBranch,
  },
  {
    number: '02',
    name: 'Build',
    command: './mvnw quarkus:build',
    copy: 'Compile, test, and scan with Trivy. Maven runs in a container with memory isolation so the pipeline does not fall over on open files.',
    outcome: 'A reproducible image with a CVE report attached.',
    Icon: Braces,
  },
  {
    number: '03',
    name: 'Promote',
    command: 'argocd app sync',
    copy: 'Argo CD App-of-Apps and Helm values move a known Git commit onto OpenShift. Webhooks start the Tekton run.',
    outcome: 'A GitOps rollout with a rollback path.',
    Icon: Rocket,
  },
  {
    number: '04',
    name: 'Observe',
    command: 'oc logs -f wazuh',
    copy: 'Wazuh on the cluster for centralized security monitoring and log analysis—so drift shows up as a signal, not a surprise.',
    outcome: 'A platform that tells the truth under pressure.',
    Icon: Network,
  },
];

export const principles: { title: string; copy: string; Icon: LucideIcon }[] = [
  {
    title: 'Platform as product',
    copy: 'Clear paths, useful defaults, and a cluster you can explain to the person on call.',
    Icon: Cloud,
  },
  {
    title: 'Reliability is a feature',
    copy: 'Health checks, GitOps promotion, and image scanning are part of the design—not cleanup.',
    Icon: ShieldCheck,
  },
  {
    title: 'Curious, then practical',
    copy: 'I like new tools. I like them more when they make tomorrow’s incident smaller.',
    Icon: Cpu,
  },
];

export const navItems = [
  { id: 'about', label: 'About', preview: 'Who I am, where I work, and the stack I actually use.' },
  { id: 'work', label: 'Work', preview: 'IT22 systems, plus quarkus-doctor — the plugin I built.' },
  { id: 'approach', label: 'Approach', preview: 'How a commit becomes a scanned, running pod.' },
  { id: 'now', label: 'Now', preview: 'Shipping at IT22 B.V. in Islamabad.' },
  { id: 'contact', label: 'Contact', preview: 'Email, CV, and a direct line.' },
] as const;
