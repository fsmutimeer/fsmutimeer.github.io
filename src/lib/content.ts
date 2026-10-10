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
  /** Optional "What I'd change" note, shown on the case-study page when set. */
  retrospective?: string;
  proprietary?: boolean;
};

export type Project = {
  number: string;
  slug: string;
  scope: string;
  /** Short badge on the card, e.g. license/version or "Production · private". */
  status?: string;
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
    slug: 'migrax',
    scope: 'Open source · Java CLI',
    status: 'MIT · v0.1.3',
    title: 'Migrax',
    subtitle: 'migrax · CLI · Maven / Gradle plugin',
    copy: 'Generates SQL migrations from JPA entities, then lints them for table locks and data-loss risks before they reach production.',
    metrics: ['Hibernate 5.4 – 7.4 · 6 databases', 'SQL + rollback · drift detection'],
    tags: ['Java 17', 'Hibernate', 'JPA', 'SQL', 'Maven', 'Gradle'],
    detail:
      'Migrax reads JPA entities, compares them with the database, and writes plain SQL migrations with a rollback script. It covers tables, columns, keys, indexes, join tables, element collections, sequences, and inheritance the way Hibernate maps them. lint flags statements that lock large tables, fail on existing rows, or break running instances; verify checks the result against your Hibernate version and rolls the migration back and forward; drift compares the live database with what the migrations produce. Works with Spring Boot, Quarkus, Micronaut, Helidon, Jakarta EE, and plain Hibernate on PostgreSQL, MySQL 8, MariaDB, SQL Server, Oracle 12c+, and H2.',
    outcome: 'Schema changes ship as reviewable SQL with a tested rollback instead of hand-written scripts.',
    role: 'Author · open source',
    href: 'https://docs-migrax.github.io/',
    hrefLabel: 'Documentation',
    repoUrl: 'https://github.com/fsmutimeer/migrax',
    caseStudy: {
      problem:
        'Entities change in Java, but the migration that changes the database is still written by hand. A hand-written ALTER can lock a large table, fail on existing rows, or break instances still running the old code — and that usually shows up in production, not review.',
      design:
        'Migrax is a Java 17+ tool that reads JPA entities, compares them with the database, and writes plain SQL plus a rollback script. Each supported Hibernate version (5.4 to 7.4) reads its own mapping, so tables, keys, indexes, join tables, element collections, sequences, and inheritance match what Hibernate expects. No configuration files: it compiles the project, finds dependencies, and reads database settings from the application config.',
      path: 'Change JPA entities → migrax generate → Review SQL + rollback → migrax verify → migrax migrate',
      constraints: [
        'Early release (0.1.x), MIT licensed.',
        'Requires Java 17+.',
        'Supports PostgreSQL, MySQL 8, MariaDB, SQL Server, Oracle 12c+, and H2.',
      ],
      decisions: [
        {
          title: 'Plain SQL, not a black box',
          copy: 'Every migration is SQL you can read and edit before it runs, with a rollback script next to it.',
        },
        {
          title: 'Lint before it ships',
          copy: 'Statements that lock large tables, fail on existing rows, or break running instances are flagged before the migration reaches production.',
        },
        {
          title: 'Destructive changes are opt-in',
          copy: 'When a dropped column looks like a rename, Migrax asks and renames it to keep the data. Drops only happen with --allow-destructive.',
        },
      ],
      limits: [
        'Lint flags known risk patterns; a clean lint is not a guarantee that a migration is safe on every dataset.',
        'verify proves the rollback works against your Hibernate version — it does not replace testing with production-sized data.',
      ],
    },
  },
  {
    number: '02',
    slug: 'quarkus-doctor',
    scope: 'Open source · Maven plugin',
    status: 'Early preview',
    title: 'quarkus-doctor',
    subtitle: 'quarkus-doctor · Maven plugin · CLI',
    copy: 'A Java CLI and Maven plugin that compares Quarkus config with Kubernetes, Helm, and Kustomize YAML in CI. No live cluster required.',
    metrics: ['Fails mvn verify on config errors', 'JDK 11+ · no cluster needed'],
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
  {
    number: '03',
    slug: 'gitops-tekton',
    scope: 'Delivery and security',
    status: 'Production · private',
    title: 'Tekton, Argo CD, and Trivy',
    subtitle: 'Tekton · Argo CD · Helm · DevSecOps',
    copy: 'Designed automated Tekton build pipelines, Trivy container security scans, and Argo CD GitOps delivery.',
    metrics: ['Commit → Tekton → Trivy → Argo CD', 'Helm App-of-Apps'],
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

export const navItems = [
  {
    id: 'about',
    label: 'About',
    preview: 'Kalash, photography, IT, and the mountains still called home.',
    href: '/about/',
  },
  { id: 'experience', label: 'Experience', preview: 'Backend and platform engineering from Jan 2023. Node.js work before that.' },
  { id: 'work', label: 'Work', preview: 'Two open source tools — Migrax and quarkus-doctor — plus GitOps delivery on OpenShift.' },
  { id: 'now', label: 'Contact', preview: 'Email, phone, CV, GitHub, LinkedIn.' },
] as const;

export type NavItem = (typeof navItems)[number];
