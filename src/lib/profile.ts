export const profile = {
  name: 'Feroz Shah',
  initials: 'FS',
  handle: 'fsmutimeer',
  role: 'platform engineer',
  email: 'ishpata@hotmail.com',
  terminalUser: 'fsmutimeer@platform ~/signal',
  whoami: 'feroz.shah — platform engineer',
  timezone: 'UTC−05',
  githubUrl: 'https://github.com/fsmutimeer',
  linkedinUrl: 'https://www.linkedin.com/in/fsmutimeer/',
  about: {
    headline: 'Engineer at the seam of services and platforms.',
    summary:
      'I build and operate cloud-native systems where Java services meet Kubernetes reality. My day-to-day sits across Quarkus and Apache Camel integrations, Kafka-backed event flows, and OpenShift / OKD platforms delivered through GitOps and Tekton pipelines.',
    focus: [
      {
        title: 'Java · Quarkus · Apache Camel',
        copy: 'Service design, integration flows, and contracts that stay small enough to reason about in production.',
      },
      {
        title: 'Kafka',
        copy: 'Event-driven paths with clear ownership, replayability, and failure modes you can explain under pressure.',
      },
      {
        title: 'OpenShift · OKD · Kubernetes',
        copy: 'Namespaces, workload lifecycle, and platform guardrails that make the happy path the default path.',
      },
      {
        title: 'GitOps · Argo CD · Tekton',
        copy: 'Repeatable promotion from commit to cluster—pipelines, sync, and rollback without heroics.',
      },
    ],
    stack: [
      'Java',
      'Quarkus',
      'Apache Camel',
      'Kafka',
      'OpenShift',
      'OKD',
      'Kubernetes',
      'GitOps',
      'Argo CD',
      'Tekton',
    ],
  },
} as const;
