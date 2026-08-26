import { withBasePath } from '@/lib/base-path';

export const profile = {
  name: 'Feroz Shah',
  initials: 'FS',
  handle: 'fsmutimeer',
  role: 'software engineer',
  company: 'IT22 B.V.',
  companyUrl: 'https://it22.nl/',
  location: 'Islamabad, Pakistan',
  email: 'ishpata@hotmail.com',
  phone: '+92 333 7022773',
  phoneHref: 'tel:+923337022773',
  cvUrl: withBasePath('/feroz_shah_cv.pdf'),
  terminalUser: 'fsmutimeer@platform ~/signal',
  whoami: 'feroz.shah — software engineer · IT22',
  timezone: 'PKT · UTC+5',
  githubUrl: 'https://github.com/fsmutimeer',
  linkedinUrl: 'https://www.linkedin.com/in/fsmutimeer/',
  about: {
    headline: 'Engineer at the seam of services and platforms.',
    summary:
      'Software engineer at IT22 B.V. in Islamabad. I lead backend work on Quarkus microservices—Apache Camel, Kafka, MongoDB, Keycloak—and I stand up the OpenShift / OKD clusters those services run on, with Tekton, Argo CD, and Helm taking them to production. Before that I shipped Node.js backends at ESOLS Technologies.',
    education: 'M.Sc Information Technology · Quaid-i-Azam University · 2017–2019',
    focus: [
      {
        title: 'Java · Quarkus · Apache Camel',
        copy: 'Backend services, integration flows, and contracts that stay small enough to reason about in production.',
      },
      {
        title: 'Kafka · MongoDB',
        copy: 'Event-driven paths between services, plus aggregation pipelines that keep retrieval fast as the data grows.',
      },
      {
        title: 'OpenShift · OKD · Kubernetes',
        copy: 'On-prem clusters on KVM—control plane, workers, and the guardrails that make the happy path the default path.',
      },
      {
        title: 'GitOps · Argo CD · Tekton',
        copy: 'App-of-Apps, Helm, webhooks, and pipelines that promote a known commit without heroics.',
      },
      {
        title: 'Keycloak · Wazuh · Trivy',
        copy: 'Identity, cluster security monitoring, and image scanning in the delivery path—before a CVE becomes a deploy.',
      },
    ],
    stack: [
      'Java',
      'Quarkus',
      'Apache Camel',
      'Kafka',
      'MongoDB',
      'Keycloak',
      'OpenShift',
      'OKD',
      'Kubernetes',
      'Tekton',
      'Argo CD',
      'Helm',
      'Wazuh',
      'Trivy',
      'Docker',
      'Node.js',
    ],
  },
} as const;
