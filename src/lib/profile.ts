import { withBasePath } from '@/lib/base-path';

export const profile = {
  name: 'Feroz Shah',
  initials: 'FS',
  handle: 'fsmutimeer',
  role: 'Backend & platform engineer',
  company: 'IT22 B.V.',
  companyUrl: 'https://it22.nl/',
  location: 'Islamabad, Pakistan',
  email: 'ishpata@hotmail.com',
  secondaryEmail: 'ishpata@icloud.com',
  emails: ['ishpata@hotmail.com', 'ishpata@icloud.com'],
  phone: '+92 333 7022773',
  phoneHref: 'tel:+923337022773',
  cvUrl: withBasePath('/feroz_shah_cv.pdf'),
  terminalUser: 'fsmutimeer@platform ~/signal',
  whoami: 'feroz.shah — software engineer · backend & platform',
  timezone: 'PKT · UTC+5',
  githubUrl: 'https://github.com/fsmutimeer',
  linkedinUrl: 'https://www.linkedin.com/in/fsmutimeer/',
  facebookUrl: 'https://facebook.com/fsmutimeer',
  instagramUrl: 'https://instagram.com/fsmutimeer',
  hero: {
    eyebrow: 'backend & platform engineer',
    copy:
      'I build Java and Quarkus backend services, event-driven integrations, and cloud-native Kubernetes platforms.',
    stack: 'Java · Quarkus · Kafka · Kubernetes · OpenShift · GitOps',
  },
  about: {
    headline: 'From Kalash to the Cluster.',
    summary: [
      'Growing up in the remote mountain valleys of Kalash, my path evolved from self-taught programming to designing distributed enterprise systems.',
      'Today I engineer at the intersection of backend architecture and cloud platforms—building decoupled Quarkus services and resilient Kubernetes environments.',
    ],
    education: 'M.Sc Information Technology · Quaid-i-Azam University · 2017–2019',
    capabilities: [
      {
        category: 'Backend',
        tools: ['Java', 'Quarkus', 'Hibernate', 'Apache Camel', 'Node.js'],
        copy: 'Services and integration flows, connecting Quarkus, Camel, Kafka, and Keycloak.',
      },
      {
        category: 'Database',
        tools: ['MongoDB', 'MySQL'],
        copy: 'Aggregation pipelines in MongoDB; relational data in MySQL.',
      },
      {
        category: 'Messaging',
        tools: ['Kafka'],
        copy: 'Event-driven service communication via Apache Kafka.',
      },
      {
        category: 'Platform',
        tools: ['OpenShift', 'OKD', 'Kubernetes', 'Docker', 'Podman', 'Helm', 'Tekton', 'Argo CD'],
        copy: 'On-premise OpenShift and OKD platforms; GitOps delivery with Tekton and Argo CD.',
      },
      {
        category: 'Security',
        tools: ['Keycloak', 'Wazuh', 'Trivy'],
        copy: 'RBAC via Keycloak, image scanning with Trivy, cluster monitoring with Wazuh.',
      },
    ],
  },
} as const;
