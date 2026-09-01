import { withBasePath } from '@/lib/base-path';

export const profile = {
  name: 'Feroz Shah',
  initials: 'FS',
  handle: 'fsmutimeer',
  role: 'Software engineer · backend & platform',
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
  hero: {
    eyebrow: 'software engineer · backend & platform · IT22 B.V. · islamabad',
    lines: ['Software engineer', 'backend and platform', 'IT22 B.V.'] as const,
    copy:
      'a software engineer at IT22 B.V. in Islamabad. The job title is software engineer. The work is Java/Quarkus services and the OpenShift/OKD clusters they run on, with Kafka, GitOps, and Tekton.',
    stack: 'Java · Quarkus · Kafka · Kubernetes · OpenShift · GitOps',
  },
  about: {
    headline: 'Software engineer. Backend and platform.',
    summary: [
      'I am a software engineer at IT22 B.V. in Islamabad. That is my current employer and job title.',
      'The work covers two areas in the same role: backend services (Java, Quarkus, Apache Camel, Kafka, MongoDB, Keycloak) and the platforms those services run on (OpenShift, OKD, Kubernetes, Tekton, Argo CD, Helm, Trivy, Wazuh).',
      'I care about services that are straightforward to operate and a delivery path that does not need a special ritual for each release.',
    ],
    previous:
      'Before IT22 I was a software engineer at ESOLS Technologies (Aug 2021 – Jan 2023), writing Node.js backends. That is previous employment, not my current stack.',
    education: 'M.Sc Information Technology · Quaid-i-Azam University · 2017–2019',
    focus: [
      {
        title: 'Java · Quarkus · Apache Camel',
        copy: 'I write the backend services and integration flows at IT22.',
      },
      {
        title: 'Kafka · MongoDB',
        copy: 'Services talk over Kafka. Data and retrieval sit in MongoDB, including aggregation pipelines.',
      },
      {
        title: 'OpenShift · OKD · Kubernetes',
        copy: 'I deployed an OpenShift cluster and an OKD cluster on-prem on KVM. Each has three control-plane nodes and one worker.',
      },
      {
        title: 'GitOps · Argo CD · Tekton',
        copy: 'A Git commit is built in Tekton, scanned, and synced to OpenShift with Argo CD and Helm.',
      },
      {
        title: 'Keycloak · Wazuh · Trivy',
        copy: 'Keycloak for RBAC. Trivy scans images in CI. Wazuh monitors the cluster.',
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
    ],
  },
} as const;
