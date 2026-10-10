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
    headline: 'I build the backends that power AI-ready products.',
    copy:
      'Software engineer building cloud-native Java and Quarkus services, Kafka event-driven integrations, and the Kubernetes / OpenShift platforms they run on.',
    stack: 'Java · Quarkus · Kafka · Kubernetes · OpenShift · GitOps',
    // Background portrait on the right of the landing page, cropped to the face.
    // To replace it: drop a new file into public/images/, point `image` at it, and set
    // `face` to where the face sits in that image, as fractions of its width/height
    // (x = centre of the face, top = top of the hair, height = hair-to-chin).
    image: '/images/hero.png',
    face: { x: 0.49, top: 0.11, height: 0.29 },
  },
  about: {
    headline: ['Backend engineer,', 'platform owner.'],
    summary: [
      'I build Java and Quarkus microservices and help run the OpenShift clusters they ship to, from Kafka integrations to GitOps delivery.',
      'Self-taught, my path started in the mountain valleys of Kalash and led to distributed enterprise systems.',
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
