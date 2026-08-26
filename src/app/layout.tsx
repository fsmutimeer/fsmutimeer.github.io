import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import { withBasePath } from '@/lib/base-path';
import { profile } from '@/lib/profile';
import './globals.css';

export const metadata: Metadata = {
  title: `${profile.name} — Software Engineer`,
  description:
    `${profile.name} is a software engineer at IT22 B.V. in Islamabad, building Quarkus microservices and OpenShift / OKD platforms with Kafka, Tekton, and Argo CD.`,
  robots: { index: true, follow: true },
  alternates: { canonical: '/' },
  openGraph: {
    title: `${profile.name} — Software Engineer`,
    description:
      'Code that survives production. Java, Quarkus, OpenShift, OKD, Kafka, GitOps, and Tekton.',
    type: 'website',
    url: '/',
    siteName: profile.name,
  },
  twitter: {
    card: 'summary_large_image',
    title: `${profile.name} — Software Engineer`,
    description:
      'Code that survives production. Java, Quarkus, OpenShift, OKD, Kafka, GitOps, and Tekton.',
  },
  icons: { icon: withBasePath('/favicon.svg') },
};

const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'Person',
  name: profile.name,
  jobTitle: 'Software Engineer',
  worksFor: {
    '@type': 'Organization',
    name: 'IT22 B.V.',
    url: 'https://it22.nl/',
  },
  address: {
    '@type': 'PostalAddress',
    addressLocality: 'Islamabad',
    addressCountry: 'PK',
  },
  description:
    'Software engineer building Quarkus microservices and OpenShift platforms at IT22 B.V.',
  email: `mailto:${profile.email}`,
  telephone: profile.phoneHref.replace('tel:', ''),
  url: 'https://fsmutimeer.github.io/',
  sameAs: [profile.githubUrl, profile.linkedinUrl],
  knowsAbout: [
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
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="theme-color" content="#050807" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
