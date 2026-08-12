import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import { profile } from '@/lib/profile';
import './globals.css';

export const metadata: Metadata = {
  title: `${profile.name} — Platform Engineer`,
  description:
    `${profile.name} is a hands-on platform engineer turning Java and Quarkus services into reliable cloud-native systems on OpenShift and Kubernetes.`,
  robots: { index: true, follow: true },
  alternates: { canonical: '/' },
  openGraph: {
    title: `${profile.name} — Platform Engineer`,
    description:
      'Code that survives production. Java, Quarkus, OpenShift, Kubernetes, observability, and CI/CD.',
    type: 'website',
    url: '/',
    siteName: profile.name,
  },
  twitter: {
    card: 'summary_large_image',
    title: `${profile.name} — Platform Engineer`,
    description:
      'Code that survives production. Java, Quarkus, OpenShift, Kubernetes, observability, and CI/CD.',
  },
  icons: { icon: '/favicon.svg' },
};

const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'Person',
  name: profile.name,
  jobTitle: 'Platform Engineer',
  description:
    'Hands-on platform engineer turning Java services into reliable cloud-native systems.',
  email: `mailto:${profile.email}`,
  sameAs: [profile.githubUrl, profile.linkedinUrl],
  knowsAbout: [
    'Java',
    'Quarkus',
    'OpenShift',
    'OKD',
    'Kubernetes',
    'DevOps',
    'Observability',
    'CI/CD',
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
