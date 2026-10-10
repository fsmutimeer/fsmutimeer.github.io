import type { Metadata } from "next";
import type { ReactNode } from "react";
import { DM_Mono, Manrope, Sora } from "next/font/google";
import { withBasePath } from "@/lib/base-path";
import { profile } from "@/lib/profile";
import "./globals.css";

const manrope = Manrope({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-manrope",
  display: "swap",
});
const sora = Sora({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-sora",
  display: "swap",
});
const dmMono = DM_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-dm-mono",
  display: "swap",
});

const ogImage = {
  url: withBasePath("/og.png"),
  width: 1200,
  height: 630,
  alt: `${profile.name} — Backend & Platform Engineer`,
};

export const metadata: Metadata = {
  metadataBase: new URL("https://fsmutimeer.github.io"),
  title: `${profile.name} — Backend & Platform Engineer`,
  description: `${profile.name} is a backend and platform engineer at IT22 B.V. in Islamabad, working with Java, Quarkus, Kafka, OpenShift, and OKD.`,
  robots: { index: true, follow: true },
  alternates: { canonical: "/" },
  openGraph: {
    title: `${profile.name} — Backend & Platform Engineer`,
    description:
      "Backend and platform engineer at IT22 B.V. Java, Quarkus, Kafka, OpenShift, and OKD. Islamabad.",
    type: "website",
    url: "/",
    siteName: profile.name,
    images: [ogImage],
  },
  twitter: {
    card: "summary_large_image",
    title: `${profile.name} — Backend & Platform Engineer`,
    description:
      "Backend and platform engineer at IT22 B.V. Java, Quarkus, Kafka, OpenShift, and OKD. Islamabad.",
    images: [ogImage.url],
  },
  icons: { icon: withBasePath("/favicon.svg") },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "Person",
  name: profile.name,
  jobTitle: "Backend & Platform Engineer",
  worksFor: {
    "@type": "Organization",
    name: "IT22 B.V.",
    url: "https://it22.nl/",
  },
  address: {
    "@type": "PostalAddress",
    addressLocality: "Islamabad",
    addressCountry: "PK",
  },
  description:
    "Backend and platform engineer at IT22 B.V. building Java and Quarkus services, Kafka event flows, and OpenShift / OKD platforms.",
  email: `mailto:${profile.email}`,
  telephone: profile.phoneHref.replace("tel:", ""),
  url: "https://fsmutimeer.github.io/",
  sameAs: [profile.githubUrl, profile.linkedinUrl],
  knowsAbout: [
    "Java",
    "Quarkus",
    "Apache Camel",
    "Kafka",
    "MongoDB",
    "Keycloak",
    "OpenShift",
    "OKD",
    "Kubernetes",
    "Tekton",
    "Argo CD",
    "Helm",
    "Wazuh",
    "Trivy",
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${manrope.variable} ${sora.variable} ${dmMono.variable}`}
    >
      <head>
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
