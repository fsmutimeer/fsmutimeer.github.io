import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { CaseStudy } from '@/components/case-study';
import { getProjectBySlug, projects } from '@/lib/content';
import { profile } from '@/lib/profile';

export function generateStaticParams() {
  return projects.map((project) => ({ slug: project.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const project = getProjectBySlug(slug);
  if (!project) return { title: profile.name };
  return {
    title: `${project.title} — ${profile.name}`,
    description: project.copy,
    openGraph: {
      title: `${project.title} — ${profile.name}`,
      description: project.copy,
      type: 'article',
    },
  };
}

export default async function CaseStudyPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const project = getProjectBySlug(slug);
  if (!project) notFound();
  return <CaseStudy project={project} />;
}
