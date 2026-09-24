import type { Metadata } from 'next';
import { AboutStory } from '@/components/about-story';
import { profile } from '@/lib/profile';

export const metadata: Metadata = {
  title: `About Me — ${profile.name}`,
  description:
    'Feroz Shah on growing up in the Kalash valleys, finding a way into software, and working toward a return home.',
  alternates: { canonical: '/about/' },
  openGraph: {
    title: `About Me — ${profile.name}`,
    description:
      'A longer note on Kalash, photography, IT, and the mountains still called home.',
    type: 'article',
    url: '/about/',
  },
};

export default function AboutPage() {
  return <AboutStory />;
}
