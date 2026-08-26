'use client';

import { profile } from '@/lib/profile';
import { SplitTitle } from './split-title';

export function About() {
  return (
    <section className="section about" id="about" aria-labelledby="about-heading">
      <div className="container">
        <div className="about-grid">
          <div className="about-copy">
            <div className="section-label mono">01 / about</div>
            <SplitTitle
              id="about-heading"
              lines={['Engineer at the seam', 'of services and platforms.']}
            />
            <p className="section-intro">{profile.about.summary}</p>
            <p className="section-intro about-education mono">{profile.about.education}</p>
            <div className="about-stack" aria-label="Core technologies">
              {profile.about.stack.map((item) => (
                <span
                  className="pill pill-accent"
                  data-testid={`tag-${item.toLowerCase().replace(/\s/g, '-')}`}
                  key={item}
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
          <div className="about-side">
            <div className="about-focus" data-testid="list-about-focus">
              {profile.about.focus.map((item, index) => (
                <article
                  className="about-focus-item"
                  key={item.title}
                  data-testid={`card-about-focus-${index + 1}`}
                >
                  <span className="about-focus-no mono">{String(index + 1).padStart(2, '0')}</span>
                  <div>
                    <h3>{item.title}</h3>
                    <p>{item.copy}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
