import { aboutStory } from '@/lib/about';
import { profile } from '@/lib/profile';
import { withBasePath } from '@/lib/base-path';
import { SceneFallback } from './scene/fallback';

function PhotoPlaceholder({
  slot,
  caption,
  src,
}: {
  slot: string;
  caption: string;
  src?: string;
}) {
  return (
    <figure className="story-figure">
      {src ? (
        <img className="story-photo" src={withBasePath(src)} alt={caption} />
      ) : (
        <div
          className="story-photo-slot"
          data-photo-slot={slot}
          role="img"
          aria-label={`${caption}. Placeholder until the photo is added.`}
        >
          <span className="story-photo-mark mono">photo placeholder</span>
          <span className="story-photo-hint mono">public/about/{slot}.jpg</span>
        </div>
      )}
      <figcaption className="mono">{caption}</figcaption>
    </figure>
  );
}

export function AboutStory() {
  const home = withBasePath('/');
  const work = withBasePath('/#work');

  return (
    <main className="portfolio-shell">
      <SceneFallback />
      <div className="grain" aria-hidden="true" />
      <article className="study story">
        <header className="study-bar">
          <div className="container study-bar-inner">
            <a className="brand" href={home}>
              <span className="brand-mark">{profile.initials}</span>
              <span>
                {profile.name}
                <span className="brand-dot">.</span>
              </span>
            </a>
            <nav className="study-bar-links" aria-label="About page">
              <a href={home}>Home</a>
              <a href={work}>Selected work</a>
              <a href={profile.cvUrl} target="_blank" rel="noopener noreferrer">
                CV
              </a>
            </nav>
          </div>
        </header>

        <div className="container study-body">
          <p className="eyebrow mono">{aboutStory.eyebrow}</p>
          <h1 className="study-title">{aboutStory.title}</h1>
          <p className="study-lede">{aboutStory.lede}</p>

          {aboutStory.sections.map((section, index) => (
            <section
              className="story-section"
              id={section.id}
              key={section.id}
              aria-labelledby={`${section.id}-heading`}
              data-align={index % 2 === 0 ? 'image-left' : 'image-right'}
            >
              <PhotoPlaceholder
                slot={section.photoSlot}
                caption={section.photoCaption}
                src={section.photoSrc}
              />
              <div className="story-copy">
                <span className="story-no mono">{section.number}</span>
                <h2 id={`${section.id}-heading`}>{section.title}</h2>
                {section.paragraphs.map((paragraph) => (
                  <p key={paragraph.slice(0, 40)}>{paragraph}</p>
                ))}
              </div>
            </section>
          ))}

          <p className="story-signoff">{aboutStory.signoff}</p>

          <div className="study-footer">
            <a className="text-link" href={home}>
              Back to the portfolio
            </a>
            <a className="text-link" href={work}>
              Selected work
            </a>
          </div>
        </div>
      </article>
    </main>
  );
}
