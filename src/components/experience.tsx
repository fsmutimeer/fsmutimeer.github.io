'use client';

import { experience } from '@/lib/content';
import { SplitTitle } from './split-title';

export function Experience() {
  return (
    <section className="section experience" id="experience" aria-labelledby="experience-heading">
      <div className="container">
        <div className="experience-head">
          <div>
            <div className="section-label mono">02 / experience</div>
            <SplitTitle id="experience-heading" lines={['Two jobs,', 'both in Islamabad.']} />
          </div>
          <p className="section-intro">
            Current employer is IT22 B.V. The previous employer is ESOLS Technologies. Dates match
            the CV.
          </p>
        </div>
        <ol className="experience-list" data-testid="list-experience">
          {experience.map((job) => (
            <li className="experience-item" key={job.id} data-testid={`card-experience-${job.id}`}>
              <div className="experience-when mono">
                <span>{job.dates}</span>
                <span>{job.location}</span>
              </div>
              <div className="experience-body">
                <p className="experience-role">{job.role}</p>
                <h3>
                  <a
                    href={job.companyUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    data-cursor="hover"
                    data-cursor-label="open"
                  >
                    {job.company}
                  </a>
                </h3>
                <ul>
                  {job.bullets.map((bullet) => (
                    <li key={bullet}>{bullet}</li>
                  ))}
                </ul>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
