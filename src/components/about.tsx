"use client";

import Link from "next/link";
import Image from "next/image";
import { profile } from "@/lib/profile";
import { withBasePath } from "@/lib/base-path";
import { SplitTitle } from "./split-title";

export function About() {
  return (
    <section
      className="section about"
      id="about"
      aria-labelledby="about-heading"
    >
      <div className="container">
        <div className="about-grid">
          <div className="about-copy">
            <div className="section-label mono">01 / about</div>
            <div className="about-profile-wrap">
              <div className="about-photo-card">
                <div className="about-photo-frame">
                  <Image
                    src={withBasePath("/feroz.jpeg")}
                    alt={profile.name}
                    width={200}
                    height={260}
                    className="about-portrait-img"
                    priority
                  />
                  <div className="about-photo-badge mono">
                    <span className="status-dot" />
                    <span>Islamabad, PK</span>
                  </div>
                </div>
              </div>
              <div className="about-narrative">
                <SplitTitle
                  id="about-heading"
                  lines={["From Kalash to", "the Cluster."]}
                />
                {profile.about.summary.map((paragraph) => (
                  <p className="about-narrative-p" key={paragraph.slice(0, 48)}>
                    {paragraph}
                  </p>
                ))}
                <p className="about-education mono">
                  {profile.about.education}
                </p>
                <p className="about-story-cta">
                  <Link
                    className="text-link"
                    href={withBasePath("/about/")}
                    data-testid="link-about-story"
                    data-cursor="hover"
                    data-cursor-label="story"
                  >
                    The longer story →
                  </Link>
                </p>
              </div>
            </div>
          </div>
          <div className="about-side">
            <div className="about-capabilities" data-testid="list-about-capabilities" aria-label="Capabilities">
              <div className="about-capabilities-header mono">Tools &amp; Stack</div>
              {profile.about.capabilities.map((group) => (
                <article
                  className="about-cap-row"
                  key={group.category}
                  data-testid={`cap-${group.category.toLowerCase()}`}
                >
                  <span className="about-cap-label mono">{group.category}</span>
                  <div className="about-cap-body">
                    <span className="about-cap-tools">
                      {group.tools.join(" · ")}
                    </span>
                    <span className="about-cap-copy">{group.copy}</span>
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
