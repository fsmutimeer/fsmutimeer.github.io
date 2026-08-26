'use client';

export function SplitTitle({
  id,
  className = 'section-title',
  lines,
}: {
  id?: string;
  className?: string;
  lines: readonly string[];
}) {
  return (
    <h2 className={className} id={id}>
      {lines.map((line) => (
        <span className="line-mask" key={line}>
          <span className="line">{line}</span>
        </span>
      ))}
    </h2>
  );
}
