"use client";

import { useEffect, useState } from "react";
import gsap from "gsap";

export function Preloader({ onDone }: { onDone: () => void }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const state = { value: 0 };
    const tween = gsap.to(state, {
      value: 100,
      duration: 1.85,
      ease: "power2.inOut",
      onUpdate: () => setProgress(state.value),
      onComplete: () => {
        gsap.to(".preloader", {
          clipPath: "inset(0% 0% 100% 0%)",
          duration: 0.95,
          ease: "power4.inOut",
          delay: 0.12,
          onComplete: onDone,
        });
      },
    });
    return () => {
      tween.kill();
    };
  }, [onDone]);

  return (
    <div className="preloader" aria-hidden="true">
      <div className="preloader-inner">
        <span className="mono preloader-label">loading ...</span>
        <strong className="preloader-count">
          {progress.toFixed(1).padStart(5, "0")}
        </strong>
        <div className="preloader-bar">
          <span style={{ width: `${progress}%` }} />
        </div>
      </div>
    </div>
  );
}
