'use client';

import { useEffect, useState } from 'react';

export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(media.matches);
    update();
    media.addEventListener('change', update);
    sceneSync(media.matches);
    return () => media.removeEventListener('change', update);
  }, []);

  return reduced;
}

function sceneSync(reduced: boolean) {
  import('@/lib/scene-state').then(({ sceneState }) => {
    sceneState.set({ reducedMotion: reduced });
  });
}
