import gsap from 'gsap';

const GLYPHS = '01#%$<>/\\*ABCDEF';
const running = new WeakMap<HTMLElement, gsap.core.Tween>();

function reducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function scrambleTo(el: HTMLElement, finalText: string, duration = 0.55) {
  running.get(el)?.kill();

  if (reducedMotion()) {
    el.textContent = finalText;
    return () => undefined;
  }

  const proxy = { t: 0 };
  const tween = gsap.to(proxy, {
    t: 1,
    duration,
    ease: 'power2.out',
    onUpdate: () => {
      const locked = Math.floor(finalText.length * proxy.t);
      let next = finalText.slice(0, locked);
      for (let i = locked; i < finalText.length; i += 1) {
        next += finalText[i] === ' ' ? ' ' : GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
      }
      el.textContent = next;
    },
    onComplete: () => {
      el.textContent = finalText;
      running.delete(el);
    },
  });
  running.set(el, tween);

  return () => {
    tween.kill();
    running.delete(el);
    el.textContent = finalText;
  };
}
