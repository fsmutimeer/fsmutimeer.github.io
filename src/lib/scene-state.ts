export type SceneSection = 'hero' | 'about' | 'experience' | 'work' | 'approach' | 'contact';

type SceneSnapshot = {
  progress: number;
  section: SceneSection;
  workIndex: number;
  techIndex: number;
  mouseX: number;
  mouseY: number;
  ready: boolean;
  reducedMotion: boolean;
  isMobile: boolean;
  scrollVelocity: number;
};

const state: SceneSnapshot = {
  progress: 0,
  section: 'hero',
  workIndex: 0,
  techIndex: 0,
  mouseX: 0,
  mouseY: 0,
  ready: false,
  reducedMotion: false,
  isMobile: false,
  scrollVelocity: 0,
};

type Listener = (snapshot: SceneSnapshot) => void;
const listeners = new Set<Listener>();

function emit() {
  listeners.forEach((listener) => listener(state));
}

export const sceneState = {
  get(): SceneSnapshot {
    return state;
  },
  set(partial: Partial<SceneSnapshot>) {
    Object.assign(state, partial);
    emit();
  },
  subscribe(listener: Listener) {
    listeners.add(listener);
    listener(state);
    return () => {
      listeners.delete(listener);
    };
  },
};
