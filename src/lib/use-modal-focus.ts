import { useEffect, useRef, type RefObject } from 'react';

const focusableSelector = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

export function useModalFocus(
  dialogRef: RefObject<HTMLElement | null>,
  active: boolean,
  onEscape?: () => void,
) {
  const onEscapeRef = useRef(onEscape);
  onEscapeRef.current = onEscape;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!active || !dialog) return;

    const previouslyFocused =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const inertStates = new Map<HTMLElement, boolean>();
    let branch = dialog;

    while (branch.parentElement) {
      const parent = branch.parentElement;
      for (const child of Array.from(parent.children)) {
        if (child instanceof HTMLElement && child !== branch) {
          inertStates.set(child, child.inert);
          child.inert = true;
        }
      }
      if (parent === document.body) break;
      branch = parent;
    }

    const focusableElements = () =>
      Array.from(dialog.querySelectorAll<HTMLElement>(focusableSelector)).filter(
        (element) => element.tabIndex >= 0 && !element.closest('[hidden], [inert]'),
      );

    const focusInitial = () => {
      const elements = focusableElements();
      const initial = dialog.querySelector<HTMLElement>(
        '[data-modal-autofocus]',
      );
      (initial ?? elements[0] ?? dialog).focus();
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onEscapeRef.current?.();
        return;
      }
      if (event.key !== 'Tab') return;

      const elements = focusableElements();
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (!first || !last) {
        event.preventDefault();
        dialog.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    const onFocusIn = (event: FocusEvent) => {
      if (event.target instanceof Node && !dialog.contains(event.target)) {
        focusInitial();
      }
    };

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('focusin', onFocusIn);
    focusInitial();

    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('focusin', onFocusIn);
      for (const [element, wasInert] of inertStates) element.inert = wasInert;
      if (previouslyFocused?.isConnected) previouslyFocused.focus();
    };
  }, [active, dialogRef]);
}