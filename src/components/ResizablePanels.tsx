import {
  useEffect,
  useRef,
  useState,
  type ReactNode,
  type PointerEvent,
  type KeyboardEvent,
} from "react";

const STORAGE_KEY = "vizml.layout.split.v1";
const DEFAULT_SPLIT = 0.38;

export default function ResizablePanels({
  architecture,
  children,
  expanded,
}: {
  architecture: ReactNode;
  children: ReactNode;
  expanded: boolean;
}) {
  const container = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [isDragging, setIsDragging] = useState(false);
  const [landscape, setLandscape] = useState(
    () => window.innerWidth > window.innerHeight,
  );
  const [split, setSplit] = useState(() => {
    try {
      const stored = Number(localStorage.getItem(STORAGE_KEY));
      if (stored >= 0.18 && stored <= 0.82) return stored;
    } catch {
      /* Layout remains usable without browser storage. */
    }
    return DEFAULT_SPLIT;
  });
  const splitValue = useRef(split);
  const landscapeValue = useRef(landscape);
  landscapeValue.current = landscape;

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    // The library and inspector can leave a narrow, tall workspace even on a
    // landscape monitor. Stack the views there so the long architecture gets
    // the full available width; place them side by side only when the actual
    // editor has room for both.
    const updateOrientation = () => {
      const { width, height } = element.getBoundingClientRect();
      if (width && height) setLandscape(width > height * 1.15);
    };
    const observer = new ResizeObserver(updateOrientation);
    observer.observe(element);
    updateOrientation();
    return () => observer.disconnect();
  }, []);

  const update = (value: number, persist = false) => {
    const scale = landscapeValue.current
      ? (container.current?.clientWidth ?? 1000) - 9
      : (container.current?.clientHeight ?? 600) - 9;
    const lower = Math.max(
      0.18,
      Math.min((landscapeValue.current ? 280 : 120) / scale, 0.45),
    );
    const upper = Math.min(
      0.82,
      1 - Math.min((landscapeValue.current ? 300 : 160) / scale, 0.45),
    );
    const next = Math.max(lower, Math.min(upper, value));
    splitValue.current = next;
    setSplit(next);
    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, String(next));
      } catch {
        /* Keep the current session layout. */
      }
    }
  };
  useEffect(() => {
    update(splitValue.current);
  }, [landscape]);

  const move = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragging.current || !container.current) return;
    const rect = container.current.getBoundingClientRect();
    update(
      landscape
        ? (event.clientX - rect.left - 4.5) / (rect.width - 9)
        : (event.clientY - rect.top - 4.5) / (rect.height - 9),
    );
  };
  const finish = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return;
    dragging.current = false;
    setIsDragging(false);
    update(splitValue.current, true);
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };
  const keyboard = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 0.1 : 0.025;
    if (landscape) {
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        update(
          splitValue.current + (event.key === "ArrowRight" ? step : -step),
          true,
        );
      }
    } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      update(
        splitValue.current + (event.key === "ArrowUp" ? -step : step),
        true,
      );
    }
    if (event.key === "Home" || event.key === "Enter") {
      event.preventDefault();
      update(DEFAULT_SPLIT, true);
    }
  };

  return (
    <div
      className={`editor-stack resizable-editor ${landscape ? "horizontal" : ""} ${isDragging ? "is-resizing" : ""}`}
      ref={container}
    >
      <div
        className="architecture-slot"
        style={{
          flex: expanded
            ? "1 1 0%"
            : landscape
              ? `${1 - split} 1 0%`
              : `${split} 1 0%`,
        }}
      >
        {architecture}
      </div>
      <div
        className="panel-splitter"
        role="separator"
        aria-label="Resize architecture and model graph"
        aria-orientation={landscape ? "vertical" : "horizontal"}
        aria-valuemin={18}
        aria-valuemax={82}
        aria-valuenow={Math.round(split * 100)}
        aria-valuetext={
          landscape
            ? `Model graph ${Math.round(split * 100)}%, architecture ${Math.round((1 - split) * 100)}%`
            : `Architecture ${Math.round(split * 100)}%, model graph ${Math.round((1 - split) * 100)}%`
        }
        tabIndex={expanded ? -1 : 0}
        style={{ display: expanded ? "none" : undefined }}
        title="Drag to resize · arrow keys to adjust · double-click to reset"
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.preventDefault();
          dragging.current = true;
          setIsDragging(true);
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={move}
        onPointerUp={finish}
        onPointerCancel={finish}
        onLostPointerCapture={finish}
        onKeyDown={keyboard}
        onDoubleClick={() => update(DEFAULT_SPLIT, true)}
      >
        <span />
        <span />
        <span />
      </div>
      <div
        className="graph-slot"
        style={{
          flex: landscape ? `${split} 1 0%` : `${1 - split} 1 0%`,
          display: expanded ? "none" : undefined,
        }}
      >
        {children}
      </div>
    </div>
  );
}
