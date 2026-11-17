// Shared helpers for the model pages.
const C = {
  navy: "#1f4e79", blue: "#3a78b8", light: "#7ea6cf", pale: "#e9eff5",
  ink: "#1a1a1a", muted: "#5f6770", grid: "#e3e7eb", free: "#b8c0c8", obs: "#8a939c",
  accent: "#e8590c",
};
const FONT = "12px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";

// Size a canvas for the screen and return a context in CSS pixels.
function fitCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = FONT;
  return { ctx, w, h };
}

// Redraw at most once per frame, however fast a control moves.
function throttled(draw) {
  let waiting = false;
  return () => {
    if (waiting) return;
    waiting = true;
    requestAnimationFrame(() => { waiting = false; draw(); });
  };
}

function decodeU16(b64) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Uint16Array(bytes.buffer);
}
