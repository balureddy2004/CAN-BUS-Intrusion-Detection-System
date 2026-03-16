export function Progress({ value = 0, className }: { value?: number; className?: string }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className={"w-full h-3 bg-gray-200 rounded " + (className ?? "")}>
      <div className="h-full bg-gray-900 rounded" style={{ width: `${pct}%` }} />
    </div>
  );
}
