import * as React from "react";
type Props = { checked?: boolean; onCheckedChange?: (v: boolean) => void } & Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "onChange">;
export function Switch({ checked, onCheckedChange, className, ...rest }: Props) {
  return (
    <button
      type="button"
      aria-pressed={checked}
      onClick={() => onCheckedChange?.(!checked)}
      className={`h-5 w-10 rounded-full border px-0.5 transition-colors ${checked ? "bg-green-500" : "bg-gray-200"} ${className ?? ""}`}
      {...rest}
    >
      <span className={`block h-4 w-4 rounded-full bg-white transition-transform ${checked ? "translate-x-5" : ""}`} />
    </button>
  );
}
