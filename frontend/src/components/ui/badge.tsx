import * as React from "react";
type V = "default" | "secondary" | "destructive" | "outline";
export function Badge({ variant="default", className, ...props }:{ variant?: V } & React.HTMLAttributes<HTMLSpanElement>) {
  const s: Record<V,string> = {
    default: "bg-gray-900 text-white",
    secondary: "bg-gray-200 text-gray-900",
    destructive: "bg-red-600 text-white",
    outline: "border border-gray-300"
  };
  return <span className={`inline-flex items-center rounded px-2 py-0.5 text-xs ${s[variant]} ${className ?? ""}`} {...props} />;
}
