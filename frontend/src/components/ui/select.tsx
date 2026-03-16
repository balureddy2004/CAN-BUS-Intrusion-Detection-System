import * as React from "react";
export function Select({ value, onValueChange, children }:{ value?: string; onValueChange?: (v:string)=>void; children: React.ReactNode }) {
  return <div data-select>{React.Children.map(children, c => React.isValidElement(c) ? React.cloneElement(c as any, { selectedValue:value, onValueChange }) : c)}</div>;
}
export function SelectTrigger({ children }:{ children: React.ReactNode }) { return <div className="border rounded px-3 py-2">{children}</div>; }
export function SelectValue({ placeholder }:{ placeholder?: string }) { return <span className="text-sm">{placeholder}</span>; }
export function SelectContent({ children }:{ children: React.ReactNode }) { return <div className="mt-1 border rounded overflow-hidden">{children}</div>; }
export function SelectItem({ value, children, onValueChange }:{ value: string; children: React.ReactNode; onValueChange?: (v:string)=>void }) {
  return <button type="button" onClick={() => onValueChange?.(value)} className="block w-full text-left px-3 py-1 text-sm hover:bg-gray-100">{children}</button>;
}
