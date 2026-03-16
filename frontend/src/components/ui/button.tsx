import * as React from "react";

type Variant = "default" | "secondary" | "destructive" | "outline" | "ghost";
type Size = "sm" | "md" | "lg";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const variantClasses: Record<Variant, string> = {
  default: "bg-gray-900 text-white hover:bg-gray-800",
  secondary: "bg-gray-200 text-gray-900 hover:bg-gray-300",
  destructive: "bg-red-600 text-white hover:bg-red-700",
  outline: "bg-transparent border border-gray-300 hover:bg-gray-100",
  ghost: "bg-transparent hover:bg-gray-100",
};

const sizeClasses: Record<Size, string> = {
  sm: "h-8 px-3 text-sm",
  md: "h-9 px-3 text-sm",
  lg: "h-10 px-4 text-base",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = "", variant = "default", size = "md", type = "button", disabled, ...props }, ref) => {
    const base =
      "inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none " +
      "focus:ring-2 focus:ring-offset-2 focus:ring-gray-400 disabled:opacity-50 disabled:pointer-events-none";
    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled}
        className={`${base} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";
