/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",  // scan React components
  ],
  theme: {
    extend: {}, // you can add custom colors, fonts etc later
  },
  plugins: [],
}
