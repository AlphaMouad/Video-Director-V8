import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: { serif: ['"Cormorant Garamond"', 'serif'], sans: ['Inter', 'sans-serif'] },
      colors: { gold: '#ca8a04' }
    },
  },
  plugins: [],
} satisfies Config
