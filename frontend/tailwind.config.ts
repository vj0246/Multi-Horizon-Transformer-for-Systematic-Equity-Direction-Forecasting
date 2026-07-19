import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#070a11",
        panel: "#0e131d",
        panel2: "#141b28",
        edge: "#1e2839",
        edge2: "#2c3a51",
        accent: "#2dd4bf",
        accent2: "#a78bfa",
        danger: "#fb7185",
        warn: "#fbbf24",
        muted: "#94a3b8",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
