import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#0a0e14",
        panel: "#111722",
        panel2: "#161d2b",
        edge: "#232c3d",
        accent: "#4ade80",
        accent2: "#38bdf8",
        danger: "#f87171",
        muted: "#8b98ad",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
