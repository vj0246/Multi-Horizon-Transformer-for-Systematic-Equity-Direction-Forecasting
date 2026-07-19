import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Multi-Horizon Transformer · Nifty 50 Direction Forecasting",
  description:
    "Multi-horizon Transformer for Nifty 50 direction, with rigorous evidence it has no edge.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-mono antialiased">{children}</body>
    </html>
  );
}
