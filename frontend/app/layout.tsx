import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Multi-Horizon Transformer · Nifty 50 Direction Forecasting",
  description:
    "A multi-output Transformer predicting Nifty 50 directional movement across 20 forward horizons. Real backtested results, no look-ahead leakage.",
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
