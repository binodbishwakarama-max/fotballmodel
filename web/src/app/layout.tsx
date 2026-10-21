import type { Metadata } from "next";
import "./globals.css";
import { SiteHeader } from "@/components/site-header";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: "BWB Forecast — Public Scoreboard",
  description:
    "A transparent, public record of football forecasts and their measured performance.",
  openGraph: {
    title: "BWB Forecast — Public Scoreboard",
    description:
      "A transparent, public record of football forecasts and their measured performance.",
    siteName: "BWB Forecast",
    type: "website",
    images: [{ url: "/opengraph-image", width: 1200, height: 630, alt: "BWB Forecast public scoreboard" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "BWB Forecast — Public Scoreboard",
    description:
      "A transparent, public record of football forecasts and their measured performance.",
    images: ["/opengraph-image"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <SiteHeader />
        <main className="mx-auto w-full max-w-7xl px-5 pb-20 pt-9 sm:px-8 sm:pt-14">
          {children}
        </main>
        <footer className="mx-auto flex max-w-7xl flex-col gap-2 border-t border-[var(--line)] px-5 py-7 text-xs text-[var(--muted)] sm:flex-row sm:items-center sm:justify-between sm:px-8">
          <span>BWB Forecast · Public record</span>
          <span>Published forecasts are presented as returned by the API.</span>
        </footer>
      </body>
    </html>
  );
}
