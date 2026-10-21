"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Upcoming" },
  { href: "/track-record", label: "Track record" },
  { href: "/calibration", label: "Calibration" },
  { href: "/method", label: "Method" },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="border-b border-[var(--line)]">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-4 sm:px-8 md:flex-row md:items-center md:justify-between">
        <Link href="/" className="flex items-center gap-3 self-start" aria-label="BWB Forecast home">
          <span className="grid size-9 place-items-center rounded-xl bg-[var(--accent)] text-sm font-black text-[#14200d]">B</span>
          <span>
            <span className="block text-sm font-bold tracking-wide">BWB FORECAST</span>
            <span className="block text-[10px] uppercase tracking-[0.2em] text-[var(--muted)]">Public scoreboard</span>
          </span>
        </Link>
        <nav aria-label="Main navigation" className="-mx-1 flex gap-1 overflow-x-auto pb-1 md:mx-0 md:pb-0">
          {links.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={`shrink-0 rounded-full px-3 py-2 text-sm transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)] ${
                  active
                    ? "bg-[var(--surface-raised)] text-[var(--accent)]"
                    : "text-[var(--muted)] hover:bg-[var(--surface)] hover:text-[var(--foreground)]"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
