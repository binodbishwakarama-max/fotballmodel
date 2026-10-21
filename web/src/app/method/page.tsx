import { PageHeading } from "@/components/page-heading";

const principles = [
  {
    title: "Forecast before the event",
    copy: "Published predictions are tied to a recorded cutoff and kickoff. The public view does not invent or backfill missing forecasts.",
  },
  {
    title: "Compare like with like",
    copy: "A model and its baselines should be measured on the same eligible fixtures. Coverage and missing observations matter alongside each score.",
  },
  {
    title: "Probability-first evaluation",
    copy: "Proper scoring rules evaluate the full home/draw/away distribution. Accuracy is a secondary summary and does not replace probability quality.",
  },
  {
    title: "Chronological validation",
    copy: "Walk-forward evaluation respects time: each test period follows the training data, and calibration is kept within the historical training period.",
  },
  {
    title: "An auditable record",
    copy: "Forecasts are intended to remain immutable. Corrections are identified separately, while batch provenance and verification status belong to the underlying publication record.",
  },
];

export default function MethodPage() {
  return (
    <>
      <PageHeading
        eyebrow="How to read the board"
        title="Method, in plain language."
        description="The scoreboard is built to make forecasts and measured outcomes easier to inspect. These principles describe how to interpret the record."
      />
      <section className="grid gap-3 sm:grid-cols-2" aria-label="Evaluation principles">
        {principles.map((principle) => (
          <article key={principle.title} className="rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-5 sm:p-6">
            <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">Principle</p>
            <h2 className="text-base font-semibold">{principle.title}</h2>
            <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{principle.copy}</p>
          </article>
        ))}
      </section>
    </>
  );
}
