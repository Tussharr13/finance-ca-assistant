"use client";

const SUGGESTIONS = [
  {
    title: "Old or new regime?",
    detail: "I earn 18 LPA in Pune and pay ₹25,000 rent a month.",
    prompt: "Which tax regime should I choose? I earn 18 LPA in Pune and pay rent of 25k per month.",
  },
  {
    title: "Plan my salary tax",
    detail: "Find legal deductions and documents to keep.",
    prompt: "I am earning salary and want to save tax legally. Act like my CA.",
  },
  {
    title: "SA 500 audit evidence",
    detail: "What makes evidence sufficient and appropriate?",
    prompt: "What does SA 500 say about audit evidence?",
  },
  {
    title: "Form 3CD clause 44",
    detail: "What must the tax auditor report?",
    prompt: "What does clause 44 of Form 3CD require?",
  },
];

export default function EmptyState({ onPick, disabled }: { onPick: (prompt: string) => void; disabled: boolean }) {
  return (
    <div className="empty">
      <div>
        <h1>What can I help you work out?</h1>
        <p>
          Ask about income tax, GST, audit or accounting standards. Tax figures come from a calculator that shows
          each step and the section behind it; answers about the law quote their source and page.
        </p>
      </div>
      <div className="suggestions">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion.title}
            type="button"
            className="suggestion"
            onClick={() => onPick(suggestion.prompt)}
            disabled={disabled}
          >
            <strong>{suggestion.title}</strong>
            <span>{suggestion.detail}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
