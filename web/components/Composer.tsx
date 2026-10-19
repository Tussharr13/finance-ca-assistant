"use client";

import { useEffect, useRef, useState } from "react";
import { SendIcon } from "./icons";

type Props = {
  disabled: boolean;
  onSend: (text: string) => void;
  autoFocusKey?: string | null;
};

const MAX_CHARS = 4000;

export default function Composer({ disabled, onSend, autoFocusKey }: Props) {
  const [text, setText] = useState("");
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    textarea.current?.focus();
  }, [autoFocusKey]);

  useEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [text]);

  function submit() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  }

  return (
    <div className="composer-wrap">
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <label htmlFor="composer-input" className="sr-only">
          Message
        </label>
        <textarea
          id="composer-input"
          ref={textarea}
          rows={1}
          value={text}
          maxLength={MAX_CHARS}
          placeholder="Ask about income tax, GST, audit or accounting…"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <button type="submit" className="send" disabled={disabled || !text.trim()} aria-label="Send message">
          <SendIcon />
        </button>
      </form>
      <p className="disclaimer">Calculations and source summaries for decision support, not filing advice. Confirm with a CA.</p>
    </div>
  );
}
