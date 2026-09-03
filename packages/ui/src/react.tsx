import type { ButtonHTMLAttributes, MouseEvent, ReactNode } from "react";

export function cn(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function LoadingState({ message = "Loading indexed artifacts…" }: { message?: string }) {
  return (
    <div className="flex items-center justify-center gap-3 rounded-xl border border-line bg-panel px-6 py-12 text-sm text-muted shadow-[0_18px_50px_rgba(0,0,0,0.2)]">
      <span className="size-4 animate-spin rounded-full border-2 border-[#35514a] border-t-mint" aria-hidden="true" />
      {message}
    </div>
  );
}

export function EmptyState({ title, message, className }: { title?: string; message?: string; className?: string }) {
  return (
    <div className={cn("px-6 py-12 text-center text-sm text-muted", className)}>
      {title ? <strong className="block text-sm font-semibold text-copy">{title}</strong> : null}
      {message ? <p className="mt-2 mb-0 text-xs">{message}</p> : null}
    </div>
  );
}

export function ErrorCard({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-xl border border-line bg-panel px-6 py-12 text-center text-sm text-muted shadow-[0_18px_50px_rgba(0,0,0,0.2)]">
      <strong className="block text-sm font-semibold text-coral">Could not load this view</strong>
      <p className="mt-2 mb-0 text-xs">{message}</p>
    </div>
  );
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={cn("min-w-0 rounded-xl border border-line bg-panel shadow-[0_18px_50px_rgba(0,0,0,0.2)]", className)}>{children}</section>;
}

export function SectionHeading({ title, copy, action }: { title: string; copy?: string; action?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-line px-5 py-[18px]">
      <div>
        <h3 className="m-0 text-sm font-semibold text-copy">{title}</h3>
        {copy ? <p className="mt-1 mb-0 text-[11px] text-faint">{copy}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function LiveBadge({ children }: { children: ReactNode }) {
  return <span className="inline-flex items-center gap-2 text-[11px] text-muted"><span className="size-1.5 rounded-full bg-mint-strong shadow-[0_0_0_4px_rgba(66,198,149,0.1)]" aria-hidden="true" />{children}</span>;
}

export function StatusBadge({ value }: { value?: string | null }) {
  const normalized = String(value || "unknown").toLowerCase();
  const tone = ["running", "claimed", "starting", "checkpointing", "queued", "interrupted", "paused"].includes(normalized)
    ? "amber"
    : ["failed", "cancelled", "orphaned"].includes(normalized) ? "coral" : "mint";
  return <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2 py-1 text-[10px] capitalize", tone === "amber" && "bg-[#2b291d] text-amber", tone === "coral" && "bg-[#30201f] text-coral", tone === "mint" && "bg-[#1a2928] text-mint")}><span className="size-1.5 rounded-full bg-current" aria-hidden="true" />{normalized}</span>;
}

export function Button({ children, variant = "default", className, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "primary" | "quiet" | "danger" }) {
  return <button className={cn("inline-flex min-h-9 items-center justify-center rounded-lg border px-3 py-2 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-mint-strong disabled:cursor-wait disabled:opacity-55", variant === "default" && "border-line-strong bg-panel-raised text-copy hover:border-mint-strong hover:bg-[#1b2b2b]", variant === "primary" && "border-[#4f9e7f] bg-[#205440] text-[#e7fff2] hover:bg-[#28634d]", variant === "quiet" && "border-line-strong bg-transparent text-muted hover:border-mint-strong hover:bg-[#1b2b2b] hover:text-copy", variant === "danger" && "border-[#70433f] bg-[#2b1d1c] text-coral hover:bg-[#382321]", className)} {...props}>{children}</button>;
}

export function Field({ label, htmlFor, children, className }: { label: string; htmlFor?: string; children: ReactNode; className?: string }) {
  return <label className={cn("grid min-w-0 gap-1.5", className)} htmlFor={htmlFor}><span className="text-[10px] font-bold uppercase tracking-[0.08em] text-faint">{label}</span>{children}</label>;
}

export const inputClassName = "min-h-9 w-full rounded-lg border border-line-strong bg-[#0e1516] px-2.5 py-2 text-xs text-copy outline-none transition-colors placeholder:text-faint focus:border-mint-strong focus:ring-1 focus:ring-mint-strong";

export function DataTable({ headers, children, hasRows = true, emptyTitle = "Nothing indexed yet", emptyMessage = "Run an inventory refresh after producing a Rust artifact." }: { headers: string[]; children?: ReactNode; hasRows?: boolean; emptyTitle?: string; emptyMessage?: string }) {
  if (!hasRows) return <EmptyState title={emptyTitle} message={emptyMessage} />;
  return <div className="overflow-x-auto"><table className="w-full border-collapse text-xs"><thead><tr>{headers.map((header) => <th key={header} className="whitespace-nowrap px-5 py-3 text-left text-[10px] font-bold uppercase tracking-[0.08em] text-faint">{header}</th>)}</tr></thead><tbody className="[&_tr:hover]:bg-[#152021] [&_td]:border-t [&_td]:border-[#203031] [&_td]:px-5 [&_td]:py-3 [&_td]:align-middle [&_td]:text-muted [&_td:first-child]:font-medium [&_td:first-child]:text-copy">{children}</tbody></table></div>;
}

export function PageIntro({ kicker, title, description, action }: { kicker: string; title: string; description: string; action?: ReactNode }) {
  return <div className="mb-6 flex items-end justify-between gap-6 max-[720px]:items-start max-[720px]:flex-col"><div><p className="mb-1.5 text-[10px] font-extrabold uppercase tracking-[0.14em] text-mint">{kicker}</p><h2 className="m-0 text-[23px] font-semibold tracking-[-0.035em] text-copy">{title}</h2><p className="mt-2 mb-0 max-w-[570px] text-[13px] leading-relaxed text-muted">{description}</p></div>{action}</div>;
}

export function StatCard({ label, value, note, href, onClick }: { label: string; value: ReactNode; note?: string; href?: string; onClick?: (event: MouseEvent<HTMLAnchorElement>) => void }) {
  const content = <><span className="text-[11px] text-muted">{label}</span><strong className="mt-4 block text-[29px] font-semibold tracking-[-0.04em] text-copy">{value}</strong>{note ? <span className="mt-1 block truncate text-[10px] text-faint">{note}</span> : null}</>;
  if (href) return <a className="block min-h-[122px] rounded-xl border border-line bg-panel px-[18px] py-[17px] shadow-[0_18px_50px_rgba(0,0,0,0.2)] transition-colors hover:border-line-strong hover:bg-panel-raised" href={href} onClick={onClick}>{content}</a>;
  return <div className="min-h-[122px] rounded-xl border border-line bg-panel px-[18px] py-[17px] shadow-[0_18px_50px_rgba(0,0,0,0.2)]">{content}</div>;
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("font-mono text-[11px]", className)}>{children}</span>;
}
