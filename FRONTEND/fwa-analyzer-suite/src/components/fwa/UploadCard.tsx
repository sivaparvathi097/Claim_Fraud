import { useRef, useState } from "react";
import { CheckCircle2, FileSpreadsheet, Loader2, UploadCloud, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

export type UploadedFile = {
  name: string;
  records: number | null;
  sizeKb: number;
  file: File;
  valid: boolean;
  error: string | null;
};

export function UploadCard({
  label,
  hint,
  onUploaded,
  file,
  validate,
}: {
  label: string;
  hint: string;
  onUploaded: (file: UploadedFile) => void;
  file: UploadedFile | null;
  /** Backend schema preflight — returns the real record count. */
  validate: (file: File) => Promise<{ record_count: number }>;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [checking, setChecking] = useState(false);

  function handleFile(picked: File | undefined) {
    if (!picked || checking) return;
    setChecking(true);
    const base = {
      name: picked.name,
      sizeKb: Math.max(1, Math.round(picked.size / 1024)),
      file: picked,
    };
    // The ACTUAL selected file is sent to the backend preflight endpoint —
    // the record count and the schema verdict come from the backend model
    // contract, never invented client-side.
    validate(picked)
      .then((res) => onUploaded({ ...base, records: res.record_count, valid: true, error: null }))
      .catch((err) =>
        onUploaded({
          ...base,
          records: null,
          valid: false,
          error: err instanceof Error ? err.message : String(err),
        }),
      )
      .finally(() => setChecking(false));
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFile(e.dataTransfer.files[0]);
      }}
      className={`surface-panel animate-rise flex flex-col items-center gap-5 px-6 py-12 text-center transition-all duration-400 ${
        dragging ? "border-accent/60 shadow-lift" : ""
      }`}
    >
      <span className="gradient-navy flex size-16 items-center justify-center rounded-2xl text-navy-foreground shadow-float">
        <UploadCloud className="size-8" />
      </span>
      <div>
        <h3 className="text-base font-semibold tracking-[0.08em] uppercase text-foreground">{label}</h3>
        <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">{hint}</p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx,.xls,.json,.txt"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      <Button
        onClick={() => inputRef.current?.click()}
        disabled={checking}
        className="rounded-full px-7 py-5 text-sm font-semibold shadow-float transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lift"
      >
        {checking ? <Loader2 className="size-4 animate-spin" /> : null}
        {checking ? "Validating on backend…" : label}
      </Button>

      {file ? (
        <div className="animate-rise mt-2 grid w-full max-w-2xl gap-3 rounded-xl border border-border/70 bg-secondary/40 p-4 text-left sm:grid-cols-3">
          <Detail icon label="File name" value={file.name} />
          <Detail label="Records detected" value={file.records === null ? "—" : file.records.toLocaleString()} />
          {file.valid ? (
            <Detail label="Upload status" value="Validated · Ready" ok />
          ) : (
            <Detail label="Upload status" value={file.error ?? "Invalid file"} bad wrap />
          )}
        </div>
      ) : null}
    </div>
  );
}

function Detail({
  label,
  value,
  icon,
  ok,
  bad,
  wrap,
}: {
  label: string;
  value: string;
  icon?: boolean;
  ok?: boolean;
  bad?: boolean;
  wrap?: boolean;
}) {
  return (
    <div className="min-w-0">
      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{label}</p>
      <p
        className={`mt-1 flex items-start gap-1.5 text-sm font-medium ${
          bad ? "text-risk-high" : "text-foreground"
        } ${wrap ? "" : "truncate"}`}
      >
        {icon ? <FileSpreadsheet className="size-4 shrink-0 text-primary" /> : null}
        {ok ? <CheckCircle2 className="size-4 shrink-0 text-risk-low" /> : null}
        {bad ? <XCircle className="size-4 shrink-0 text-risk-high" /> : null}
        <span className={wrap ? "break-words" : "truncate"}>{value}</span>
      </p>
    </div>
  );
}
