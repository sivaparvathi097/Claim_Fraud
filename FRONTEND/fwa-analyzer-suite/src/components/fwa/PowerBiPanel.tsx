import { useEffect, useState, type ReactNode } from "react";
import { BarChart3, ExternalLink, Loader2 } from "lucide-react";
import { Panel } from "./primitives";

/**
 * Power BI embed container. When a report URL is configured
 * (VITE_POWERBI_EMBED_URL) the report is embedded natively in an iframe with a
 * polished loading state. Filter context for the selected claim/provider is
 * appended as a report-level filter. Without a configured workspace, the panel
 * renders the in-app analytics fallback passed as `children`.
 */
export function PowerBiPanel({
  title = "Power BI Dashboard",
  description,
  filterTable,
  filterColumn,
  filterValue,
  children,
}: {
  title?: string;
  description?: string;
  filterTable?: string;
  filterColumn?: string;
  filterValue?: string;
  children: ReactNode;
}) {
  const base = import.meta.env['VITE_POWERBI_EMBED_URL'] as string | undefined;
  const [loading, setLoading] = useState(Boolean(base));

  const src = base
    ? `${base}${base.includes("?") ? "&" : "?"}${
        filterTable && filterColumn && filterValue
          ? `filter=${encodeURIComponent(`${filterTable}/${filterColumn} eq '${filterValue}'`)}&`
          : ""
      }navContentPaneEnabled=false`
    : undefined;

  useEffect(() => {
    setLoading(Boolean(src));
  }, [src]);

  return (
    <Panel
      title={title}
      description={description ?? (filterValue ? `Filtered context: ${filterValue}` : "Embedded analytics workspace")}
      icon={BarChart3}
      bodyClassName="p-0"
      actions={
        <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-2.5 py-1 text-[11px] font-medium text-muted-foreground">
          <ExternalLink className="size-3.5" />
          {src ? "Live embed" : "In-app analytics"}
        </span>
      }
    >
      {src ? (
        <div className="relative aspect-16/9 w-full">
          {loading ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-secondary/50">
              <Loader2 className="size-6 animate-spin text-primary" />
              <p className="text-xs text-muted-foreground">Loading Power BI report…</p>
            </div>
          ) : null}
          <iframe
            title={title}
            src={src}
            className="size-full border-0"
            allowFullScreen
            onLoad={() => setLoading(false)}
          />
        </div>
      ) : (
        <div className="animate-fade p-5">{children}</div>
      )}
    </Panel>
  );
}
