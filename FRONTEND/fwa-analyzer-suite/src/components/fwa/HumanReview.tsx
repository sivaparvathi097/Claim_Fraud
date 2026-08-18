import { Check, RotateCcw, UserCheck, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Panel, StatusBadge } from "./primitives";
import type { ReviewStatus } from "@/lib/fwa";

export function HumanReview({
  status,
  onChange,
  subject,
}: {
  status: ReviewStatus;
  onChange: (status: ReviewStatus) => void;
  subject: string;
}) {
  return (
    <Panel title="Human Review" description={`Final determination for ${subject}`} icon={UserCheck}>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs text-muted-foreground">Current status</p>
          <div className="mt-2">
            <StatusBadge status={status} />
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            onClick={() => onChange("Accepted")}
            className="rounded-full bg-risk-low px-5 text-primary-foreground transition-all duration-300 hover:-translate-y-0.5 hover:bg-risk-low/90 hover:shadow-float"
          >
            <Check className="size-4" /> Accept
          </Button>
          <Button
            onClick={() => onChange("Rejected")}
            className="rounded-full bg-risk-high px-5 text-primary-foreground transition-all duration-300 hover:-translate-y-0.5 hover:bg-risk-high/90 hover:shadow-float"
          >
            <X className="size-4" /> Reject
          </Button>
          <Button
            variant="outline"
            onClick={() => onChange("Resubmission Required")}
            className="rounded-full px-5 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-float"
          >
            <RotateCcw className="size-4" /> Request resubmission
          </Button>
        </div>
      </div>
      <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
        The AI system provides anomaly detection and explainability. The domain expert performs the final
        verification and records the outcome above.
      </p>
    </Panel>
  );
}
