import { useState } from "react";
import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { Fingerprint, KeyRound, Lock, ShieldCheck, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Secure Portal — FWA Risk Investigator" },
      { name: "description", content: "Authenticate to access the FWA investigation workspace." },
      { property: "og:title", content: "Secure Portal — FWA Risk Investigator" },
      { property: "og:description", content: "Authenticate to access the FWA investigation workspace." },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ username: "", userId: "", password: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.username || !form.userId || !form.password) {
      setError("All credential fields are required.");
      return;
    }
    setError("");
    setBusy(true);
    setTimeout(() => navigate({ to: "/portal" }), 650);
  }

  return (
    <div className="gradient-mesh relative flex min-h-screen items-center justify-center overflow-hidden px-4 py-12">
      <div className="grid-lines animate-drift pointer-events-none absolute -inset-40 opacity-60" />

      <div className="relative z-10 w-full max-w-md">
        <Link to="/" className="mb-6 flex items-center justify-center gap-2.5">
          <span className="gradient-navy flex size-9 items-center justify-center rounded-xl text-navy-foreground shadow-float">
            <ShieldCheck className="size-5" />
          </span>
          <span className="text-[11px] font-semibold tracking-[0.22em] text-foreground">
            FWA RISK INVESTIGATOR
          </span>
        </Link>

        <div className="surface-panel animate-rise p-8 shadow-lift">
          <div className="text-center">
            <span className="inline-flex size-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <Lock className="size-6" />
            </span>
            <h1 className="text-gradient-brand mt-4 text-2xl font-semibold tracking-[0.14em]">SECURE PORTAL</h1>
            <p className="mt-2 text-xs text-muted-foreground">
              Authorized payment-integrity personnel only. Activity is audited.
            </p>
          </div>

          <form onSubmit={submit} className="mt-7 space-y-4">
            <Field
              id="username"
              label="Username"
              icon={User}
              value={form.username}
              placeholder="j.mercer"
              onChange={(v) => setForm({ ...form, username: v })}
            />
            <Field
              id="userId"
              label="User ID"
              icon={Fingerprint}
              value={form.userId}
              placeholder="INV-48211"
              onChange={(v) => setForm({ ...form, userId: v })}
            />
            <Field
              id="password"
              label="Password"
              icon={KeyRound}
              type="password"
              value={form.password}
              placeholder="••••••••"
              onChange={(v) => setForm({ ...form, password: v })}
            />

            {error ? <p className="text-xs font-medium text-destructive">{error}</p> : null}

            <Button
              type="submit"
              disabled={busy}
              className="w-full rounded-full py-6 text-sm font-semibold tracking-[0.14em] uppercase shadow-float transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lift"
            >
              {busy ? "Authenticating…" : "Authenticate"}
            </Button>
          </form>
        </div>

        <p className="mt-5 text-center text-[11px] text-muted-foreground">
          Demo environment — any credential set is accepted for evaluation.
        </p>
      </div>
    </div>
  );
}

function Field({
  id,
  label,
  icon: Icon,
  value,
  onChange,
  type = "text",
  placeholder,
}: {
  id: string;
  label: string;
  icon: React.ElementType;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  placeholder?: string;
}) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </Label>
      <div className="relative">
        <Icon className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          id={id}
          type={type}
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          className="h-11 rounded-xl pl-9"
        />
      </div>
    </div>
  );
}
