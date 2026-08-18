import { Link, useRouterState } from "@tanstack/react-router";
import { ShieldCheck, FileSearch, Building2, LayoutDashboard, LogOut } from "lucide-react";
import type { ReactNode } from "react";

const NAV = [
  { to: "/portal", label: "Uncover FWA", icon: LayoutDashboard },
  { to: "/claims", label: "Claims", icon: FileSearch },
  { to: "/providers", label: "Providers", icon: Building2 },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="min-h-screen gradient-mesh">
      <header className="sticky top-0 z-40 border-b border-border/70 bg-background/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-[1500px] items-center gap-4 px-4 sm:px-6">
          <Link to="/portal" className="flex items-center gap-2.5">
            <span className="gradient-navy flex size-9 items-center justify-center rounded-xl text-navy-foreground shadow-float">
              <ShieldCheck className="size-5" />
            </span>
            <span className="hidden leading-tight sm:block">
              <span className="block text-[13px] font-semibold tracking-[0.16em] text-foreground">
                FWA RISK INVESTIGATOR
              </span>
              <span className="block text-[11px] text-muted-foreground">Payment Integrity Suite</span>
            </span>
          </Link>

          <nav className="ml-auto flex items-center gap-1">
            {NAV.map((item) => {
              const active = pathname === item.to || pathname.startsWith(item.to + "/");
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`flex items-center gap-2 rounded-full px-3 py-2 text-[13px] font-medium transition-all duration-300 ${
                    active
                      ? "bg-primary/10 text-primary shadow-inset-hair"
                      : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                  }`}
                >
                  <item.icon className="size-4" />
                  <span className="hidden md:inline">{item.label}</span>
                </Link>
              );
            })}
            <Link
              to="/login"
              className="ml-2 flex items-center gap-2 rounded-full border border-border bg-card px-3 py-2 text-[13px] font-medium text-muted-foreground transition-all duration-300 hover:-translate-y-0.5 hover:text-foreground hover:shadow-float"
            >
              <LogOut className="size-4" />
              <span className="hidden lg:inline">Sign out</span>
            </Link>
          </nav>
        </div>
      </header>

      <main key={pathname} className="mx-auto max-w-[1500px] animate-fade px-4 pb-20 pt-8 sm:px-6">
        {children}
      </main>
    </div>
  );
}
