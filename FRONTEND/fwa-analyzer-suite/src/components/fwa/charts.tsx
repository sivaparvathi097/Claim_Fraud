import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Series } from "@/lib/fwa";

const AXIS = { stroke: "var(--muted-foreground)", fontSize: 11 };
const TOOLTIP = {
  contentStyle: {
    borderRadius: 12,
    border: "1px solid var(--border)",
    background: "var(--card)",
    fontSize: 12,
    boxShadow: "var(--shadow-float)",
  },
};

export function ChartFrame({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border/70 bg-card p-4">
      <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
        {title}
      </p>
      <div className="h-56 w-full">{children}</div>
    </div>
  );
}

export function BarSeries({ data, color = "var(--chart-1)" }: { data: Series; color?: string }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
    <BarChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
      <XAxis dataKey="name" tickLine={false} axisLine={false} {...AXIS} />
      <YAxis tickLine={false} axisLine={false} {...AXIS} />
      <Tooltip {...TOOLTIP} />
      <Bar dataKey="value" radius={[8, 8, 4, 4]} fill={color} animationDuration={900} />
    </BarChart>
    </ResponsiveContainer>
  );
}

export function RiskDonut({ data }: { data: Series }) {
  const colors = ["var(--risk-low)", "var(--risk-medium)", "var(--risk-high)"];
  return (
    <ResponsiveContainer width="100%" height="100%">
    <PieChart>
      <Tooltip {...TOOLTIP} />
      <Legend wrapperStyle={{ fontSize: 11 }} />
      <Pie data={data} dataKey="value" nameKey="name" innerRadius={52} outerRadius={80} paddingAngle={3}>
        {data.map((_, i) => (
          <Cell key={i} fill={colors[i % colors.length]} stroke="var(--card)" strokeWidth={2} />
        ))}
      </Pie>
    </PieChart>
    </ResponsiveContainer>
  );
}

export function PeerCompare({ data }: { data: { name: string; subject: number; peer: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
    <AreaChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
      <defs>
        <linearGradient id="gSubject" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.45} />
          <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0.03} />
        </linearGradient>
      </defs>
      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
      <XAxis dataKey="name" tickLine={false} axisLine={false} {...AXIS} />
      <YAxis tickLine={false} axisLine={false} {...AXIS} />
      <Tooltip {...TOOLTIP} />
      <Legend wrapperStyle={{ fontSize: 11 }} />
      <Area
        type="monotone"
        dataKey="subject"
        name="Subject"
        stroke="var(--chart-1)"
        fill="url(#gSubject)"
        strokeWidth={2}
      />
      <Line type="monotone" dataKey="peer" name="Peer median" stroke="var(--chart-2)" strokeWidth={2} dot={false} />
    </AreaChart>
    </ResponsiveContainer>
  );
}

export function SignalRadar({ data }: { data: Series }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
    <RadarChart data={data} outerRadius="72%">
      <PolarGrid stroke="var(--border)" />
      <PolarAngleAxis dataKey="name" tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
      <Tooltip {...TOOLTIP} />
      <Radar
        dataKey="value"
        name="Signal strength"
        stroke="var(--chart-2)"
        fill="var(--chart-2)"
        fillOpacity={0.28}
        animationDuration={900}
      />
    </RadarChart>
    </ResponsiveContainer>
  );
}

export function TrendLine({ data }: { data: Series }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
    <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
      <XAxis dataKey="name" tickLine={false} axisLine={false} {...AXIS} />
      <YAxis tickLine={false} axisLine={false} {...AXIS} />
      <Tooltip {...TOOLTIP} />
      <Line
        type="monotone"
        dataKey="value"
        stroke="var(--chart-3)"
        strokeWidth={2.5}
        dot={{ r: 3, fill: "var(--chart-3)" }}
        animationDuration={900}
      />
    </LineChart>
    </ResponsiveContainer>
  );
}
