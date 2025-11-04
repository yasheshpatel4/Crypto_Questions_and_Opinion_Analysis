"use client"

import { useEffect, useMemo, useState } from "react"
import { Bar, BarChart, CartesianGrid, Legend, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

type L1 = "noise" | "objective" | "subjective" | null
type L2 = "neutral" | "negative" | "positive" | undefined
type L3 = "neutral_sentiments" | "questions" | "advertisements" | "misc" | undefined

type Detail = {
  level1: L1
  level2?: L2
  level3?: L3
}

const allKeys = {
  // Level 1
  noise: "Noise",
  objective: "Objective",
  subjective: "Subjective",
  // Level 2
  neutral: "Neutral",
  negative: "Negative",
  positive: "Positive",
  // Level 3 (only under Neutral)
  neutral_sentiments: "Neutral Sentiments",
  questions: "Questions",
  advertisements: "Advertisements",
  misc: "Miscellaneous",
} as const

export function AnswerGraph() {
  const [detail, setDetail] = useState<Detail>({ level1: null })

  useEffect(() => {
    function onUpdate(e: any) {
      setDetail({
        level1: e.detail.level1 ?? null,
        level2: e.detail.level2,
        level3: e.detail.level3,
      })
    }
    window.addEventListener("classification:update", onUpdate as any)
    return () => window.removeEventListener("classification:update", onUpdate as any)
  }, [])

  const { data, bars } = useMemo(() => {
    const baseData: any[] = [
      {
        group: "Level 1",
        noise: detail.level1 === "noise" ? 1 : 0,
        objective: detail.level1 === "objective" ? 1 : 0,
        subjective: detail.level1 === "subjective" ? 1 : 0,
      },
    ]
    let bars = [
      <Bar key="noise" dataKey="noise" name="noise" fill="var(--color-noise)" />,
      <Bar key="objective" dataKey="objective" name="objective" fill="var(--color-objective)" />,
      <Bar key="subjective" dataKey="subjective" name="subjective" fill="var(--color-subjective)" />,
    ]

    if (detail.level1 === "subjective") {
      baseData.push({
        group: "Level 2",
        neutral: detail.level2 === "neutral" ? 1 : 0,
        negative: detail.level2 === "negative" ? 1 : 0,
        positive: detail.level2 === "positive" ? 1 : 0,
      })
      bars.push(
        <Bar key="neutral" dataKey="neutral" name="neutral" fill="var(--color-neutral)" />,
        <Bar key="negative" dataKey="negative" name="negative" fill="var(--color-negative)" />,
        <Bar key="positive" dataKey="positive" name="positive" fill="var(--color-positive)" />
      )
    }

    if (detail.level1 === "subjective" && detail.level2 === "neutral") {
      baseData.push({
        group: "Level 3",
        neutral_sentiments: detail.level3 === "neutral_sentiments" ? 1 : 0,
        questions: detail.level3 === "questions" ? 1 : 0,
        advertisements: detail.level3 === "advertisements" ? 1 : 0,
        misc: detail.level3 === "misc" ? 1 : 0,
      })
      bars.push(
        <Bar key="neutral_sentiments" dataKey="neutral_sentiments" name="neutral_sentiments" fill="var(--color-neutral_sentiments)" />,
        <Bar key="questions" dataKey="questions" name="questions" fill="var(--color-questions)" />,
        <Bar key="advertisements" dataKey="advertisements" name="advertisements" fill="var(--color-advertisements)" />,
        <Bar key="misc" dataKey="misc" name="misc" fill="var(--color-misc)" />
      )
    }

    return { data: baseData, bars }
  }, [detail])

  // Map labels and colors for legend + theme-injected CSS variables
  const config = {
    noise: { label: allKeys.noise, color: "hsl(var(--chart-3))" },
    objective: { label: allKeys.objective, color: "hsl(var(--chart-2))" },
    subjective: { label: allKeys.subjective, color: "hsl(var(--chart-1))" },
    neutral: { label: allKeys.neutral, color: "hsl(var(--chart-2))" },
    negative: { label: allKeys.negative, color: "hsl(var(--chart-5))" },
    positive: { label: allKeys.positive, color: "hsl(var(--chart-4))" },
    neutral_sentiments: { label: allKeys.neutral_sentiments, color: "hsl(var(--chart-2))" },
    questions: { label: allKeys.questions, color: "hsl(var(--chart-1))" },
    advertisements: { label: allKeys.advertisements, color: "hsl(var(--chart-5))" },
    misc: { label: allKeys.misc, color: "hsl(var(--chart-3))" },
  }

  return (
    <ChartContainer config={config} className="h-[280px] w-full">
      <BarChart data={data} margin={{ top: 8, right: 12, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="group" />
        <YAxis domain={[0, 1]} ticks={[0, 1]} />
        <ChartTooltip
          content={
            <ChartTooltipContent
              labelKey="group"
              formatter={(value: any, name: any) => {
                const v = typeof value === "number" ? value : 0
                return (
                  <div className="flex items-center gap-2">
                    <span>{config[name as keyof typeof config]?.label || name}</span>
                    <span className="font-mono">{v}</span>
                  </div>
                )
              }}
            />
          }
        />
        {bars}
      </BarChart>
    </ChartContainer>
  )
}
