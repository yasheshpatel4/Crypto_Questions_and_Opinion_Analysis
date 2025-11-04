"use client"

import type * as React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

type Result = {
  path: string
  level1: "noise" | "objective" | "subjective" | null
  level2?: "neutral" | "negative" | "positive"
  level3?: "neutral_sentiments" | "questions" | "advertisements" | "misc"
}

export function ClassifierForm() {
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setResult(null)
    if (!text.trim()) {
      setError("Please enter a sentence.")
      return
    }
    setLoading(true)
    try {
      const res = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`)
      }
      const data = await res.json()
      const path: string = data.path || ""
      const [l1, l2, l3] = path.split("->")
      const parsed: Result = {
        path,
        level1: (l1 as Result["level1"]) ?? null,
        level2: l2 as Result["level2"],
        level3: l3 as Result["level3"],
      }
      setResult(parsed)
      window.dispatchEvent(new CustomEvent("classification:update", { detail: parsed }))
    } catch (err: any) {
      setError(err?.message ?? "Something went wrong.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={onSubmit} className="grid gap-4">
      <div className="grid gap-2">
        <label htmlFor="sentence" className="text-sm font-medium">
          Sentence
        </label>
        <Textarea
          id="sentence"
          placeholder="e.g. Should I buy BTC before the halving or is this a bull trap?"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          className="resize-y"
        />
      </div>

      <div className="flex items-center justify-between gap-3">
        <div className="text-xs text-muted-foreground">
          Model path format: noise | objective | subjective&gt;neutral|negative|positive [&gt;
          neutral_sentiments|questions|advertisements|misc]
        </div>
        <Button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </Button>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-4 text-destructive text-sm">{error}</CardContent>
        </Card>
      )}

      {result && (
        <Card>
          <CardContent className="pt-4">
            <div className="text-sm text-muted-foreground">Predicted path</div>
            <div className="mt-1 font-mono text-sm">{result.path}</div>

            <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
              <Tag label="Level 1" value={result.level1 ?? "-"} />
              <Tag label="Level 2" value={result.level2 ?? "-"} />
              <Tag label="Level 3" value={result.level3 ?? "-"} />
            </div>
          </CardContent>
        </Card>
      )}
    </form>
  )
}

function Tag({ label, value }: { label: string; value: string }) {
  const isFilled = value !== "-" && value !== ""
  return (
    <div
      className={cn(
        "rounded-md border px-2 py-1 flex items-center justify-between",
        isFilled ? "bg-primary/5 border-primary text-primary" : "text-muted-foreground",
      )}
      aria-live="polite"
    >
      <span>{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  )
}
