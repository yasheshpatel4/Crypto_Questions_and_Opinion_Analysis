import { type NextRequest, NextResponse } from "next/server"
import { classifyHeuristic, type ClassificationPath } from "@/lib/classifier"

export async function POST(req: NextRequest) {
  try {
    const { text } = await req.json()
    if (typeof text !== "string" || !text.trim()) {
      return NextResponse.json({ error: "Invalid text" }, { status: 400 })
    }

    const external = process.env.CLASSIFIER_API_URL
    if (external) {
      // Expect external endpoint to return: { path: string }
      const r = await fetch(external, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })
      if (r.ok) {
        const data = await r.json()
        if (typeof data?.path === "string") {
          return NextResponse.json({ path: data.path })
        }
      }
      // If external fails, we fall back silently.
    }

    const path: ClassificationPath = classifyHeuristic(text)
    return NextResponse.json({ path })
  } catch (e) {
    return NextResponse.json({ error: "Server error" }, { status: 500 })
  }
}
