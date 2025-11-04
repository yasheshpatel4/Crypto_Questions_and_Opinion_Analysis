import { Suspense } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ClassifierForm } from "@/components/classifier-form"
import { HierarchyTree } from "@/components/hierarchy-tree"
import { AnswerGraph } from "@/components/answer-graph"

export default function Page() {
  return (
    <main className="min-h-dvh">
      <header className="border-b">
        <div className="mx-auto max-w-6xl px-4 py-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-md bg-primary/10 flex items-center justify-center">
              <span className="text-primary font-mono text-sm">Ξ</span>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-pretty">Crypto Opinion & Intent Analyzer</h1>
              <p className="text-muted-foreground text-sm">Classify crypto tweets/reddits into a clear hierarchy</p>
            </div>
          </div>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-4 py-8 space-y-6">
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-balance">Analyze a sentence</CardTitle>
              <CardDescription>Paste a crypto-related sentence to see where it falls in the hierarchy.</CardDescription>
            </CardHeader>
            <CardContent>
              <ClassifierForm />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-balance">Classification hierarchy</CardTitle>
              <CardDescription>The complete hierarchy structure used for classification.</CardDescription>
            </CardHeader>
            <CardContent>
              <Suspense>
                <HierarchyTree />
              </Suspense>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-balance">Answer graph</CardTitle>
            <CardDescription>
              Dynamic bar graphs showing the predicted classes at each level.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Suspense>
              <AnswerGraph />
            </Suspense>
          </CardContent>
        </Card>
      </section>

      <footer className="border-t">
        <div className="mx-auto max-w-6xl px-4 py-6 text-sm text-muted-foreground">
          Built for crypto-related questions and opinion analysis. No financial advice.
        </div>
      </footer>
    </main>
  )
}
