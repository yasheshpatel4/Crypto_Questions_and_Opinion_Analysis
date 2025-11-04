// Static hierarchy tree showing the complete classification structure.
// This displays the full hierarchy without dynamic highlighting.

export function HierarchyTree() {
  return (
    <div className="flex flex-col items-center space-y-4 p-4 bg-card rounded-lg border">
      {/* Root */}
      <div className="text-center">
        <div className="bg-primary text-primary-foreground px-4 py-2 rounded-md font-semibold">
          Tweets / Reddits
        </div>
      </div>

      {/* Level 1 */}
      <div className="flex items-center space-x-8">
        <div className="text-center">
          <div className="w-8 h-1 bg-primary mb-2"></div>
          <Node label="Noise" />
        </div>
        <div className="text-center">
          <div className="w-8 h-1 bg-primary mb-2"></div>
          <Node label="Objective" />
        </div>
        <div className="text-center">
          <div className="w-8 h-1 bg-primary mb-2"></div>
          <Node label="Subjective" />
        </div>
      </div>

      {/* Level 2 */}
      <div className="flex items-center space-x-8">
        <div className="text-center">
          <div className="w-8 h-1 bg-primary mb-2"></div>
          <Node label="Neutral" />
          {/* Level 3 */}
          <div className="mt-4 space-y-2">
            <div className="w-4 h-1 bg-primary mx-auto mb-2"></div>
            <div className="grid grid-cols-2 gap-2">
              <Node label="Neutral Sentiments" small />
              <Node label="Questions" small />
              <Node label="Advertisements" small />
              <Node label="Miscellaneous" small />
            </div>
          </div>
        </div>
        <div className="text-center">
          <div className="w-8 h-1 bg-primary mb-2"></div>
          <Node label="Negative" />
        </div>
        <div className="text-center">
          <div className="w-8 h-1 bg-primary mb-2"></div>
          <Node label="Positive" />
        </div>
      </div>
    </div>
  )
}

function Node({ label, small }: { label: string; small?: boolean }) {
  return (
    <div
      className={`border-2 px-3 py-2 rounded-md font-medium border-muted bg-background text-muted-foreground ${
        small ? "text-xs px-2 py-1" : "text-sm"
      }`}
    >
      {label}
    </div>
  )
}
