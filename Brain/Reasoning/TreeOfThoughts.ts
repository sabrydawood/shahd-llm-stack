// Tree of Thoughts (Phase 7): instead of one linear chain, explore several reasoning branches and
// keep the most promising. This is a beam search over injected Expand (propose next thoughts from a
// partial path) and Score (value a partial path), keeping the top-Beam paths at each depth. Pure
// over the injected functions, so it's deterministic and testable without a trained model; a real
// deployment supplies model-backed Expand/Score.

export type Thought = { Path: string[]; Score: number };
export type ExpandFn = (Path: string[]) => string[];
export type ScoreFn = (Path: string[]) => number;
export type TreeOptions = { Beam: number; Depth: number };

export function TreeOfThoughtsSearch(
  Root: string[],
  Expand: ExpandFn,
  Score: ScoreFn,
  Options: TreeOptions,
): Thought {
  let Best: Thought = { Path: Root, Score: Score(Root) };
  let Frontier: Thought[] = [Best];

  for (let Depth = 0; Depth < Options.Depth; Depth++) {
    const Children: Thought[] = [];
    for (const Node of Frontier) {
      for (const Next of Expand(Node.Path)) {
        const Path = [...Node.Path, Next];
        Children.push({ Path, Score: Score(Path) });
      }
    }
    if (Children.length === 0) break;
    Children.sort((A, B) => B.Score - A.Score);
    Frontier = Children.slice(0, Math.max(1, Options.Beam));
    if (Frontier.length > 0 && Frontier[0].Score > Best.Score) Best = Frontier[0];
  }
  return Best;
}
