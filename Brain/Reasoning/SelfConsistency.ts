// Self-consistency (Phase 7): sample N candidate answers, extract the final answer from each, and
// return the majority-voted one — robust to a single unlucky/incoherent sample. MajorityVote is a
// pure, deterministic utility (ties broken by first appearance); SelfConsistency wraps it over an
// injected generator + answer extractor so it's testable without a trained model.

export type VoteResult<T> = {
  Winner: T;
  Count: number; // votes for the winner
  Total: number; // total votes
  Tally: [string, number][]; // per-answer counts, in first-seen order
};

export function MajorityVote<T>(Items: T[], KeyOf: (Item: T) => string): VoteResult<T> {
  if (Items.length === 0) throw new Error("MajorityVote: no items to vote on");
  const Counts = new Map<string, number>();
  const FirstSeen = new Map<string, T>();
  for (const Item of Items) {
    const Key = KeyOf(Item);
    Counts.set(Key, (Counts.get(Key) ?? 0) + 1);
    if (!FirstSeen.has(Key)) FirstSeen.set(Key, Item);
  }
  let BestKey = "";
  let BestCount = -1;
  for (const [Key, Count] of Counts) {
    if (Count > BestCount) {
      BestCount = Count;
      BestKey = Key;
    }
  }
  const Winner = FirstSeen.get(BestKey);
  if (Winner === undefined) throw new Error("MajorityVote: winner lookup failed");
  return { Winner, Count: BestCount, Total: Items.length, Tally: [...Counts.entries()] };
}

/** Draw `Samples` generations, extract each answer, and majority-vote over them. */
export function SelfConsistency(
  Sample: () => string,
  Samples: number,
  ExtractAnswer: (Text: string) => string,
): VoteResult<string> {
  const Answers: string[] = [];
  for (let I = 0; I < Samples; I++) Answers.push(ExtractAnswer(Sample()));
  return MajorityVote(Answers, (A) => A);
}
