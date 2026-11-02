// BCA — React 18 components, global window.BCA. Each renders the reference page's own markup and classes
// (components/bundle.css); colors come from tokens.css.
import type { ReactNode, ButtonHTMLAttributes } from 'react';

/** A test situation. */
export type Sit = 'recorded' | 'strong_current' | 'big_waves' | 'big_kicks';
/** A world. */
export type World = 'stall' | 'no_stall';
/** Any other colored thing. */
export type Thing = 'region' | 'stall' | 'learn' | 'hold' | 'recording' | 'stall_aware' | 'rising_thrust' | 'no_throttle';
/** A pipeline step. */
export type Step = 'collect' | 'remove' | 'train' | 'band' | 'test';
/** Names one colored thing: give exactly one of sit, world or k. */
export interface ThingRef { sit?: Sit; world?: World; k?: Thing; }
/** A color token's name, without the leading --, e.g. "k-stall". */
export type ColorToken = string;

// ---------------------------------------------------------------- Button
export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** "primary" (default): the accent fill, one per row. "ghost": outlined, for the rest. */
  variant?: 'primary' | 'ghost';
  children: ReactNode;
}
/** Play, Step and Reset buttons, a range input and a mono time label for a live figure. */
export interface TransportProps {
  /** Current time step (controlled). */
  t?: number;
  defaultT?: number;
  /** Last time step. Default 99 (a 100-step episode). */
  max?: number;
  onChange?: (t: number) => void;
  /** Milliseconds per step while playing. Default 120. */
  stepMs?: number;
  /** The label beside the slider. Default "t = 37". */
  label?: (t: number) => string;
  ariaLabel?: string;
}

// ---------------------------------------------------------------- Option
export interface OptionProps {
  /** Which kind of thing the option names; sets its key shape (sit: dot, world: line, ctrl: swatch line). */
  group: 'sit' | 'world' | 'ctrl';
  /** The thing: a Sit for "sit", a World for "world", or recording | stall_aware | rising_thrust | no_throttle for "ctrl". */
  value: string;
  /** A few words. */
  name: ReactNode;
  /** One sentence. */
  description?: ReactNode;
  selected?: boolean;
  onSelect?: () => void;
  /** Draw the controller's swatch dashed (no throttle). */
  dash?: boolean;
  /** One line on the pipeline step the option stands for; usually starts with a StepChip. */
  step?: ReactNode;
}
/** A numbered question with its options in a grid; one option is pressed. */
export interface OptionGroupProps {
  n?: number;
  question: ReactNode;
  hint?: ReactNode;
  group: OptionProps['group'];
  options: Array<Omit<OptionProps, 'group' | 'selected' | 'onSelect'>>;
  value?: string;
  defaultValue?: string;
  onChange?: (value: string) => void;
}
export interface PanelProps { title?: ReactNode; children: ReactNode; }
export interface KeyValueProps {
  items: Array<{ label: ReactNode; value: ReactNode }>;
  /** Each label over its text (kv1) instead of two columns. */
  stacked?: boolean;
  /** Labels in accent-ink (the Finding's form). */
  pro?: boolean;
}

// ---------------------------------------------------------------- Tag
export interface TagProps {
  /** ok: pass. acc: exploratory or a note. warn: the stall and real warnings only. None: the text color. */
  variant?: 'ok' | 'acc' | 'warn';
  /** One or two lowercase words. */
  children: ReactNode;
}
export interface StatusListProps { items: Array<{ tag: ReactNode; variant?: TagProps['variant']; text: ReactNode }>; }

// ---------------------------------------------------------------- Key
export interface KeyProps extends ThingRef {
  /** dot: a situation · line: a world, controller or learned part · dash: no throttle · dots: the throttle cost ·
   *  tile: a region · removed: data removed · ring: a float or no signal · sitall: the situation you picked. */
  shape?: 'dot' | 'line' | 'dash' | 'dots' | 'tile' | 'removed' | 'ring' | 'sitall';
}
/** A colored mention in running text: the word stays in ink, the mark sits before it. */
export interface MentionProps extends ThingRef { children: ReactNode; }
export interface LegendItem extends KeyProps {
  label: ReactNode;
  /** Draw a 10px dot in this color token instead of a key. */
  swatch?: ColorToken;
  /** Draw a hollow ring (no signal, a starting point). */
  ring?: boolean;
  /** Draw a short line in this color token. */
  line?: ColorToken;
}
export interface LegendProps { items: LegendItem[]; }

// ---------------------------------------------------------------- Callout
export interface CalloutProps {
  /** warn (default): a warning on warn-soft. box: a neutral aside. stop: a deciding rule, neutral with an ink edge. */
  variant?: 'warn' | 'box' | 'stop';
  /** warn only: the bold opening words. */
  lead?: ReactNode;
  /** box and stop: the heading. */
  title?: ReactNode;
  children: ReactNode;
  role?: string;
}

// ---------------------------------------------------------------- Finding
export interface FindingProps {
  /** The finding as a sentence a reader could repeat. */
  title: ReactNode;
  /** The idea · The test and its rule · Verdict · What the data show · Open questions. The verdict is the
   *  predefined test's outcome under its own rule and nothing else; the reading and the open questions are
   *  written whatever the verdict. */
  parts: Array<{ label: ReactNode; text: ReactNode }>;
  /** stacked (default): each label over its text. columns: label left, text right. */
  layout?: 'stacked' | 'columns';
}

// ---------------------------------------------------------------- SymbolLegend
export interface SymbolLegendProps {
  /** Default "What the symbols mean". */
  heading?: ReactNode;
  groups: Array<{
    /** A group heading after the first group, e.g. "Also new in this step". */
    title?: ReactNode;
    items: Array<{ symbol: ReactNode; meaning: ReactNode; key?: KeyProps }>;
  }>;
  /** The page's own numbers run through the symbols. */
  example?: ReactNode;
  /** Wrap it in a details element with the heading as its summary. */
  collapsible?: boolean;
  defaultOpen?: boolean;
}

// ---------------------------------------------------------------- Table
export interface TableColumn {
  key: string;
  label: ReactNode;
  /** Right-aligned mono cells. */
  numeric?: boolean;
  /** What the column means: shown in the shared tooltip, and the header gets a dotted underline. */
  desc?: ReactNode;
}
export type TableRow =
  | ({ group: ReactNode } & ThingRef)
  | { cells: Record<string, ReactNode>; highlight?: boolean; anchor?: boolean };
export interface TableProps {
  /** How to read the table, with the scale. */
  caption?: ReactNode;
  columns: TableColumn[];
  /** A group row ({group, sit}) takes the situation's tint and edge; highlight: warn-soft (a row about the stall); anchor: the reference row. */
  rows: TableRow[];
  className?: string;
}

// ---------------------------------------------------------------- Readout
export interface ReadoutProps {
  items: Array<{
    /** Often a sentence; shown small in ink-2. */
    name: ReactNode;
    value: ReactNode;
    /** stall: warn, bold. region: k-region-ink. */
    tone?: 'stall' | 'region';
  }>;
  /** Announce changes to screen readers (leave off for values that change every step). */
  live?: boolean;
}

// ---------------------------------------------------------------- DotRange
export interface DotRangeRow {
  name: ReactNode;
  /** Plain-text name for the tooltip and screen readers when name is not a string. */
  label?: string;
  /** Mean over seeds: the dot. */
  mean: number;
  /** Lowest and highest seed: the line. */
  lo: number;
  hi: number;
  /** hi: an arm the chart is about (dr-hi). lo: the rest (dr-lo). ref: a starting point or no signal (hollow, ink-2). */
  kind?: 'hi' | 'lo' | 'ref';
  /** Set the name in bold (the main arm). */
  strong?: boolean;
  /** Plain text after the name, e.g. "(main arm)". */
  note?: string;
}
export interface DotRangeProps {
  min: number;
  max: number;
  ticks: number[];
  rows: Array<DotRangeRow | { group: ReactNode }>;
  /** Names the scale and its ends. */
  axisLabel?: ReactNode;
  nameHeading?: ReactNode;
  valueHeading?: ReactNode;
  legend?: LegendItem[];
  caption?: ReactNode;
  format?: (v: number) => string;
  tickFormat?: (v: number) => string;
}

// ---------------------------------------------------------------- Chart
export interface ChartAxis {
  min: number;
  max: number;
  ticks?: number[];
  label?: string;
  /** Values in the reading line and, unless tickFormat is given, the ticks. */
  format?: (v: number) => string;
  tickFormat?: (v: number) => string;
  /** The x value at the start of the reading line. */
  readFormat?: (v: number) => string;
}
export interface ChartSeries {
  id: string;
  label: string;
  points: Array<[number, number]>;
  /** A line class from bundle.css (ln-stall, ln-nostall, ln-dose, ln-cost, ln-sa, ln-rt, ln-cap2, ln-ref, ln-rawq, ln-bca). */
  className?: string;
  /** The color token of its dot and reading swatch (and of the line when no className). */
  color?: ColorToken;
  format?: (v: number) => string;
}
export interface ChartProps {
  x: ChartAxis;
  y: ChartAxis;
  /** Drawn in order: the last on top. */
  series: ChartSeries[];
  /** Vertical (x) or horizontal (y) guide lines: knee, knee k80, k565. */
  guides?: Array<{ x?: number; y?: number; className?: string }>;
  /** Points: pt-stall, pt-nostall, pt-dose, pt-ft, pt-normal, pt-removed… */
  marks?: Array<{ x: number; y: number; className: string; r?: number }>;
  /** Direct labels in data coordinates: className lab (default, on a surface halo) or ann. */
  labels?: Array<{ x: number; y: number; text: string; anchor?: 'start' | 'middle' | 'end'; className?: string }>;
  legend?: LegendItem[];
  caption?: ReactNode;
  /** The reading step along x (pointer and arrow keys). Default 1/100 of the range. */
  step?: number;
  /** The reading line before the reader points. */
  hint?: ReactNode;
  width?: number;
  height?: number;
  margin?: { l?: number; r?: number; t?: number; b?: number };
  wide?: boolean;
  ariaLabel?: string;
}

// ---------------------------------------------------------------- Tooltip
/** A term that explains itself: a dotted underline, and the shared tooltip on hover or focus (Escape closes it). */
export interface TooltipProps {
  /** The term, as the tooltip's heading. */
  title?: ReactNode;
  /** A definition in one or two sentences, starting with the term. */
  text: ReactNode;
  /** A second part under a rule, e.g. where the term is used. */
  more?: ReactNode;
  /** The term in the running text. */
  children: ReactNode;
  /** The element that carries the term. Default "i". */
  as?: string;
}
export interface TooltipBoxProps { title?: ReactNode; text?: ReactNode; more?: ReactNode; }

// ---------------------------------------------------------------- PipelineStrip
export interface PipelineStep {
  step: Step;
  title: ReactNode;
  /** The compact strip's label. Default: title. */
  short?: ReactNode;
  /** One or more paragraphs. */
  text?: ReactNode | ReactNode[];
  why?: ReactNode;
  href?: string;
  /** The "go to" link's words. */
  go?: ReactNode;
}
export interface PipelineStripProps {
  steps: PipelineStep[];
  /** One row of pills instead of five cards. */
  compact?: boolean;
  /** Compact only: the steps the block stands for, outlined. */
  here?: Step[];
  defaultHere?: Step[];
  onHereChange?: (here: Step[]) => void;
  /** Compact only: clicking a step selects it (and does not follow its link). */
  selectable?: boolean;
  onSelect?: (step: Step) => void;
  ariaLabel?: string;
}
export interface StepChipProps { step: Step | 'none'; inline?: boolean; children?: ReactNode; }

// ---------------------------------------------------------------- DeckSlide
export interface DeckSlideProps {
  /** navy (default): cover, part and question slides. paper: content slides. */
  kind?: 'navy' | 'paper';
  /** The uppercase line above the title, its parts joined by " · ". */
  kicker?: ReactNode;
  title: ReactNode;
  /** navy: the line above the title (the last answer, or the cover's lead). */
  lead?: ReactNode;
  /** navy: the line under the title. paper: a line under the title in deck-soft. */
  body?: ReactNode;
  /** navy: the row of question dots. */
  progress?: { current: number; total: number };
  /** navy: the cover's sizes and its orange rule. */
  cover?: boolean;
  /** paper: what is not yet shown, in deck-caution. */
  caution?: ReactNode;
  /** At the slide's foot. */
  footnote?: ReactNode;
  /** paper: the slide's content (tables, figures, boxes). */
  children?: ReactNode;
  /** The 1920 by 1080 slide is drawn at this scale. Default 0.25. */
  scale?: number;
}

declare const BCA: {
  Button: (p: ButtonProps) => JSX.Element; Transport: (p: TransportProps) => JSX.Element;
  Option: (p: OptionProps) => JSX.Element; OptionGroup: (p: OptionGroupProps) => JSX.Element;
  Panel: (p: PanelProps) => JSX.Element; KeyValue: (p: KeyValueProps) => JSX.Element;
  Tag: (p: TagProps) => JSX.Element; StatusList: (p: StatusListProps) => JSX.Element;
  Key: (p: KeyProps) => JSX.Element; Mention: (p: MentionProps) => JSX.Element; Legend: (p: LegendProps) => JSX.Element;
  Callout: (p: CalloutProps) => JSX.Element; Finding: (p: FindingProps) => JSX.Element;
  SymbolLegend: (p: SymbolLegendProps) => JSX.Element;
  Table: (p: TableProps) => JSX.Element; Readout: (p: ReadoutProps) => JSX.Element;
  DotRange: (p: DotRangeProps) => JSX.Element; Chart: (p: ChartProps) => JSX.Element;
  Tooltip: (p: TooltipProps) => JSX.Element; TooltipBox: (p: TooltipBoxProps) => JSX.Element;
  PipelineStrip: (p: PipelineStripProps) => JSX.Element; StepChip: (p: StepChipProps) => JSX.Element;
  DeckSlide: (p: DeckSlideProps) => JSX.Element;
  format: { num: (v: number, decimals?: number) => string };
};
export default BCA;
