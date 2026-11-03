import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { MessageSquareText, Lightbulb } from "lucide-react";

export const Route = createFileRoute("/")({
  component: Index,
});

const suggestedPrompts = [
  "Build a landing page for my app",
  "Create a dashboard with a sidebar navigation",
  "Design a sign-up and login flow",
  "Make a homepage with a hero section and features",
];

function Index() {
  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center relative overflow-hidden">
      {/* Ambient glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-primary/[0.05] blur-[120px]" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 max-w-lg mx-auto px-6 text-center"
      >
        {/* Icon */}
        <div className="mx-auto mb-6 w-14 h-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
          <MessageSquareText className="w-6 h-6 text-primary" />
        </div>

        {/* Message */}
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight mb-3">
          Uh oh, something interrupted the AI
        </h1>
        <p className="text-muted-foreground text-base leading-relaxed mb-8">
          Your project is still there, but the preview couldn't be completed
          due to a temporary technical issue.
        </p>

        {/* Suggested prompts */}
        <div className="text-left rounded-xl border border-border/60 bg-card/50 backdrop-blur-sm p-5">
          <div className="flex items-center gap-2 mb-3">
            <Lightbulb className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">Try prompting</span>
          </div>
          <div className="space-y-2">
            {suggestedPrompts.map((prompt) => (
              <div
                key={prompt}
                className="text-sm text-muted-foreground py-2 px-3 rounded-lg bg-muted/40 border border-border/40"
              >
                "{prompt}"
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
