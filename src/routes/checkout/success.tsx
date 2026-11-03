import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/checkout/success")({
  component: CheckoutSuccess,
});

function CheckoutSuccess() {
  return (
    <main className="mx-auto flex min-h-[60vh] max-w-lg flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-100 text-green-600">
        <svg
          width="32"
          height="32"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M20 6 9 17l-5-5" />
        </svg>
      </div>
      <h1 className="text-2xl font-semibold">Thank you for your purchase!</h1>
      <p className="text-muted-foreground">
        Your order is confirmed — a receipt is on its way to your email.
      </p>
      <a
        href="/"
        className="mt-2 inline-flex items-center rounded-lg bg-primary px-5 py-2.5 font-medium text-primary-foreground transition hover:opacity-90"
      >
        Continue shopping
      </a>
    </main>
  );
}
