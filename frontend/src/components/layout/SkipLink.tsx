/** The first tab stop on every page: jumps past the nav to the page content (PageShell's <main>). */
export function SkipLink() {
  return (
    <a
      href="#main-content"
      className="skip-link"
      onClick={(event) => {
        // A hash link would add a history entry and can fight the router; move focus directly.
        event.preventDefault()
        const main = document.getElementById('main-content')
        main?.focus()
        main?.scrollIntoView?.()
      }}
    >
      Skip to main content
    </a>
  )
}
