import { useEffect } from 'react'
import { pageTitle } from '@/lib/page-names'

/** Sets the browser tab title for the page ("Draft Board · Gridiron Oracle"). Call it once per page. */
export function usePageTitle(name?: string | null) {
  useEffect(() => {
    document.title = pageTitle(name)
  }, [name])
}
