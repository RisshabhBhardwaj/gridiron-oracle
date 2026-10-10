import '@testing-library/jest-dom'
import { server } from './handlers'
import { beforeAll, afterEach, afterAll, vi } from 'vitest'

  // jsdom doesn't implement ResizeObserver; polyfill for recharts ResponsiveContainer
  ; (global as any).ResizeObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))

// jsdom has no PointerEvent. Base UI (Select, Dialog, Slider) listens for pointer
// events, so give it a minimal one built on MouseEvent.
if (typeof window.PointerEvent === 'undefined') {
  class PointerEventPolyfill extends MouseEvent {
    pointerId: number
    pointerType: string
    constructor(type: string, init: PointerEventInit = {}) {
      super(type, init)
      this.pointerId = init.pointerId ?? 1
      this.pointerType = init.pointerType ?? 'mouse'
    }
  }
  ; (window as any).PointerEvent = PointerEventPolyfill
}
if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = vi.fn()

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
