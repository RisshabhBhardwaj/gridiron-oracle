import '@testing-library/jest-dom'
import { server } from './handlers'
import { beforeAll, afterEach, afterAll, vi } from 'vitest'

  // jsdom doesn't implement ResizeObserver; polyfill for recharts ResponsiveContainer
  ; (global as any).ResizeObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
