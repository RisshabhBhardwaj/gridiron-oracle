/** A working localStorage for the test, whatever the test environment provides. */
export function installMemoryStorage(): Storage {
  const data = new Map<string, string>()
  const storage = {
    getItem: (k: string) => (data.has(k) ? data.get(k)! : null),
    setItem: (k: string, v: string) => { data.set(k, String(v)) },
    removeItem: (k: string) => { data.delete(k) },
    clear: () => data.clear(),
    key: (i: number) => [...data.keys()][i] ?? null,
    get length() { return data.size },
  } as Storage
  Object.defineProperty(window, 'localStorage', { value: storage, configurable: true, writable: true })
  return storage
}

