import { describe, expect, it } from 'vitest'
import { isAvailabilityUnknown } from '../lib/availability'

describe('isAvailabilityUnknown', () => {
  it('is true only for "Unknown", ignoring case and spaces', () => {
    expect(isAvailabilityUnknown('Unknown')).toBe(true)
    expect(isAvailabilityUnknown(' unknown ')).toBe(true)
  })
  it('is false for every other status and for a missing field', () => {
    for (const v of ['Questionable', 'Doubtful', 'Out', 'Healthy', '', null, undefined]) {
      expect(isAvailabilityUnknown(v)).toBe(false)
    }
  })
})
