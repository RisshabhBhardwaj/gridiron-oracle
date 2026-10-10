import { Slider as SliderPrimitive } from '@base-ui/react/slider'
import { cn } from '@/lib/utils'
import { toneClass, type Tone } from '@/lib/tokens'

/**
 * Single-value slider on Base UI. Keyboard (arrows, Home, End, Page Up/Down),
 * touch and screen readers come from the primitive; the thumb holds a real
 * <input type="range"> for assistive technology.
 *
 * Copied from the shadcn/ui Base UI slider and adapted to this project's
 * Tailwind 3 and colour tokens (docs/decisions/0004). Range sliders with two
 * thumbs are not needed yet; add a second Thumb here when they are.
 */
type SliderProps = Omit<SliderPrimitive.Root.Props<number>, 'aria-label'> & {
  /** Colour of the fill and the thumb. */
  tone?: Tone
  /** Accessible name. Required when there is no visible label. */
  'aria-label'?: string
}

function Slider({ className, tone = 'accent', 'aria-label': ariaLabel, ...props }: SliderProps) {
  return (
    <SliderPrimitive.Root
      data-slot="slider"
      thumbAlignment="edge"
      className={cn(toneClass(tone), 'w-full', className)}
      {...props}
    >
      <SliderPrimitive.Control className="relative flex w-full touch-none items-center py-2 select-none data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50">
        <SliderPrimitive.Track
          data-slot="slider-track"
          className="relative h-2 w-full rounded-full bg-surface-overlay/80 select-none"
        >
          <SliderPrimitive.Indicator
            data-slot="slider-range"
            className="rounded-full bg-tone/40 select-none"
          />
          <SliderPrimitive.Thumb
            data-slot="slider-thumb"
            aria-label={ariaLabel}
            className={cn(
              'block size-4 rounded-full border-2 border-surface bg-tone select-none',
              'shadow-[0_0_8px_rgb(var(--tone)/0.5)] transition-shadow motion-reduce:transition-none',
              // Bigger hit area than the visible thumb, for fingers.
              'after:absolute after:-inset-2',
              'hover:shadow-[0_0_14px_rgb(var(--tone)/0.8)]',
              // The focusable element is the <input> inside the thumb.
              'has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-fg',
            )}
          />
        </SliderPrimitive.Track>
      </SliderPrimitive.Control>
    </SliderPrimitive.Root>
  )
}

export { Slider }
